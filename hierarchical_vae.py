"""VAE functions."""

from models import ActionEncoder, ActionDecoder, Skill_prior
from models import EncoderTransformer, DecoderTransformer
from utils import hyper_params
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.stateless import functional_call
import torch
import wandb
import os
import numpy as np
from utils import GD_full_update, params_extraction
import pdb



class VAE(hyper_params):
    def __init__(self, config):
        super().__init__(config)

        self.models = {}
        self.names = {}

        self.ActionEncoder = ActionEncoder(self.action_dim,
                                           self.z_action,
                                           self.hidden_dim,
                                           1).to(self.device)

        self.ActionDecoder = ActionDecoder(self.z_action,
                                           self.action_dim,
                                           self.hidden_dim,
                                           1).to(self.device)

        self.prior = Skill_prior(self.state_dim,
                                 self.hrchy[len(self.hrchy) - 1]['z'],
                                 self.hidden_dim).to(self.device)

        self.ActionEncoder = self.ActionEncoder.double()
        self.ActionDecoder = self.ActionDecoder.double()
        self.prior = self.prior.double()
        
        for idx, level in enumerate(self.hrchy):
            if idx == 0:
                old_z = self.z_action
            else:
                old_z = list(self.hrchy.values())[idx-1]['z']
                
            self.encoder = EncoderTransformer(
                old_z,
                self.hrchy[level]['z'],
                self.hidden_dim,
                1).to(self.device)

            self.decoder = DecoderTransformer(
                self.hrchy[level]['z'],
                old_z,
                self.hidden_dim,
                1,
                self.hrchy[level]['length']).to(self.device)

            self.models[f'SeqEncoder{level}'] = self.encoder.double()
            self.models[f'SeqDecoder{level}'] = self.decoder.double()

            self.names[idx] = [f'SeqEncoder{level}', f'SeqDecoder{level}']

    def load_dataset(self, path_to_forward, path_to_backward):
        self.dataset = self.prepare_dataset(path_to_forward, path_to_backward)
        dset_train = Drivedata(self.dataset)

        self.loader = DataLoader(dset_train, shuffle=True, num_workers=8,
                                 batch_size=self.vae_batch_size)

        self.test_loader = DataLoader(dset_train, shuffle=False, num_workers=8,
                                      batch_size=self.dataset['actions'].shape[0])


    def train_level(self, params, lr, beta, level):
        for action, obs in self.loader:
            action, obs = action.to(self.device), obs.to(self.device)
            action, obs = action.view(-1, action.shape[-1]), obs.view(-1, obs.shape[-1])
            recon_loss, kl_loss, kl_prior = self.loss(action, obs, params,
                                                      level)
                        
            loss = recon_loss + beta * kl_loss
            if level != (len(self.hrchy) - 1):
                losses = [loss, recon_loss] 
                params_names = self.names[level]
            elif level == (len(self.hrchy) - 1):
                losses = [loss, recon_loss, kl_prior]
                params_names = self.names[level]
                params_names.append('Prior')
            params = GD_full_update(params, losses, params_names, lr)

        return params
    

    def loss(self, action, obs, params, level):
        z_seq, pdf, _, _ = self.evaluate_encoder_hrchy(action, params, level)
        rec = self.evaluate_decoder_hrchy(z_seq, params, level)

        rec_loss = F.mse_loss(action, rec)
        
        N = Normal(0, 1)
        kl_loss = torch.mean(kl_divergence(pdf, N))
        
        if level == len(self.hrchy) - 1:
            kl_prior = self.train_prior(params, pdf, obs)
        else:
            kl_prior = None

        return rec_loss, kl_loss, kl_prior
            
    def evaluate_encoder_hrchy(self, action, params, level):
        with torch.no_grad():
            z, _, mu, _ = functional_call(self.ActionEncoder,
                                          params['ActionEncoder'],
                                          action)

        for i in range(level+1):
            mu = mu.reshape(-1, self.hrchy[level]['length'], mu.shape[-1])
            z, pdf, mu, std = functional_call(self.models[f'SeqEncoder{i}'],
                                              params[f'SeqEncoder{i}'], mu)

        return z, pdf, mu, std

    def evaluate_decoder_hrchy(self, rec, params, level):
        for i in range(level, -1, -1):
            rec = functional_call(self.models[f'SeqDecoder{i}'],
                                  params[f'SeqDecoder{i}'], rec)
            rec = rec.reshape(-1, rec.shape[-1])
            
        rec = functional_call(self.ActionDecoder,
                              params['ActionDecoder'],
                              rec)        
        return rec

    def train_prior(self, params, pdf, obs):
        index = obs.shape[0] // pdf.loc.shape[0]
        obs = obs[::index, :]
        prior = functional_call(self.prior, params['Prior'], obs)
        return torch.mean(kl_divergence(pdf, prior))

    def train_action_VAE(self, params, lr, beta, epoch):
        for action, obs in self.loader:
            action, obs = action.to(self.device), obs.to(self.device)
            action, obs = action.view(-1, action.shape[-1]), obs.view(-1, obs.shape[-1])
            z_act, pdf, _, _ = functional_call(
                self.ActionEncoder,
                params['ActionEncoder'],
                action)
            z_rec = functional_call(self.ActionDecoder,
                                    params['ActionDecoder'],
                                    z_act)
            N = Normal(0, 1)
            rec_loss = F.mse_loss(z_rec, action)
            kl_loss = torch.mean(kl_divergence(pdf, N))
            losses = [rec_loss + beta * kl_loss, rec_loss]
            names = ['ActionEncoder', 'ActionDecoder']
            params = GD_full_update(params, losses, names, lr)

        var_act = torch.square(action).mean()

        wandb.log({'Square of Actions': var_act.cpu()})
        wandb.log({'loss': rec_loss.cpu()})
        wandb.log({'kl_loss': kl_loss.cpu()})
        wandb.log({'Std for rec': wandb.Histogram(z_rec.std(0).cpu().detach())})
        wandb.log({'Std for action': wandb.Histogram(action.std(0).cpu())})

        return params

    def test_level(self, params, level, epoch):
        for action, obs in self.test_loader:
            action = action.to(self.device)
            action = action.view(-1, action.shape[-1])
            with torch.no_grad():
                z_seq, pdf, mu, std = self.evaluate_encoder_hrchy(action, params, level)
                rec = self.evaluate_decoder_hrchy(z_seq, params, level)

                N = Normal(0, 1)
                rec_loss = F.mse_loss(action, rec, reduction='none')
                kl_loss = torch.mean(kl_divergence(pdf, N))

        if level == (len(self.hrchy) - 1):
            expd_rec_loss = rec_loss.mean(1).view(-1, self.skill_length).mean(1)
            expd_act_std = action.abs().mean(1).view(-1, self.skill_length).mean(1)
            rel_loss = expd_rec_loss / expd_act_std
            
            argmin, argmax = rel_loss.argmin(), rel_loss.argmax()
            
            print(f'Best mse is: {rel_loss.min()} for epoch {epoch}')
            print(f'Worst mse is: {rel_loss.max()} for epoch {epoch}')

            best = {'action': action[argmin*self.skill_length: (argmin+1)*self.skill_length],
                    'recon': action[argmin*self.skill_length: (argmin+1)*self.skill_length],
                    'z': z_seq[argmin, :]}

            worst = {'action': action[argmax*self.skill_length: (argmax+1)*self.skill_length],
                     'recon': action[argmax*self.skill_length: (argmax+1)*self.skill_length],
                     'z': z_seq[argmax, :]}

            path = f'Skills/l{len(self.hrchy)}_len{self.hrchy[0]["length"]}_z{self.hrchy[0]["z"]}_e{epoch}'
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(best, f'{path}/best.pt')
            torch.save(worst, f'{path}/worst.pt')
                
        wandb.log({f'mu {level}': wandb.Histogram(mu.cpu().detach())})
        wandb.log({f'std {level}': wandb.Histogram(std.cpu().detach())})
        wandb.log({'Mean Rec Std': rec.std(0).mean().cpu()})
        wandb.log({'loss': rec_loss.mean().cpu()})
        wandb.log({'kl': kl_loss.cpu()})
        

    def prepare_dataset(self, path_forward, path_backward):
        data_f = torch.load(path_forward)
        data_b = torch.load(path_backward)
        dataset_seqs = {}
        
        for key in data_f:
            # There are 10 runs, from epochs 10 through 100. I'll use runs
            # 0 and 9 because they have the starkest difference. That way the
            # VAE should learn to interpolate
            # I won't use backward data for now.
            # skills = data_f[key][0:999424, :]
            #skills = skills.view(-1, self.skill_length, skills.shape[-1])           
            skills = torch.cat((data_f[key][-1], data_f[key][-2],
                                data_f[key][-3], data_f[key][-4]), dim=0)
            skills = skills[:, :960, :]
            skills = skills.reshape(-1, self.skill_length, skills.shape[-1])           
            dataset_seqs[key] = skills.double()

        return dataset_seqs

    
class Drivedata(Dataset):
    """Dataset loader."""
    def __init__(self, dataset, transform=None):
        self.xs = dataset['actions']
        self.ys = dataset['states']

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return len(self.xs)
