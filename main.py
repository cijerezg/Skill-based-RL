"""Run everything."""

from models import Policy, Critic
from sampler import Sampler
from algos import MSAC
import torch
import pickle
import numpy as np
import os
from hierarchical_vae import VAE
import itertools
from utils import params_extraction, load_pretrained_models, vae_config
import pdb
import wandb

# Little test
os.environ['WANDB_SILENT'] = "true"

wandb.login()

sweep_config = {'method': 'grid'}

metric = {'name': 'loss',
          'goal': 'minimize'}

sweep_config['metric'] = metric


parameters_dict = {
    'meta_batch_size': {
        'value': 20},
    'batch_size': {
        'value': 20},
    'vae_batch_size': {
        'value': 1024},
    'case': {
        'values': [0, 1, 2]},
    'rl_lr': {
        'value': .001},
    'actions_lr': {
        'value': 0.1},
    'discount': {
        'value':  0.99},
    'alpha': {
        'value': 0.1},
    'env_id': {
        'value': 'HalfCheetah-v4'},
    'device': {
        'value': 'cuda:0'},
    'hidden_dim': {
        'value': 128},
    'epochs': {
        'value': 600},
    'z_action': {
        'value': 2},
    'beta': {
        'value': 0.1},
    'use_pretrained_action_VAE': {
        'value': True},
    'use_pretrained_VAE': {
        'value': True},
    'train_policy': {
        'value': True}}


sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='Hierarchical-Offline-RL')

# path_to_forward = 'Dataset/d4rl_halfcheetah_expert.pt'

path_to_forward = 'Dataset/data_forward_vel.pt'
path_to_backward = 'Dataset/data_backward_vel.pt'

folder = 'results/Experiment'

def main(config=None):
    with wandb.init(config=config):

        config = wandb.config
        if not os.path.exists(folder):
            os.makedirs(folder)

        config = vae_config(config)
        
        vae = VAE(config)
        vae.load_dataset(path_to_forward, path_to_backward)
        
        z_dim = vae.hrchy[len(vae.hrchy) - 1]['z']
        policy = Policy(vae.state_dim, z_dim,
                        vae.hidden_dim).to(vae.device)
        critic = Critic(vae.state_dim, z_dim,
                        vae.hidden_dim).to(vae.device)
        policy = policy.double()
        critic = critic.double()

        sampler = Sampler(policy, vae.evaluate_decoder_hrchy,
                          vae.ActionDecoder, config)

        msac = MSAC(sampler, vae, policy, critic, config)

        vae_models = list(vae.models.values())

        models = [vae.ActionEncoder, vae.ActionDecoder, *vae_models,
                  vae.prior, msac.policy, msac.critic, msac.critic]

        vae_names = list(itertools.chain(*list(vae.names.values())))
        names = ['ActionEncoder', 'ActionDecoder', *vae_names,
                 'Prior', 'Policy', 'Critic', 'Target_critic']

        pretrained_params = load_pretrained_models(config)
        pretrained_params.extend([None] * (len(names) - len(pretrained_params)))

        params = params_extraction(models, names, pretrained_params)

        if not config.use_pretrained_action_VAE:
            for vae_epoch in range(400):
                params = vae.train_action_VAE(params, config.actions_lr,
                                              config.beta, vae_epoch)
            torch.save(params['ActionDecoder'],
                       f'trained_models/ActionDecoder_MyData_z{config.z_action}.pt')
            torch.save(params['ActionEncoder'],
                       f'trained_models/ActionEncoder_MyData_z{config.z_action}.pt')
                

        # VAE training warm up
        level_epoch = 360 // len(vae.hrchy)
        if not config.use_pretrained_VAE:
            path = f'VAE_models/l{config.levels}_len{config.level_length}_z{config.z_vae}'
            if not os.path.exists(path):
                os.makedirs(path)
            for j in range(len(vae.hrchy)):
                print(f'Training level {j}')
                for vae_epoch in range(level_epoch):
                    params = vae.train_level(params, config.vae_lr,
                                             config.beta, j)
                    if vae_epoch % 20 == 0:
                        vae.test_level(params, j, vae_epoch)
            vae_mdls = {key: params[key] for key in params if key in vae_names}
            torch.save(vae_mdls, f'{path}/params.pt')
            with open(f'{path}/class', 'wb') as file:
                pickle.dump(vae, file)
                        
        # Main training loop
        print('==============================================================')
        print(f'New Run with case {config.case}')
        print('==============================================================')
        if config.train_policy:
            for i in range(config.epochs+1):
                critic_warmup = True if i < 30 else False
                tasks = np.random.uniform(1.99, 2.0, (sampler.meta_batch_size,))
                data, params = msac.train_episode(params, tasks, config.rl_lr,
                                                  critic_warmup=critic_warmup,
                                                  testing=False)
                if i % 1 == 0 and i > -5:
                    mean_total_reward = []
                    mean_total_reward_run = []
                    for idx, test_task in enumerate(sampler.test_tasks):
                        if idx > 2:
                            continue
                        test_task = np.repeat(test_task, sampler.meta_batch_size)
                        # d_test, test_params = msac.train_episode(params, test_task,
                        #                                          config.rl_lr,
                        #                                          critic_warmup,
                        #                                          testing=True)
                        # test_data = msac.test_episode(test_params)
                        test_data = msac.test_episode(params)
                        mean_reward = np.sum(np.stack(test_data['reward_test']))
                        mean_reward_run = np.mean(np.stack(test_data['reward_run']))
                        mean_total_reward.append(mean_reward)
                        mean_total_reward_run.append(mean_reward_run)
                    mean_total_reward = np.mean(np.stack(mean_total_reward))
                    mean_total_reward_run = np.mean(np.stack(mean_total_reward_run))
                    wandb.log({'reward epoch': mean_total_reward})
                    wandb.log({'reward run epoch': mean_total_reward_run})
                    wandb.log({'Rewards Hist':
                               wandb.Histogram(np.stack(test_data['reward']))})
                    wandb.log({'Reward Run Hist':
                               wandb.Histogram(np.stack(test_data['reward_run']))})
                    wandb.log({'Policy Stds':
                               wandb.Histogram(params['Policy']['log_std'].detach().cpu().numpy())})
                    wandb.log({'Epoch': i})
                    if i % 20 == 0:
                        path = f'Policy_models/l{config.levels}_len{config.level_length}_z{config.z_vae}/{i}'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        torch.save({'Policy': params['Policy']}, f'{path}/params_{i}.pt')
                        with open(f'{path}/class', 'wb') as file:
                            pickle.dump(msac.policy, file)
                    if i == config.epochs:
                        wandb.log({'loss': -mean_total_reward})
                        

wandb.agent(sweep_id, main)
