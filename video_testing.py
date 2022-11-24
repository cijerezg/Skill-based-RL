import torch
from torch.nn.utils.stateless import functional_call
import numpy as np
import gym
import pickle
from hierarchical_vae import VAE
from models import Policy
import wandb
import pdb


wandb.init(project='Video_Test_VAE')


class video_evaluation:
    def __init__(self, env_id):
        self.env = gym.make(env_id, render_mode="rgb_array")

    def record_premade_action(self, filename, path, name):
        best = torch.load(f'Skills/{filename}/best.pt')
        worst = torch.load(f'Skills/{filename}/worst.pt')
        print(f'Z for best is {best["z"]}')
        print(f'Z for worst is {worst["z"]}')
        names = ['action', 'recon']
        for name in names:
            self.record_video(path, f'best_{name}',
                              best[name].cpu().detach().numpy())
            self.record_video(path, f'worst_{name}',
                              worst[name].cpu().detach().numpy())

    def record_vae_model(self, z1, z2, vae_model, path, name):
        z1 = torch.load(f'Skills/{z1}/best.pt')
        z2 = torch.load(f'Skills/{z2}/worst.pt')
        z1, z2 = z1['z'], z2['z']
        self.init_models(vae_model, policy=None)
        for i in range(5):
            z = i/4 * z1 + (1-i/4) * z2
            z = z.reshape(1, -1)
            actions = self.vae.evaluate_decoder_hrchy(z, self.params,
                                                      len(self.vae.hrchy) -1)
            # Check the shape of actions after it comes out
            actions = actions.reshape(self.vae.skill_length, actions.shape[-1])
            actions = actions.cpu().detach().numpy()
            self.record_video(f'{path}_VAE', f'{name}_{i}', actions)

    def record_policy(self, policy, vae_model, path, name):
        self.init_models(vae_model, policy)
        self.record_video(path, name, policy=True)
        
    def eval_policy(self, obs, i):
        if self.counter == 0:
            obs = torch.from_numpy(obs).to(self.vae.device).view(1, -1)
            with torch.no_grad():
                z, _, _, mu = functional_call(self.policy,
                                              self.params['Policy'],
                                              obs)

                actions = self.vae.evaluate_decoder_hrchy(z, self.params,
                                                          len(self.vae.hrchy)-1)
                actions = actions.reshape(self.vae.skill_length, actions.shape[-1])
            actions = actions.cpu().detach().numpy()
            self.policy_actions = actions        
            self.counter += actions.shape[0]
            self.counter -= 1
            return self.policy_actions[i % self.vae.skill_length, :]
            
        else:
            self.counter -= 1
            return self.policy_actions[i % self.vae.skill_length, :]
        
    def record_video(self, path, name, actions=None, policy=None):
        #steps = actions.shape[0] if policy is None else 256
        steps = 80
        env = self.env
        env._max_episode_steps = steps
        env = gym.wrappers.RecordVideo(env, f'Videos/{path}/{name}')
        obs = env.reset()
        vel = []
        self.counter = 0
        for i in range(steps):
            k = i % 16
            action = actions[k, :] if policy is None else self.eval_policy(obs, i)
            obs, reward, done, _, info = env.step(action)
            vel.append(info['reward_run'])

        print(f'Mean speed  is {np.array(vel).mean()}')
        env.close()

    def init_models(self, vae_model, policy=None):
        parent = 'VAE_models'
        parent_p = 'Policy_models'
        with open(f'{parent}/{vae_model}/class', 'rb') as file:
            self.vae = pickle.load(file)
        self.params = torch.load(f'{parent}/{vae_model}/params.pt')

        params_act_decoder = torch.load('trained_models/ActionDecoder_lr_0.1_z_2.pt')
        self.params['ActionDecoder'] = params_act_decoder
        if policy is not None:
            with open(f'{parent_p}/{policy}/class', 'rb') as file:
                self.policy = pickle.load(file)
            mod = torch.load(f'{parent_p}/{policy}/params.pt')
            self.params['Policy'] = mod['Policy']
    
        
video = video_evaluation('HalfCheetah-v4')

# models = ['l1_len16_z6_e200', 'l2_len4_z6_e200', 'l4_len2_z6_e200']

# for model in models:
#     video.record_premade_action(model, model, 'skills')


# z1 = 'l3_len4_z12_e400'
# z2 = 'l3_len4_z12_e400'
# vae_model = 'l3_len4_z12'

# video.record_vae_model(z1, z2, vae_model, vae_model, 'VAE')

policies = ['l4_len2_z6/1200', 'l2_len4_z6/1200', 'l1_len16_z6/1200']

vae_models = ['l4_len2_z6', 'l2_len4_z6', 'l1_len16_z6']

for policy, vae_model in zip(policies, vae_models):
    video.record_policy(policy, vae_model, policy, 'Policy')
