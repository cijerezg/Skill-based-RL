import gym
import d4rl
import torch

env = gym.make('halfcheetah-expert-v2')
dataset = env.get_dataset()

vals = ['actions', 'observations']

vae_dataset = {val: torch.tensor(dataset[val]) for val in vals}

torch.save(vae_dataset, 'Dataset/d4rl_halfcheetah_expert.pt')



