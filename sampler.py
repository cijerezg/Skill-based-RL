"""Run tasks."""

import envpool
import numpy as np
from torch.nn.utils.stateless import functional_call
import torch
import scipy.signal
from utils import hyper_params
import pdb


MAXIMUM_EPISODE_STEPS = 256

class Sampler(hyper_params):
    def __init__(self, policy, decoder, actiondecoder, args):
        super().__init__(args)
        self.policy = policy
        self.decoder = decoder
        self.ActionDecoder = actiondecoder

        self.num_envs = self.meta_batch_size * self.batch_size
        
        self.env = envpool.make(self.env_id, env_type='gym',
                                num_envs=self.num_envs,
                                max_episode_steps=MAXIMUM_EPISODE_STEPS)
        self.variables = ['state', 'next_state', 'z', 'next_z', 'actions',
                          'density', 'next_density', 'reward_run',
                          'reward_ctrl', 'states', 'reward', 'reward_test']

        self.test_tasks = np.sort(np.random.uniform(
            1.99, 2.0, (self.meta_batch_size,)))
        
    def full_episode(self, params):        
        data = {var: [] for var in self.variables}

        state = self.env.reset()[0]
        done = np.array([False, False])
        while ~done.all():
            state, data,  done = self.skill_step(params, state,
                                                 data, self.test_tasks)
    
        return data

    def skill_length_step(self, params, data, tasks, done=False):
        if len(data['state']) == 0 or done.all():
            state = self.env.reset()[0]
        else:
            state = data['state'][-1]
        state, data, done = self.skill_step(params, state, data, tasks)
        return data, done
    
    def skill_execution(self, actions, env_ids):
        states = []
        reward_run = []
        reward_ctrl = []
        dones = []
        for i in range(actions.shape[1]):
            state, rew, _, done, info = self.env.step(actions[:, i, :],
                                                      env_id=env_ids)
            states.append(state)
            reward_run.append(info['reward_run'])
            reward_ctrl.append(info['reward_ctrl'])
            dones.append(done)
        if np.array(dones).all() is True:
            done = True
        return state, states, reward_run, reward_ctrl, done

    def skill_step(self, params, state, data, tasks):
        state_tensor = torch.from_numpy(state).to(self.device)
        z, _, density, _ = functional_call(self.policy,
                                           params['Policy'],
                                           state_tensor)

        with torch.no_grad():
            actions = self.decoder(z, params, len(self.hrchy) - 1)
            actions = actions.reshape(-1, self.skill_length, actions.shape[-1])
        actions = actions.cpu().detach().numpy()
        clipped_actions = np.clip(actions, -1, 1)

        # z is shape (N, latent_var_dim), where N is number of samples.
        # actions is shape (N, seq, act_dim).
            
        next_state, states, rew_run, rew_ctrl, done = self.skill_execution(
            clipped_actions, np.arange(self.num_envs))

        reward, reward_test = self.reward_calculation(rew_run, rew_ctrl, tasks)
        next_state_tensor = torch.from_numpy(next_state).to(self.device)

        with torch.no_grad():
            next_z, _, next_density, _ = functional_call(self.policy,
                                                         params['Policy'],
                                                         next_state_tensor)

        data['state'].append(state)
        data['next_state'].append(next_state)
        data['z'].append(z)
        data['next_z'].append(next_z)
        data['actions'].append(actions)
        data['density'].append(density)
        data['next_density'].append(next_density)
        data['reward_run'].append(rew_run)
        data['reward_ctrl'].append(rew_ctrl)
        data['states'].append(states)
        data['reward'].append(reward)
        data['reward_test'].append(reward_test)

        return next_state, data, done

    def reward_calculation(self, rew_run, rew_ctrl, tasks):
        rew_run_np, rew_ctrl_np = np.array(rew_run), np.array(rew_ctrl)
        batched_tasks = np.repeat(tasks[:, np.newaxis], self.batch_size,
                                  axis=1)
        final_tasks = np.repeat(batched_tasks.flatten()[np.newaxis],
                                self.skill_length, axis=0)

        reward = -np.abs(rew_run_np - final_tasks) + 0.5 * rew_ctrl_np
        reward_train = np.sum(reward, axis=0)
        reward_test = np.mean(reward, axis=1)
        
        return reward_train, reward_test
