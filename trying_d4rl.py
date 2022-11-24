import gym
import d4rl

env = gym.make('kitchen-complete-v0')
env = gym.wrappers.RecordVideo(env, 'Videos/cool1')

dataset = env.get_dataset()
#env1 = gym.make('HalfCheetah-v3')

env.reset()

for i in range(100):
    #env1.render()
    state, rew, info, done = env.step(dataset['actions'][i, :])
    

#dataset = env.get_dataset()

env.close()
#[print(name, va.shape) for name, va in dataset.items()]

