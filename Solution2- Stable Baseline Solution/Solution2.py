#basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#importing the environment class of minipong game
from minipong import Minipong


import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


#<-------------------------------------------------------training----------------------------------------------------------->
pong = Minipong(level = 3, size = 5)
env = DummyVecEnv([lambda: pong])

model = DQN(MlpPolicy, env, verbose=1, gamma=0.95, tensorboard_log="./MinipongLog/")
model.learn(total_timesteps = 80000)
#saving the model for future usability
model.save('model/DQN_model_minipong')

#you can check the tensor board log page easily by using this command in bash 
#tensorboard --logdir ./MinipongLog/ --host --localhost 

#displaying the training reward plot
cummulative_reward_per_episode = env.get_attr('running_reward_list_per_episode')[0]
plt.title('Training reward per episode')
plt.xlabel('Number of episodes')
plt.ylabel('Cummulative reward sum')
plt.plot(cummulative_reward_per_episode)
plt.show()


#<---------------------------------------------------------testing---------------------------------------------------------->

pong = Minipong(level=3, size = 5)
testing_env = DummyVecEnv([lambda: pong])
model = DQN.load('model/DQN_model_minipong')
number_of_episodes = 100
for _ in range(number_of_episodes):
    done = False
    state = testing_env.reset()
    while not done:
        action, _ = model.predict(state)
        state, reward, done, info = testing_env.step(action)
        
        

#displaying the testing reward plot
Episode_wise_reward_sum = testing_env.get_attr('episode_wise_reward_sum')[0]

print('Test-Average = ', np.array(Episode_wise_reward_sum).mean())
print('Test-StandardDeviation = ', np.array(Episode_wise_reward_sum).std())

