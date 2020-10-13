#basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#importing the environment class of minipong game
from minipong import Minipong



#<-------------------------------------------------------------manual policy----------------------------------------------------------------->
def mystaticpolicy(state):
        # return a action based on the relative position diff
        thresold = 0.1     # it decides after which distance difference paddle has to move
        if state < -thresold:
            return 1
        elif state > thresold:
            return 2
        return 0

#<-----------------------------------------------------------DDQN implementation----------------------------------------------------------> 

#importing libraires
import gym
import tensorflow as tf
from collections import deque

import random
import math

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.models import model_from_json,load_model



class ExperienceReplay:                                                                                                                 #this class is for Replay buffer, to store experience 
    def __init__(self, maxlen, num_states):
        self.buffer = deque(maxlen=maxlen)
        self.num_states = num_states
    
    
    def store(self, state, action, reward, next_state, terminated):                                                 # function to add experience into the buffer
        self.buffer.append((state, action, reward, next_state, terminated))
              
            
    def get_batch(self, batch_size):                                                                                               # function to get batch for the gradient descent 
        if batch_size > len(self.buffer):
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)
        
        
    def get_arrays_from_batch(self, batch):                                                                                   # to get the array of each element from the batch 
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(self.num_states) if x[3] is None else x[3]) 
                                for x in batch])
        
        return states, actions, rewards, next_states
        
    
    def buffer_size(self):
        return len(self.buffer)


class DDQNAgent:
    def __init__(self, num_states, num_actions, replay_buffer_size,  batch_size, max_eps, min_eps, lam, Gamma, Tau): 
        
        """parameters description :-
        
        num_states                              = Total no of states in the Environment used for the input layer of NN
        num_actions                            = Total no of actions can be performed on the Environment and used for the output layer of the NN
        replay_buffer_size                    = it is size of replay buffer
        batch_size                                = it is size of batch which is used for the gradient Descent
        max_eps                                   = it is the value of epsilon in the starting and it will be decreased to its min value iteration by iteration.
        min_eps                                    = it is min. value of the epsilon upto which it will be decreased.
        lam                                           = it is the rate by which epsilon will decrease lamda value in expo
        Gamma                                    =  it is discount rate which is used to account the future rewards
        Tau                                           = it is rate by which target network will be updated 
        neu                                          = for softmax parameter
        """
        
        #intialising all the parameters
        self.num_states = num_states
        self.num_actions = num_actions
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.epsilon = self.max_eps                                                                                           # eps value at current step
        self.lam = lam
        self.Gamma = Gamma
        self.Tau = Tau
        
        
        #building Replaybuffer
        self.experience_replay = ExperienceReplay(self.replay_buffer_size, self.num_states)
        
        #building_Neural_network
        
        self.primary_network = self.build_network()                                                                                                                            #main network
        self.primary_network.compile(optimizer='adam', loss = 'mse')
        
        self.target_network = self.build_network()                                                                                                                              #target network
        
        return
        
              
    def build_network(self):
        # network = Sequential()
        # network.add(Dense(30, activation='relu', kernel_initializer=he_normal()))
        # network.add(Dense(30, activation='relu', kernel_initializer=he_normal()))
        # network.add(Dense(self.num_actions))
        # network = Sequential()
        # network.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = self.num_states))   
        # network.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))  
        # network.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu')) 
        # network.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu')) 
        # network.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu')) 
        # network.add(Dense(units = self.num_actions, kernel_initializer = 'uniform', activation = 'linear'))  
        network = Sequential()
        network.add(Dense(30, activation='relu', kernel_initializer=he_normal(),  input_dim = self.num_states))
        network.add(Dense(30, activation='relu', kernel_initializer=he_normal()))
        network.add(Dense(self.num_actions))
        return network
        
    
    def align_epsilon(self, step):
        self.epsilon = self.min_eps + (self.max_eps - self.min_eps) * math.exp(-self.lam * step)
    
    
    def align_target_network(self):
        for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables): 
            t.assign(t * (1 - self.Tau) + e * self.Tau)
    
    
    def choose_action(self, state):
#         print(self.epsilon)\
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions - 1)
        else:
            q_values = self.primary_network.predict(state.reshape(1, -1))
            return np.argmax(q_values)
    
    
    def choose_action_for_testing(self,state):
        q_values = self.primary_network.predict(state.reshape(1, -1))
        return np.argmax(q_values)
        
    
    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.store(state, action, reward, next_state, terminated)
    
    
    def train(self):
        if self.experience_replay.buffer_size() < self.batch_size * 3:
            return 0
        
        batch = self.experience_replay.get_batch(self.batch_size)
        states, actions, rewards, next_states = self.experience_replay.get_arrays_from_batch(batch)
        
        # Predict Q(s,a) and Q(s',a') given the batch of states
        q_values_state = self.primary_network.predict(states)
        q_values_next_state = self.primary_network.predict(next_states)
        
        # Copy the q_values_state into the target
        target = q_values_state
        updates = np.zeros(rewards.shape)
                
        valid_indexes = np.array(next_states).sum(axis=1) != 0
        batch_indexes = np.arange(self.batch_size)

        action = np.argmax(q_values_next_state, axis=1)
        q_next_state_target = self.target_network.predict(next_states)
        updates[valid_indexes] = rewards[valid_indexes] + self.Gamma * q_next_state_target[batch_indexes[valid_indexes], action[valid_indexes]]
        
        target[batch_indexes, actions] = updates
        loss = self.primary_network.train_on_batch(states, target)

        # update target network parameters slowly from primary network
        self.align_target_network()
        
        return loss
    
    def save_model(self):
        self.primary_network.save('model/primary_model.h5')
        self.target_network.save('model/target_model.h5')
        print("Saved model to disk")
        
    
    def load_model(self, primary_network_name = 'primary_model', target_network_name = 'target_model'):
           
        self.primary_network = load_model('model/'+ primary_network_name +'.h5')
        self.target_network = load_model('model/'+ target_network_name +'.h5')
        print("Loaded model from disk")



#<--------------------------------------------------------------training--------------------------------------------------------------->

agent = DDQNAgent(num_states = 1,num_actions = 3,replay_buffer_size= 50000,batch_size= 128,max_eps= 1,min_eps= 0.1,lam= 0.001,Gamma= 0.99,Tau= 0.01)
env = Minipong(level = 1, size = 5)


episode_reward = []
running_reward = 0
render = True
log_interval = 10
render_interval = 10
cummulative_reward = []  #for each episode i will log the cummulative reward

starttime = time.time()

for i_episode in range(160):
        state, ep_reward, done = env.reset(), 0, False
        rendernow = i_episode % render_interval == 0
        average_loss = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            
            # select action by RL agent
            action = agent.choose_action(state)

            # take the action
            next_state, reward, done = env.step(action)
            reward = float(reward)     # strange things happen if reward is an int
            agent.store(state, action, reward, next_state, done)

            loss = agent.train()
            average_loss += loss
            
            state = next_state
            agent.align_epsilon(t)

            # if render and rendernow:
            #     env.render(reward = ep_reward)

            ep_reward += reward
            if done:
                break
        
        
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        episode_reward.append(ep_reward)

        cummulative_reward.append(running_reward)
        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

plt.title('Training Reward Plot')
plt.xlabel('Number of Episodes')
plt.ylabel('Cummulative reward')
plt.plot(cummulative_reward)
plt.show()



# <-----------------------------------------------------------testing---------------------------------------------------->

episode_reward = []
running_reward = 0
render = True
log_interval = 10
render_interval = 10
cummulative_reward = []  #for each episode i will log the cummulative reward

starttime = time.time()

for i_episode in range(50):
        state, ep_reward, done = env.reset(), 0, False
        # rendernow = i_episode % render_interval == 0
        # average_loss = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            
            # select action by RL agent
            action = agent.choose_action_for_testing(state)

            # take the action
            next_state, reward, done = env.step(action)
            reward = float(reward)     # strange things happen if reward is an int
            agent.store(state, action, reward, next_state, done)

            # loss = agent.train()
            # average_loss += loss
            
            state = next_state
            # agent.align_epsilon(t)

            # if render and rendernow:
            #     env.render(reward = ep_reward)

            ep_reward += reward
            if done:
                break
        
        
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        episode_reward.append(ep_reward)

        cummulative_reward.append(running_reward)
        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

print('average reward in testing = ',np.array(episode_reward).mean())
print('standard deviation = ', np.array(episode_reward).std())
