#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 08:51:15 2020

@author: oliver
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time
from itertools import count
from collections import namedtuple


class Minipong:

    def __init__(self, level = 5, size = 5, normalise = True):
        # level is the dimensionality of state vector (1-5, 0: return pixels)
        self.level = min(5, max(0, level))
        # size is the number of paddle positions (every 3 pixels)
        self.zmax = max(3, size) - 1
        self.size = 3 * max(3, size)
        self.xymax = self.size - 2
        # whether to normalise the state representation
        self.scale = self.xymax if normalise else 1.0
        
        # the internal state is 
        # x, y, z, dx, dy
        x = random.randint(1, self.xymax)
        z = random.randint(0, self.zmax)
        self.s0 = (x,2,z,0,0)
        self.s1 = (x,2,z,1,1)

        self.p0 = np.zeros((self.size, self.size), dtype=int)
        self.p1 = np.zeros((self.size, self.size), dtype=int)

        self.list_of_bins_ranges = self.state_to_descrete_state()
        
        self.previous_state = None
        self.transition()

        return

    def observationspace(self):
        if self.level == 0:
            return self.size * self.size
        return self.level

    def state_to_descrete_state(self, min_value = -1, max_value = 1):
        bins = 20
        width = (max_value-min_value)/bins
        li = []
        for i in range(bins):
            li.append(min_value + width)
            min_value += width 
        
        return li
        
    def convert_value_to_discrete_value(self, value):
        # print(self.list_of_bins_ranges)
        # print(value)
        for i in self.list_of_bins_ranges:
            if value <= i:
                return i
        
        return -1
        

    def state(self):
        """
        Returns the (observed) state of the system.
        
        Depending on level, the observed state is an 
        array of 1 to 5 values, or a pixel representation (level 0).

        Returns
        -------
        np.array
            level 1: [dz] 
                dz: relative distance in x-coordinate between paddle and ball.
                    normalised to values between -1 and 1 
            level 2: [y, dz]
                y : ball y-coordinate, normalised to values between 0 and 1
            level 3: [y, dx, dz]
                dx: change in ball x-coordinate from last frame to now
                    -1 or 1 in most cases (can be larger after hitting paddle)
            level 4: [x, y, dx, dy]
                x: ball x-coordinate, normalised to values between 0 and 1.
                dy: change in ball y-coordinate from last frame to now (-1 or 1)
            level 5: [x, y, dx, dy, dz]
            
            level 0: a square bitmap (1 x size x size)
                the difference bitmap between this frame and last frame,
                calculated as "this frame" - 0.5 "last frame".
                "this frame" and "last frame" are binary (0/1) bitmaps.

                This difference makes it possible to detect moving and 
                static objects.

        """
        x1,y1,z1,_,_ = self.s1
        zx = z1 * 3 + 1 # paddle center location (1..xymax)
        dz = x1 - zx    # x1 - zx (1-xymax..xymax-1)
        x0,y0,_,_,_ = self.s0
        dx = x1 - x0    # 
        dy = y1 - y0    # should be between -1 and 1
        
        if self.level == 0:
            self.p1 = self.to_pix(self.s1, binary = True)
            return (self.p1 - 0.5 * self.p0).reshape((-1, self.size, self.size))
        elif self.level == 1:
            return np.array([self.convert_value_to_discrete_value(dz/self.scale)])
        elif self.level == 2:
            return np.array([self.convert_value_to_discrete_value(y1/self.scale), self.convert_value_to_discrete_value(dz/self.scale)])
        elif self.level == 3:
            return np.array([self.convert_value_to_discrete_value(y1/self.scale), self.convert_value_to_discrete_value(dx), self.convert_value_to_discrete_value(dz/self.scale)])
        elif self.level == 4:
            return np.array([self.convert_value_to_discrete_value(x1/self.scale), self.convert_value_to_discrete_value(y1/self.xymax), self.convert_value_to_discrete_value(dx), self.convert_value_to_discrete_value(dy)])
        return np.array([self.convert_value_to_discrete_value(x1/self.xymax), self.convert_value_to_discrete_value(y1/self.xymax), self.convert_value_to_discrete_value(dx), self.convert_value_to_discrete_value(dy), self.convert_value_to_discrete_value(dz/self.xymax)])

    def transition(self, action = 0):
        """
        Apply an action and update the environment.
        0: do nothing
        1: move left
        2: move right

        Parameters
        ----------
        action : int, optional
            The action applied before the update. 
            The default is 0 (representing no action).

        Returns
        -------
        np.array
            The new observed state of the environment.

        """
        self.s0 = self.s1
        self.p0 = self.p1

        if self.terminal():
            return self.state()

        x0, y0, z0, dx0, dy0 = self.s0
        x1, y1, z1, dx1, dy1 = self.s0  # s0 == s1 at this moment

        if action == 1:
            z1 = max(0, z0 - 1)
        elif action == 2:
            z1 = min(self.zmax, z0 + 1)

        if x0 == 1 and dx0 == -1:
            dx1 = 1
        if x0 == self.xymax and dx0 == 1:
            dx1 = -1
        if y0 == self.xymax and dy0 == 1:
            dy1 = -1
            
        if y0 == 2 and dy1 == -1:
            z = z1 * 3
            if x0 >= z and x0 < z + 3:
                dy1 = 1
                x1 = min(self.xymax, max(1, x0 + random.randint(-2, 2))) - dx1
                
        x1 += dx1
        y1 += dy1

        self.s1 = (x1, y1, z1, dx1, dy1)
        
        return self.state()


    def terminal(self):
        """
        Check if episode is finished.

        Returns
        -------
        bool
            True if episode is finished.

        """
        x, y, z, dx, dy = self.s1

        return y == 1

    
    def reward(self, n, p):    
        """
        Calculate immediate reward.
        Larger positive reward for hitting the paddle.

        Parameters
        ----------
        action : int
            0-3, for the 3 possible actions.

        Returns
        -------
        r : float
            immediate reward.

        """

        r = 0.0
        
        if n <= p:
            r +=8
       
        r +=2                 # for every step i am giving constant reward so that agent try to understand he has to play as long as it can


        x, y, z, dx, dy = self.s1
        if y == 2 and dy == -1:
            if x >= z*3 and x < (z+1)*3:
                r += self.xymax
            
        return r
        

    def step(self, action):
        # return tuple (state, reward, done)
        previous_state = self.transition()
        next_state = self.transition(action)
        r = self.reward(next_state, previous_state)
        done = self.terminal()
        return (next_state, r, done)


    def reset(self):
        x = random.randint(1, self.xymax)
        z = random.randint(0, self.zmax)
        self.s0 = (x,2,z,0,0)
        self.s1 = (x,2,z,1,1)

        self.p0 = np.zeros((self.size, self.size), dtype=int)
        self.p1 = np.zeros((self.size, self.size), dtype=int)

        return self.transition()
        

    def render(self, text = True, reward = None):
        if text:
            t = np.array(list(' '*(self.size+2)*(self.size+1))).reshape(self.size+2, -1)
            t[:, self.size] = '\n'
            t[0, 0:self.size] = '_'
            t[self.size+1, 0:self.size] = '='
            
            x, y, z, _, _ = self.s1
            y = self.size - y - 1

            for i in range(3):
                t[self.size, z*3 + i] = '-'
                t[y + 1, x + i - 1] = '+'
            t[y, x] = '+'
            t[y + 2, x] = '+' 
            if reward is not None:
                print('{:.3f}'.format(reward))
            print(''.join(t.ravel()))                       
        else:
            pix = self.to_pix(self.s1)
            fig, ax = plt.subplots()
            if reward is not None:
                plt.title(f'Reward: {reward}', loc='right')
            ax.axis("off")
            plt.imshow(pix, origin='lower' , cmap='red')
            plt.show()


    def to_pix(self, state, binary = False):
        """
        Generate a picture from an internal state representation 

        Parameters
        ----------
        state : np.array
            Internal state, first three values are used (x,y,z,_,_)
        binary : bool, optional
            If true pixel values will be binary 0/1. The default is False,
            in which case the paddle is represented with -1 values.

        Raises
        ------
        ValueError
            If x,y,z are outside their range.

        Returns
        -------
        image : np.array
            a square image with pixel values -1,0, and 1.

        """
        x,y,z,_,_ = state
        if x < 1 or x > self.xymax:
            raise ValueError('x-coordinate value error')
        if y < 1 or y > self.xymax:
            raise ValueError('y-coordinate value error')
        if z < 0 or z > self.zmax:
            raise ValueError('z-coordinate value error')
        paddle = 1 if binary else -1
    
        image = np.zeros((self.size, self.size), dtype = int)
        image[self.size-1, 0] = paddle
        image[self.size-1, self.size-1] = 1

        for i in range(3):
            image[0, z*3 + i] = paddle
            image[y, x + i - 1] = 1
        image[y - 1, x] = 1
        image[y + 1, x] = 1
        
        return image

