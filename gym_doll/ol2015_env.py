import os
import socket
import struct
import subprocess
import time
import math
from datetime import datetime
import gym
import numpy as np
from gym.spaces import Box, Discrete

import torch

import random
from . import VOC
from . import History

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'


w_grad_ball_potential = (0.08, 1)
# Active Object Localization with Deep Reinforcement Learning
class Ol2015_Env(gym.Env):

    def __init__(self, history_size=5,
                 update_interval=1, render=False, 
                 is_discrete=True, logger_path='log.txt'):

        super(Ol2015_Env, self).__init__()
        np.random.seed(2020)

        self.do_render    = render
        self.history      = History(history_size)
        self.history_size = history_size
        self.time_limit   = (5 * 60 * 1000) #Change
        self.is_discrete  = is_discrete
        
        self.done   = False
        self.broken = False
        
        self.logger_path     = logger_path
        self.update_interval = update_interval
        self.window_size     = (history_size//update_interval)

        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=(self.window_size, self.history.state_shape),
                                     ) # Change it
        if self.is_discrete:
            self.action_space = Discrete(9)
        else:
            self.action_space = Box(low=-1.0, high=1.0,
                                    shape=(3,),
                                    ) # Change it
            
        transforms_set = VOC.Compose([
                        VOC.ToSegmentation(),
                        #ConvertLabel(),
                        VOC.ToTensor()]
                    )

        dataset          = VOC.VOCDetection(r'Datasets/', transforms=transforms_set)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                    shuffle=True, num_workers=0)

    def start_agents(self):
        pass

    def check_agents(self):
        return True

    def stop_agents(self):
        pass

    def start(self):
        self.r_iou ,self.r_steps = 0, 0
        self.input, self.target = iter(self.data_loader).next()
        
        self.history.start(self.input, self.target)

        
    def stop(self):
        pass
    

    def _receive_state(self, action):
        
        self.history.update(action)
        state = self.history.cont_states
        state = np.array(state)
        return state

    def write_log(self):
        with open(self.logger_path, 'a') as log:
            now = datetime.now()
            steps = self.history.num_insertions
            iou = self.history.cont_states[-1][1] if steps else 0
            log.write(f"{iou},{steps},{now}\n")
            
    def reset(self):
        self.broken = False
        self.write_log()

        self.start()
        state = self._receive_state(4)
        return np.array(state)

    def bound(self, value, floor, ceil):
        if value < floor:
            return floor
        elif value > ceil:
            return ceil
        else:
            return value

    def compute_rewards(self):
        
        self.r_iou   = self.history.hist_iou[-1] - self.history.hist_iou[-2]
        self.r_steps = self.history.num_insertions * (0.001)
        return self.r_iou - self.r_steps

    def step(self, action):
        # right, left, up, down, bigger, smalller, fatter, taller , trigger

        self.done = (action == 8)
        
        state  = self._receive_state(action)
        reward = self.compute_rewards()
        
        return state, reward, self.done, self.history

    def render(self, mode='human'):
        pass

    def close(self):
        self.stop()


