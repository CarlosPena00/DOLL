import warnings
warnings.filterwarnings("ignore")

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

import cv2
import torch

import random
from . import VOC
from . import History, Stats
from matplotlib import pyplot as plt 

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'

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

        #Render
        plt.figure(0)
        self.ax1 = plt.subplot(1,2,1)
        self.ax2 = plt.subplot(1,2,2)

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

    def compute_rewards(self, done=False, use_steps_reward=False):
        #Note: The original reward is with use_steps_reward=False
        
        diff_iou   = self.history.hist_iou[-1] - self.history.hist_iou[-2]
        self.r_iou = 1 if diff_iou > 0 else -1

        self.r_steps = self.history.num_insertions * (0.0001)  \
                            if use_steps_reward else 0.0

        if done:
            big_iou = 3 if (self.history.hist_iou[-1] > 0.5) else -3
            return big_iou - self.r_steps
        
        return self.r_iou - self.r_steps

    def step(self, action):
        # right, left, up, down, bigger, smalller, fatter, taller , trigger
        
        
        self.done = (action == 8)
        if self.done and self.do_render:
            self.render()
        
        state  = self._receive_state(action)
        reward = self.compute_rewards(self.done)
        
        return state, reward, self.done, self.history

    def render(self):

        self.draw = VOC.inv_normalize(self.input[0]).cpu().numpy().transpose(1,2,0).clip(0, 255).astype(np.uint8)
        self.dtarget = ((self.target[0] >= 1) * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

        bbox, obbox = self.history.bbox, self.history.hist_bbox[-2]
        
        rect = cv2.rectangle(self.draw.copy(), (obbox[2], obbox[0]), (obbox[3], obbox[1]), (127,0,255), 3)
        rect = cv2.rectangle(rect, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255,0,0), 3)

        self.ax1.imshow(rect )
        self.ax1.title.set_text(self.history.hist_iou[-1])
        rect = cv2.rectangle(self.dtarget.copy(), (bbox[2], bbox[0]), (bbox[3], bbox[1]), (180), 2)
        self.ax2.imshow( rect, vmax=255, vmin=0)

        plt.draw()
        plt.pause(0.001)

    def close(self):
        self.stop()


