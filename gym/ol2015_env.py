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

#from Game import History, Stats
import random
import VOC

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'


w_grad_ball_potential = (0.08, 1)
# Active Object Localization with Deep Reinforcement Learning
class Ol2015_Env(gym.Env):

    def __init__(self, qtde_steps=60, history_size=60,
                 update_interval=15, render=False, 
                 is_discrete=True, logger_path='log.txt'):

        super(CoachEnv, self).__init__()
        np.random.seed(2020)

        self.do_render = render
        self.history = History(history_size)
        self.history_size = history_size
        self.qtde_steps = qtde_steps
        self.time_limit = (5 * 60 * 1000) #Change
        self.is_discrete = is_discrete
        
        self.done = False
        self.broken = False
        
        self.logger_path = logger_path
        self.update_interval = update_interval
        self.window_size = (history_size//update_interval)

        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=(self.window_size, 30),
                                     dtype=np.float32) # Change it
        if self.is_discrete:
            self.action_space = Discrete(9)
        else:
            self.action_space = Box(low=-1.0, high=1.0,
                                    shape=(3,),
                                    dtype=np.float32) # Change it
            
        transforms_set = VOC.Compose([
                        VOC.ToSegmentation(),
                        #ConvertLabel(),
                        VOC.ToTensor()]
                    )

        dataset          = VOC.VOCDetection(r'../Datasets/', transforms=transforms_set)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=None,
                                                    shuffle=True, num_workers=0)

    def start_agents(self):
        pass

    def check_agents(self):
        return True

    def stop_agents(self):
        pass

    def start(self):
        input, target = iter(data_loader).next()
        shape = input.shape
        #pred  = [int(shape[0] * 0.33), int(shape[0] * 0.66), int(shape[1] * 0.33), int(shape[1] * 0.66)] 
        self.history.start(pred)

        
    def stop(self):
        pass
    

    def _receive_state(self, action):
        
        self.history.update(action, reset=False)
        state = self.history.cont_states
        state = np.array(state)
        return state

    def write_log(self, is_first=False):
        with open(self.logger_path, 'a') as log:
            now = datetime.now()
            iou = self.history.cont_states[-1][1]
            steps = self.history.num_insertions
            log.write(f"{iou},{steps},{now}\n")
            
    def reset(self):
        self.broken = False
        is_first = not (self.fira and self.agent_yellow_process)
        self.write_log(is_first)

        self.start()
        self.history = History(self.history_size)
        state = self._receive_state(reset=True)
        return np.array(state)

    def bound(self, value, floor, ceil):
        if value < floor:
            return floor
        elif value > ceil:
            return ceil
        else:
            return value

    def compute_rewards(self, gt, proposed):
        
        r_iou   = self.history.cont_states[-1][1] - self.history.cont_states[-2][1]
        r_steps = self.history.num_insertions * (0.001)
        return r_iou - r_steps

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


class CoachEnvContinuous(CoachEnv):

    def __init__(self, render=False):
        super(CoachEnvContinuous, self).__init__(
            is_discrete=False, render=render)
