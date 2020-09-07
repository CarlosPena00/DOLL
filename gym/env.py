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

from gym_coach_vss.fira_parser import FiraParser
from gym_coach_vss.Game import History, Stats
import random

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'


w_grad_ball_potential = (0.08, 1)


class Env(gym.Env):

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
            self.action_space = Discrete(27)
        else:
            self.action_space = Box(low=-1.0, high=1.0,
                                    shape=(3,),
                                    dtype=np.float32) # Change it

    def start_agents(self):
        pass

    def check_agents(self):
        return True

    def stop_agents(self):
        pass

    def start(self):
        self.start_agents()
        
    def stop(self):
        pass

    def _receive_state(self, reset=False):
        #data = ? update here
        data = 0
        self.history.update(data, reset=reset)
        state = self.history.cont_states
        state = np.array(state)
        state = state[self.update_interval-1::self.update_interval]
        return state

    def write_log(self, is_first=False):
        with open(self.logger_path, 'a') as log:
            now = datetime.now()
            log.write(f"{now}\n")
            
    def reset(self):
        self.broken = False
        is_first = not (self.fira and self.agent_yellow_process)
        self.write_log(is_first)

        if not is_first:
            print(f"Coach {self.goal_prev_yellow}\
                 x {self.goal_prev_blue} {self.versus}")
            self.stop()

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

    def compute_rewards(self):
        reward = 0.0

        return reward

    def step(self, action):
        self.done = False
        reward = 0
        out_str = struct.pack('i', int(action))
        self.sw_conn.sendto(out_str, ('0.0.0.0', 4098))
        for _ in range(self.qtde_steps):
            state = self._receive_state()
            reward += self.compute_rewards()
            if not self.check_agents():
                self.broken = True
                self.done = True
                self.history.time = self.prev_time = 0.0

            if self.done:
                break
        return state, reward, self.done, self.history

    def render(self, mode='human'):
        pass

    def close(self):
        self.stop()


class CoachEnvContinuous(CoachEnv):

    def __init__(self, render=False):
        super(CoachEnvContinuous, self).__init__(
            is_discrete=False, render=render)
