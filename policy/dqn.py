import argparse
import collections
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pathlib
import wandb
from gym_doll import Ol2015_Env
from models import Qnet, ConvQnet, device

random.seed(42)
# Hyperparameters
learning_rate = 0.0005
gamma = 0.98  # 0.9

buffer_limit = 500
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Agent:
    def __init__(self, num_input, actions, exp_name='squeeze', 
                 save_interval=100, update_interval=10, is_test=False):

        self.num_input       = num_input
        self.actions         = actions

        self.model           = Qnet(num_input, actions).to(device)
        self.target_model    = Qnet(num_input, actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer       = optim.Adam(self.model.parameters(), 
                                       lr=learning_rate)
        self.memory          = ReplayBuffer(buffer_limit)
        self.exp_name        = exp_name
        self.is_test         = is_test
        self.epsilon         = self.update_epsilon(0)
        self.save_interval   = save_interval
        self.update_interval = update_interval
        
    def update_epsilon(self, elapsed_steps=None):
        if self.is_test:
            self.epsilon = 0.01
        else:
            self.epsilon = 0.01 + (0.99 - 0.01) * \
                    np.exp(-1. * elapsed_steps / 5_000)


    def sample_action(self, state, elapsed_steps, n_epi, epi_steps, history):
        self.update_epsilon(elapsed_steps)
        if n_epi < 500:
            pos_action = history.get_good_actions()
        else:
            pos_action = []
        force_stop = False
        if epi_steps > 200:
            force_stop = True


        return self.model.sample_action(state, self.epsilon, pos_action, force_stop)
        
    def append(self, state, action, reward, s_prime, done_mask):
        self.memory.append((state, action, reward, s_prime, done_mask))

    def load(self, path='models/zero/DQN_best.pt'):
        
        checkpoint = torch.load(path,
                            map_location=lambda storage, loc: storage)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, n_episode):
        folder_path = f'models/{self.exp_name}//'
        model_name  = f'DQN_{n_episode:06d}.pt'

        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

        save_dict = {
            'epoch': n_episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': learning_rate,
            'gamma': gamma,
            'buffer_limit':buffer_limit
            }

        torch.save(save_dict, folder_path + 'DQN_best.pt')
        if n_episode % self.save_interval == 0:
            torch.save(save_dict, folder_path + model_name)

    def train(self, n_episode):
           
        losses = []
        for _ in range(2):
            state, action, reward, s_prime, done_mask = self.memory.sample(batch_size)
            state = state.to(device)
            action = action.to(device)
            reward = reward.to(device)
            s_prime = s_prime.to(device)
            done_mask = done_mask.to(device)

            n_inputs = state.size()[1]*state.size()[2]

            state = state.view(batch_size, n_inputs)
            s_prime = s_prime.view(batch_size, n_inputs)
            q_out = self.model(state)
            q_a = q_out.gather(1, action)
            max_q_prime = self.target_model(s_prime).max(1)[0].unsqueeze(1)
            target = reward + gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target) # Change for binary_cross_entropy
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_target(n_episode)

        return np.mean(losses)

    def update_target(self, n_episode):
        if n_episode % self.update_interval:
            self.target_model.load_state_dict(self.model.state_dict())


def main(load_model=False, test=False, use_render=False):
    try:
        if not test:
            wandb.init(name="DOLL-DQN-squeeze", project="DOLL")
        env      = gym.make('DOLL-v0')
        n_inputs = env.observation_space.shape[0] * \
            env.observation_space.shape[1]
        
        agent = Agent(n_inputs, env.action_space.n, is_test=test,
                      update_interval=10)

        if load_model:
            agent.load()

        elapsed_steps = 0
        for n_epi in range(200000):
            state, info = env.reset()
            done = False
            epi_steps = 0
            score = 0.0

            while not done:

                action = agent.sample_action(state, elapsed_steps, n_epi, epi_steps, info)
                s_prime, reward, done, info = env.step(action)
                done_mask = 0.0 if done else 1.0
                agent.append(state, action, reward, s_prime, done_mask)
                state  = s_prime
                score += reward
                elapsed_steps += 1
                epi_steps     += 1
                if done:
                    pass
                    #print('Reset')

            if agent.memory.size() > batch_size and not test:
                losses = agent.train(n_epi)
                wandb.log({'Loss/DQN': np.mean(losses)},
                          step=n_epi, commit=False)
                agent.save(n_epi)
                

            if not test:
                IOU   = env.history.hist_iou[-1]
                N_ins = env.history.num_insertions
                print(f"{n_epi:03d}| Score:{score: 3.2f} | IOU: {IOU:0.3f} | N: {N_ins:03d} | Epsilon: {agent.epsilon:0.3f}")
                wandb.log({'Rewards/total': score,
                           'Loss/epsilon': agent.epsilon,
                           'Rewards/IOU': IOU,
                           'Rewards/N': N_ins
                           }, step=n_epi)
        env.close()
    except Exception as e:
        env.close()
        raise e


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Predicts your time series')
    PARSER.add_argument('--test', default=False,
                        action='store_true', help="Test mode")
    PARSER.add_argument('--load', default=False,
                        action='store_true',
                        help="Load models from examples/models/")
    PARSER.add_argument('-r','--render', default=False,
                        action='store_true',
                        help="Use render")

    ARGS = PARSER.parse_args()
    if not os.path.exists('./models'):
        os.makedirs('models')

    main(load_model=ARGS.load, test=ARGS.test, use_render=ARGS.render)