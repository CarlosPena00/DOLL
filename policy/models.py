import collections
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tmodels

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class MixNet(nn.Module):
    """Based on Active Object Localization with Deep Reinforcement Learning
       With squeezenet1_1
    """
    def __init__(self, input_shape=(224, 224), actions=9):
        super(MixNet, self).__init__()
        
        self.features = tmodels.squeezenet1_1(pretrained=True).features 
        for para in self.features.parameters():
            para.requires_grad = False
        
        ones_in = torch.ones((1,)+input_shape)
        linear_input = self.features(ones_in).reshape(-1).shape.numel()

        self.actions = actions
        self.fc1 = nn.Linear(linear_input, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, actions)

    def forward(self, x):
        x = self.features(x).reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float().to(device)
        obs = obs.view(1, -1)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.actions-1)
        else:
            return out.argmax().item()

    
class ConvQnet(nn.Module):
    def __init__(self, num_input, actions=9, extra=0):
        super(ConvQnet, self).__init__()

        self.in_channels = num_input // (224 * 224)
        self.actions = actions
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4   = nn.Linear(36864 + extra, 512)
        self.fc    = nn.Linear(512, actions)
        
    def forward(self, x):
        x = x.reshape(-1, self.in_channels, 224, 224)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 36864)
        
        #x = torch.cat((hist.unsqueeze(0), x), dim=1)
        x = F.relu(self.fc4(x))
        return self.fc(x)
    
    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float().to(device)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.actions-1)
        else:
            return out.argmax().item()
    
    
class Qnet(nn.Module):
    def __init__(self, num_input, actions):
        super(Qnet, self).__init__()
        self.actions = actions
        self.num_input = num_input
        self.fc1 = nn.Linear(num_input, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon, pos_actions=[], force_stop=False):
        obs = torch.from_numpy(obs).float().to(device)
        obs = obs.view(1, self.num_input)
        out = self.forward(obs)
        if force_stop:
            return 8
        coin = random.random()
        if coin < epsilon:
            if len(pos_actions):
                return np.random.choice(pos_actions)

            return random.randint(0, self.actions-1)
        else:
            return out.argmax().item()

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
