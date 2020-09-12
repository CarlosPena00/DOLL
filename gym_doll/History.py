import collections
import numpy as np
import torch
import torchvision.models as tmodels
from torchvision import transforms as T
from sklearn.preprocessing import OneHotEncoder
from torch.nn import functional as F

class Stats:

    def __init__(self):
        self.eps = 1e-10

    def get_IOU(self, gt, pred):
        view = gt[0].cpu().numpy().copy()
        view[pred[0]:pred[1], pred[2]:pred[3]] += 1
        
        iou = (view==2).sum() / (view>=1).sum()
        return iou
        
class History:
    def __init__(self, MAX, alfa=0.2, image_size=(224, 224), num_action=9):
        
        self.image_size=image_size

        self.MAX = MAX
        self.num_action = num_action
        
        self.cont_states = collections.deque(maxlen=MAX)
        self.disc_states = collections.deque(maxlen=MAX)
        self.hist_iou    = collections.deque(maxlen=MAX)
        
        self.num_insertions = 0
        self.time   = 0
        self.stats  = Stats()
        self.data   = None
        self.input  = None
        self.target = None
        self.bbox   = None
        self.alfa   = alfa

        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        self.onehot_encoder.fit(np.array (range(self.num_action)).reshape(-1, 1))
        self._init_features()

    def _init_features(self):
        self.features = tmodels.squeezenet1_1(pretrained=True).eval().features 
        for para in self.features.parameters():
            para.requires_grad = False
        ones_in = torch.ones((1, 3)+self.image_size)
        state_shape = self.features(ones_in).reshape(-1).shape.numel()
        self.state_shape = state_shape + self.num_action

    def start(self, input, target):
        self.input = input
        self.target = target
        self.shape = self.input.shape[2:]
        
        self.bbox  = [int(self.shape[0] * 0.33), int(self.shape[0] * 0.66), 
                      int(self.shape[1] * 0.33), int(self.shape[1] * 0.66)] 
        iou = self.stats.get_IOU(self.target, self.bbox)

        hot_action = self.onehot_encoder.transform([[5]])[0]
        features = self.get_features()
        all_feat = np.concatenate((hot_action, features),axis=0)

        for _ in range(self.MAX):
            self.cont_states.append(all_feat)
        self.num_insertions = 0
        
    def ensure_bbox(self):
        
        self.bbox[0] = int(max(self.bbox[0], 0))
        self.bbox[1] = int(min(self.bbox[1], self.shape[0]-1))
        self.bbox[2] = int(max(self.bbox[2], 0))
        self.bbox[3] = int(min(self.bbox[3], self.shape[1]-1))
        
    def change_bbox(self, action):
        # right, left, up, down, bigger, smalller, fatter, taller , trigger
        ah = self.alfa * (self.bbox[1] - self.bbox[0])
        aw = self.alfa * (self.bbox[3] - self.bbox[2])
        
        if action == 0: # right
            self.bbox[2] += aw
            self.bbox[3] += aw
        elif action == 1: # left
            self.bbox[2] -= aw
            self.bbox[3] -= aw
        elif action == 2: # up
            self.bbox[0] -= ah
            self.bbox[1] -= ah
        elif action == 3: # down
            self.bbox[0] += ah
            self.bbox[1] += ah
        elif action == 4: # bigger
            self.bbox[0] -= ah/2
            self.bbox[1] += ah/2
            self.bbox[2] -= aw/2
            self.bbox[3] += aw/2
        elif action == 5: # smaller
            self.bbox[0] += ah/2
            self.bbox[1] -= ah/2
            self.bbox[2] += aw/2
            self.bbox[3] -= aw/2    
        elif action == 6: # fatter
            self.bbox[2] -= aw/2
            self.bbox[3] += aw/2
        elif action == 7: # taller
            self.bbox[0] -= ah/2
            self.bbox[1] += ah/2
        ## Should I add more ?

        self.ensure_bbox()
        return action == 8

    def get_features(self):
        # Batch, Chanel, Height, Width
        roi = self.input[:, :, :self.bbox[0]:self.bbox[1], self.bbox[2]:self.bbox[3]]
        roi = F.interpolate(roi, size=self.image_size[0])
        with torch.no_grad():
            features = self.features(roi)
            
            return features.reshape(-1).cpu().numpy()

    def update(self, action):
        
        self.change_bbox(action)
        iou = self.stats.get_IOU(self.target, self.bbox)
        self.hist_iou.append(iou)
        hot_action = self.onehot_encoder.transform([[action]])[0]
        features = self.get_features()
        all_feat = np.concatenate((hot_action, features),axis=0)

        self.cont_states.append(all_feat)

        self.num_insertions += 1
        
        
