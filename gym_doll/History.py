import collections
import numpy as np
import torch
import torchvision.models as tmodels
from torchvision import transforms as T
from sklearn.preprocessing import OneHotEncoder
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url
from matplotlib import pyplot as plt
from policy.models import AlexNet

MIM_SIZE = 24

class Stats:

    def __init__(self):
        self.eps = 1e-10

    def get_IOU(self, gt, pred):
        view = gt[0].cpu().numpy().copy()
        view[pred[0]:pred[1], pred[2]:pred[3]] += 1
        
        iou = (view==2).sum() / (view>=1).sum()
        return iou
    

def forward_feat(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    return x

def ensure_bbox(bbox, shape):
    
    if bbox[1] - bbox[0] < MIM_SIZE:
        bbox[1] += MIM_SIZE//4
        bbox[0] -= MIM_SIZE//4
    if bbox[3] - bbox[2] < MIM_SIZE:
        bbox[3] += MIM_SIZE//4
        bbox[2] -= MIM_SIZE//4


    bbox[0] = int(max(bbox[0], 0))
    bbox[1] = int(min(bbox[1], shape[0]-1))
    bbox[2] = int(max(bbox[2], 0))
    bbox[3] = int(min(bbox[3], shape[1]-1))
    if bbox[0] == bbox[1]:
        bbox[0] = (max(bbox[0]-1, 0))
        bbox[1] = (min(bbox[1]+1, shape[0]-1))
    if bbox[2] == bbox[3]:
        bbox[2] = max(bbox[2]-1, 0)
        bbox[3] = min(bbox[3]+1, shape[1]-1)

    return bbox

class FakeModel():
    def __init__(self):
        pass
    def __call__(self, x):
        return self.forward_feat(x)

    def forward_feat(self, x):
        return x.view(-1)

 
class History:
    def __init__(self, MAX, alfa=0.2, image_size=(224, 224), 
                 num_action=9, action_per_state=10, roi_as_state=False):

        self.roi_as_state     = roi_as_state
        self.action_per_state = action_per_state
        self.image_size  = image_size
        self.MAX         = MAX
        self.num_action  = num_action
        self.cont_states = collections.deque(maxlen=MAX)
        self.disc_states = collections.deque(maxlen=MAX)
        self.hist_iou    = collections.deque(maxlen=10)
        self.hist_bbox   = collections.deque(maxlen=MAX+1)
        self.hist_hact   = collections.deque(maxlen=self.action_per_state)

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
        
        if self.roi_as_state:
            self._init_features_image()
        else:
            # self._init_features_resnet()
            #self._init_features_alexnet()
            self._init_features_squeeze()

    def _init_features_image(self):
        self.state_shape =  (self.image_size[0] * self.image_size[1] * 3)
        
    def _init_features_fake(self):
        self.features = FakeModel
        ones_in       = torch.ones((1, 3)+self.image_size)
        state_shape   = self.features.forward_feat(self.features, ones_in).reshape(-1).shape.numel()
        self.state_shape = state_shape + (self.num_action * self.action_per_state)


    def _init_features_resnet(self):
        self.features = tmodels.resnet18(pretrained=True)
        
        setattr(self.features, 'forward_feat', forward_feat)
        
        for para in self.features.parameters():
            para.requires_grad = False
        ones_in = torch.ones((1, 3)+self.image_size)
        state_shape = self.features.forward_feat(self.features, ones_in).reshape(-1).shape.numel()
        self.state_shape = state_shape + (self.num_action * self.action_per_state)

    def _init_features_squeeze(self):
        self.features = tmodels.squeezenet1_1(pretrained=True).eval().features 
        for para in self.features.parameters():
            para.requires_grad = False
        ones_in = torch.ones((1, 3)+self.image_size)
        state_shape = self.features(ones_in).reshape(-1).shape.numel()
        self.state_shape = state_shape + (self.num_action * self.action_per_state)

    def _init_features_alexnet(self):
        self.features = AlexNet().eval()
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                                              progress=True)
        model_dict = self.features.state_dict()

        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict) 
        self.features.load_state_dict(state_dict)
        for para in self.features.parameters():
            para.requires_grad = False
        ones_in = torch.ones((1, 3)+self.image_size)
        state_shape = self.features(ones_in).reshape(-1).shape.numel()
        self.state_shape = state_shape + (self.num_action * self.action_per_state)


    def start(self, input, target):
        self.input  = input
        self.target = target
        self.shape  = self.input.shape[2:]
        self.bbox   = [int(self.shape[0] * 0.1), int(self.shape[0] * 0.9), 
                       int(self.shape[1] * 0.1), int(self.shape[1] * 0.9)] 
        iou         = self.stats.get_IOU(self.target, self.bbox)

        #hot_action = self.onehot_encoder.transform([[5]])[0]
        hot_action  = np.zeros([self.num_action * self.action_per_state])

        features    = self.get_features()
        
        if self.roi_as_state:
            all_feat = features
        else:
            all_feat = np.concatenate((hot_action, features),axis=0)

        for _ in range(self.action_per_state):
            self.hist_hact.append( np.zeros(self.num_action) )

        for _ in range(self.MAX):
            self.cont_states.append(all_feat)
            self.hist_iou.append(iou)
            self.hist_bbox.append(self.bbox)

        self.num_insertions = 0
        
        
    def change_bbox(self, action, true_action=True):
        # right, left, up, down, bigger, smalller, fatter, taller , trigger
        bbox = self.bbox.copy()
        
        ah = self.alfa * (bbox[1] - bbox[0])
        aw = self.alfa * (bbox[3] - bbox[2])
        
        if action == 0: # right
            bbox[2] += aw
            bbox[3] += aw
        elif action == 1: # left
            bbox[2] -= aw
            bbox[3] -= aw
        elif action == 2: # up
            bbox[0] -= ah
            bbox[1] -= ah
        elif action == 3: # down
            bbox[0] += ah
            bbox[1] += ah
        elif action == 4: # bigger
            bbox[0] -= ah/2
            bbox[1] += ah/2
            bbox[2] -= aw/2
            bbox[3] += aw/2
        elif action == 5: # smaller
            bbox[0] += ah/2
            bbox[1] -= ah/2
            bbox[2] += aw/2
            bbox[3] -= aw/2    
        elif action == 6: # fatter
            bbox[2] -= aw/2
            bbox[3] += aw/2
        elif action == 7: # taller
            bbox[0] -= ah/2
            bbox[1] += ah/2
        ## Should I add more ?

        bbox = ensure_bbox(bbox, self.shape)
        if true_action:
            self.bbox = bbox

        return (action == 8), bbox

    def get_good_actions(self):
        actual_IOU = self.hist_iou[-1]
        good_action_list = []

        if actual_IOU >= 0.6:
            good_action_list.append(8)

        #print("Baseline: ",actual_IOU)
        for act_id in range(8):
            done, bbox = self.change_bbox(act_id, true_action=False)
            iou_id     = self.stats.get_IOU(self.target, bbox)
            if iou_id >= actual_IOU:
                good_action_list.append(act_id)

        #print(good_action_list)
        return good_action_list

    def get_roi(self):
        ymin = max(self.bbox[0]-8, 0)
        xmin = max(self.bbox[2]-8, 0)
        ymax = min(self.bbox[1]+8, self.shape[0]-1)
        xmax = min(self.bbox[3]+8, self.shape[1]-1)
        roi = self.input[:, :, ymin:ymax, xmin:xmax]
        return roi

    def get_features(self):
        # Batch, Chanel, Height, Width
        with torch.no_grad():
            roi = self.get_roi()
            roi = F.interpolate(roi, size=self.image_size[0])
            
            if self.roi_as_state:
                return roi.reshape(-1).cpu().numpy()

            #features = self.features.forward_feat(self.features, roi)
            features = self.features.forward(roi)
            return features.reshape(-1).cpu().numpy()

    def update(self, action):
        if not self.change_bbox(action)[0]:
            self.hist_bbox.append(self.bbox.copy())
            
        iou        = self.stats.get_IOU(self.target, self.bbox)
        hot_action = self.onehot_encoder.transform([[action]])[0]
        features   = self.get_features()

        self.hist_iou.append(iou)
        self.hist_hact.append(hot_action)

        if self.roi_as_state:
            self.cont_states.append(features)

        else:
            hact = np.array(self.hist_hact)[-self.action_per_state:].ravel()
            all_feat = np.concatenate((hact, features),axis=0)
            self.cont_states.append(all_feat)

        self.num_insertions += 1
        
        
