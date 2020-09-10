import collections
import numpy as np


class Stats:

    def __init__(self):
        self.eps = 1e-10

    def get_IOU(self, gt, pred):
        view = gt.copy()
        view[pred[0]:pred[1], pred[2]:pred[3]] += 1
        
        iou = (view==2).sum() / (view>=1).sum()
        return iou
        
class History:
    def __init__(self, MAX, alfa=0.2):
        self.MAX = MAX
        
        self.cont_states = collections.deque(maxlen=MAX)
        self.disc_states = collections.deque(maxlen=MAX)
        
        self.num_insertions = 0
        self.time = 0
        self.data = None
        self.stats = Stats()
        self.gt = None
        self.bbox = None
        self.alfa = alfa
        #self.gt = np.zeros([400, 600])
        #self.gt[50:150, 340:390] = 1
        #shape = self.gt.shape
        #self.bbox  = [int(shape[0] * 0.33), int(shape[0] * 0.66), int(shape[1] * 0.33), int(shape[1] * 0.66)] 
        
    def start(self, gt):
        self.gt = gt
        shape = self.gt.shape
        self.bbox  = [int(shape[0] * 0.33), int(shape[0] * 0.66), int(shape[1] * 0.33), int(shape[1] * 0.66)] 
        iou = self.Stats.get_IOU(gt, bbox)
        for _ in range(self.MAX):
            self.cont_states.append([5, iou]) # TODO
        self.num_insertions = 0
        
    def ensure_bbox(self):
        shape = self.gt.shape
        self.bbox[0] = int(max(self.bbox[0], 0))
        self.bbox[1] = int(min(self.bbox[1], shape[0]-1))
        self.bbox[2] = int(max(self.bbox[2], 0))
        self.bbox[3] = int(min(self.bbox[3], shape[1]-1))
        
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


    def update(self, action):
        
        cont_state = []
        self.change_bbox(action)
        iou = self.stats.get_IOU(self.gt, self.bbox)
            
        self.cont_states.append([action, iou])

        self.num_insertions += 1
        
        
