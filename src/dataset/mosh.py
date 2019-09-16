import sys
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from utils.util import *

class Mosh(Dataset):
    def __init__(self, data_set_path, use_flip = True, flip_prob = 0.3):
        self.data_folder = data_set_path
        self.use_flip = use_flip
        self.flip_prob = flip_prob

        self._load_data_set()

    def _load_data_set(self):
        print('start loading mosh data.')
        anno_file_path = os.path.join(self.data_folder, 'mosh_annot.h5')
        with h5py.File(anno_file_path) as fp:
            self.shapes = np.array(fp['shape'])
            self.poses = np.array(fp['pose'])
        print('finished load mosh data, total {} samples'.format(len(self.poses)))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        trival, pose, shape = np.zeros(3), self.poses[index], self.shapes[index]

        if self.use_flip and random.uniform(0, 1) <= self.flip_prob:
            pose = reflect_pose(pose)

        return {
            'theta': torch.tensor(np.concatenate((trival, pose, shape), axis = 0)).float()
        }

if __name__ == '__main__':
    mosh = Mosh('/export/home/suny/dataset/mosh_gen')
    l = len(mosh)
    import time
    for _ in range(l):
        r = mosh.__getitem__(_)
        print(r)