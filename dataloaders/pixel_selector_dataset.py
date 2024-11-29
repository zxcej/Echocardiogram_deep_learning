# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from collections import deque

class PairDataset(Dataset):
    def __init__(self, Xs, ys):
        self.Xs = Xs
        self.ys = ys
        return 
        
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):
        data = self.Xs[idx]
        label = self.ys[idx]
        return data, label