# -*- coding: utf-8 -*-
"""
IMPLEMENATION OF PIXEL CNN of : 
"""

import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

  
    
class selection_by_cnn_1(Module):
    
    def __init__(self, num_channel, h_channel, rSeed):
        super(selection_by_cnn_1, self).__init__()

        torch.manual_seed(rSeed)    
        self.conv1 = nn.Conv2d(num_channel, h_channel, (3, 3), stride=1, padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv2d(h_channel, 1, (1, 1), stride=1, padding=(0, 0))
        
    def forward(self, x0):
        
        x1 = self.relu1(self.conv1(x0))     
        x2 = torch.sigmoid(self.conv2(x1))
        x2 = x2.squeeze()
        
        return x2 