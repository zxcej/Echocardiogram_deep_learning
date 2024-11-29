# -*- coding: utf-8 -*-
"""
"""
import torch
import torch.nn as nn

class tab_model(nn.Module):
    def __init__(self,input_size,output_size):
        super(tab_model, self).__init__()

        hidden1 = 64
        hidden2 = 32
        self.processing_info = nn.Sequential(     #layer order: L -> BN -> ReLU: Zhang2020: https://arxiv.org/pdf/2004.08955.pdf; He2016: https://arxiv.org/pdf/1603.05027.pdf
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(hidden2, output_size)

    def forward(self,X):
        out =  self.processing_info(X)
        return self.fc(out)
    