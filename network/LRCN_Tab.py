# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
from collections import OrderedDict
from network.ResNetImplementation.my_vision.torchvision.models import resnet


class ConvLstmTab(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class, meta_categ ):
        # INPUT SHAPE : Batch,TimeSteps,C,H,W
        super(ConvLstmTab, self).__init__()
        self.conv_model = Pretrained_conv(latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        meta_out = 32
        #meta_out = 32
        self.meta_model = TabularMlp(meta_categ, meta_out)
        
        self.output_layer = nn.Sequential(
            nn.Linear((2 * hidden_size if bidirectional==True else hidden_size)+meta_out, n_class), #+meta_out

        )
        
        self.meta_fc = nn.Linear(meta_out, n_class)
        self.lstm_fc = nn.Linear(latent_dim,n_class)

    def forward(self, x):
        #INPUT SHAPE Batch,Chan,time,H,W
        img_buff, meta_data = x[0], x[1]
        batch_size, timesteps, channel_x, h_x, w_x = img_buff.shape
        
        conv_input = img_buff.reshape(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output = self.Lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        meta_output = self.meta_model(meta_data).flatten(1) #.flatten
        #CONCAT FUSION
        fusion = torch.cat([lstm_output, meta_output], dim=1)   
        #SE FUSION
        #fusion = lstm_output*meta_output   


        output = self.output_layer(fusion).squeeze(1)

        return output

class Pretrained_conv(nn.Module):
    def __init__(self, latent_dim):
        super(Pretrained_conv, self).__init__()
        self.conv_model = resnet.resnet18(pretrained=True)
        for param in self.conv_model.parameters():
            param.requires_grad = False
        num_ftrs = self.conv_model.fc.in_features
        self.conv_model.fc = nn.Linear(num_ftrs, latent_dim)

    def forward(self, x):
        return self.conv_model(x)


class Lstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        self.Lstm = nn.LSTM(latent_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self,x):
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output

class TabularMlp(nn.Module):
    def __init__(self,input_size,output_size):
        super(TabularMlp, self).__init__()
        hidden1 = 64
        hidden2 = output_size
        self.processing_info = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True)
            )
        ###Because we will be loading pretrained weights in train.py
# =============================================================================
#         for param in self.processing_info.parameters():
#             param.requires_grad = False
# =============================================================================
    def forward(self,X):
        return self.processing_info(X)
    
    
