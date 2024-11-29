# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from network import Res_Nl_Net
from collections import OrderedDict
from torchvision import models
from torchsummary import summary


from network.ResNetImplementation.my_vision.torchvision.models import resnet


class ConvLstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class, att_blocks = False):
        #n_class is number of binary output classes / bins in our case 3 
        # INPUT SHAPE : Batch,TimeSteps,C,H,W
        super(ConvLstm, self).__init__()
        if att_blocks == True:
            self.conv_model = Pretrained_att_conv(latent_dim)
        else:
            self.conv_model = Pretrained_conv(latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        
        #self.pool_3d = pool_3d(hidden_size,latent_dim)
        
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_size if bidirectional==True else hidden_size, n_class),
        )
        #self.pool_output = nn.Linear(14,n_class)
        
    def forward(self, x):
        #INPUT SHAPE Batch,Chan,time,H,W
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.reshape(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)


        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output = self.Lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        output = self.output_layer(lstm_output)
# =============================================================================
#       pool = False
#       if pool: 
#           pool_input = conv_output.view(batch_size, timesteps, -1)
#           pool_out = self.pool_3d(pool_input).reshape(batch_size,-1)
#           return self.pool_output(pool_out)
# =============================================================================
        
        return output

        

class Pretrained_conv(nn.Module):
    def __init__(self, latent_dim):
        super(Pretrained_conv, self).__init__()
        #self.conv_model = resnet.resnet18(pretrained=True)
        #st_dict = torch.load
        #self.conv_model.load_state_dict(st_dict)
        
        self.conv_model = models.resnet18(pretrained=True)
        # ====== freezing all of the layers ======
        for param in self.conv_model.parameters():
            param.requires_grad = False
        # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
        self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)

    def forward(self, x):
        return self.conv_model(x)
    
    
class Pretrained_att_conv(nn.Module):
    def __init__(self, latent_dim):
        super(Pretrained_att_conv, self).__init__()
        
        nl_net =  Res_Nl_Net.resnet2D18(non_local = True)
        #LOAD PRETRAINED RESNET WEIGHTS 
        # =============================================================================

        state_dict = torch.load("...")['state_dict']
        new_state_dict = OrderedDict()
        count=0
        count2=0
        for k,v in state_dict.items():
            new_k = k.split('module.')[1] 
            
            if new_k in nl_net.state_dict():
                new_state_dict[new_k] = v
                count = count + 1


        nl_net.load_state_dict(new_state_dict, strict=False) #
        print("PreLoaded ",count," Model Parameters")

        for name,param in nl_net.named_parameters():
            if not 'layer2.2.'in name:
                param.requires_grad = False
                count2 = count2+1
                
            
        print("froze  ",count2," Model Parameters")
        self.conv_model = nl_net
        num_ftrs = self.conv_model.linear.in_features
        self.conv_model.linear = nn.Linear(num_ftrs, latent_dim)
        
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


class pool_3d(nn.Module):
    def __init__ (self,out_size,kernel_size):
        super(pool_3d, self).__init__()
        self.pool3 =  nn.AvgPool2d(36)
        self.out_size = out_size
        
    def forward(self,x):
        output = self.pool3(x)

        return output
        
        