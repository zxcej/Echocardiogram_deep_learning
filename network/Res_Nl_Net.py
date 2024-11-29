# -*- coding: utf-8 -*-
"""
"""

'''
Non-Local ResNet2D-50 for CIFAR-10 dataset.
Most of the code is borrowed from https://github.com/akamaster/pytorch_resnet_cifar10

Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from network import non_local_net
#import non_local_net
from collections import OrderedDict
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, non_local=False):
        super(ResNet2D, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        
        # add non-local block after layer 2
        self.layer2, self.non_local_net = self._make_layer(block, 32, num_blocks[1], stride=2, non_local=non_local)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        

    def _make_layer(self, block, planes, num_blocks, stride, non_local=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        last_idx = len(strides)
        if non_local:
            last_idx = len(strides) - 1

        for i in range(last_idx):
            layers.append(block(self.in_planes, planes, strides[i]))
            self.in_planes = planes * block.expansion
        nl_net = None
        if non_local:
            nl_net = non_local_net.NLBlockND(in_channels=planes, dimension=2)
            self.non_local_net = nl_net
            layers.append(nl_net)
            layers.append(block(self.in_planes, planes, strides[-1]))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)[0]
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet2D18(non_local=True, **kwargs):
    """Constructs a ResNet-56 model.
    """
    return ResNet2D(BasicBlock, [3, 3, 3], non_local=non_local, **kwargs) #RES18
    #return ResNet2D(BasicBlock, [9, 9, 9], non_local=non_local, **kwargs)


if __name__=='__main__':
    # Test case for (224 x 224 x 3) input of batch size 1
    img = Variable(torch.randn(16, 3, 256, 256))
    net = resnet2D18(non_local=True)
    
    state_dict = torch.load("...")['state_dict']
    #print(state_dict)
    new_state_dict = OrderedDict()
    count1 = 0
    for k,v in state_dict.items():
        new_k = k.split('module.')[1] 
        if new_k in net.state_dict():
            new_state_dict[new_k] = v
            count1 = count1+1
        else:
            print("missibg",new_k)
    print("Loaded %d Paraneters"%(count1))
    net.load_state_dict(new_state_dict, strict=False)
    
    for param in net.parameters():
        param.requires_grad = False
    num_ftrs = net.linear.in_features
    print(num_ftrs)
    net.linear = nn.Linear(num_ftrs, 512)
    net(img)

