# -*- coding: utf-8 -*-
"""
"""
from torch.utils.data import DataLoader
import torch.utils.data.sampler as Sampler
import torch
import torchvision
import torch.nn as nn
#from tabular_network import tab_model
import sklearn
import sklearn.cluster         # For KMeans classfrom network import LRCN_Tab
from torch.autograd import Variable
from tqdm import tqdm

import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.datasets import load_boston
from dataloaders.dataset import VideoDataset




train_dataset = VideoDataset(dataset='cardiac_mini_vids', split='train',  clip_len=16)
train_dataset = VideoDataset(dataset='cardiac_mini_vids', split='test',  clip_len=16)

# =============================================================================
# class Dataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, X, y):
#         'Initialization'
#         self.X = X
#         self.y = y
# 
#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.X)
# 
#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         x = torch.from_numpy(self.X[index])
#         y = torch.from_numpy(np.asarray( self.y[index] ))
#         return x,y 
# def impl_weighted_sampler(train_dataset,test_dataset, batch, sample_test = False):
#     train_target =  torch.tensor([samp[1].item() for samp in train_dataset])
#     test_target =  torch.tensor([samp[1].item() for samp in test_dataset])
#     print('Previous target train class distribution 0/1/2: {}/{}/{}'.format(
#         (train_target == 0).sum(), (train_target == 1).sum(), (train_target == 2).sum()))
#     print('Previous target test class distribution 0/1/2: {}/{}/{}'.format(
#         (test_target == 0).sum(), (test_target == 1).sum(), (test_target == 2).sum()))
#     
#     # Computing Training dataset Sampler 
#     train_class_sample_count = torch.tensor(
#         [(train_target == t).sum() for t in torch.unique(train_target, sorted=True)])
#     train_weight = 1. / train_class_sample_count.float()
#     train_samples_weight = torch.tensor([train_weight[t] for t in train_target])
# 
#     # finally the train sampler and loader
#     train_sampler = Sampler.WeightedRandomSampler(train_samples_weight,
#                                                   len(train_samples_weight))
#     
#     train_dataloader = DataLoader(train_dataset,
#                                   batch_size=batch,sampler=train_sampler)
#     my_labels = {'0':0,'1':0,'2':0}
#     for samp in train_dataloader:
#         my_labels['0'] = my_labels['0']+(samp[1] == 0).sum().item()
#         my_labels['1'] = my_labels['1']+(samp[1] == 1).sum().item()
#         my_labels['2'] = my_labels['2']+(samp[1] == 2).sum().item()
#         
#     print("After Sampling train class distributions 0/1/2: {}/ {}/ {}".format(my_labels['0'], my_labels['1'], my_labels['2']))
# 
#     if sample_test:
#         # Computing test dataset Sampler 
#         test_class_sample_count = torch.tensor(
#             [(test_target == t).sum() for t in torch.unique(test_target, sorted=True)])
#         test_weight = 1. / test_class_sample_count.float()
#         test_samples_weight = torch.tensor([test_weight[t] for t in test_target])
#     
#         # finally the train sampler and loader
#         test_sampler = Sampler.WeightedRandomSampler(test_samples_weight, 
#                                                      len(test_samples_weight))
# 
#         test_dataloader = DataLoader(test_dataset,
#                                          batch_size=batch, sampler=test_sampler)
#     else:
#         test_dataloader = DataLoader(test_dataset, batch_size=batch)
#         
# 
#     return train_dataloader, test_dataloader
# def load_tabular_data():
#     #features = ['age (yr)','Aortic diamter/BSA']
#     features = ['Energy Loss (mean of both axis)','age (yr)','bsa','Systolic Pressure (mmHg)','Diastolic pressure (mmHg)','History of Hypertension','Diabetes (I/II)','Presents with NYHA Heart Failure Symptoms','Ascending diameter (measured from surgical TEE) (mm)']
#     target = 'Severity based on Diameter/BSA and Age'
#     #target = 'Energy Loss (mean of both axis)'
#     exc_dir = "ex_vivo_analysis_values2.0 - Copy.xlsx"
#     data = pd.read_excel(exc_dir)
#     
#     class2idx = {
#         -1:2
#     }
#     
#     idx2class = {v: k for k, v in class2idx.items()}
#     
#     data[target].replace(class2idx, inplace=True)
#      
#      
#     inputs_array = data.loc[:,features].dropna().to_numpy()
#     targets_array = data.loc[:,target].dropna().to_numpy()
#     
#     
#     return inputs_array, targets_array
# 
# 
# #X, y = sklearn.datasets.load_wine(return_X_y=True)
#  
# X,y = load_tabular_data()
# print("X shape",X.shape," y shape ",y.shape)
# print(y)
# 
# 
# 
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=5)
# scaler = sklearn.preprocessing.MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_train, y_train = np.array(X_train), np.array(y_train)
# X_test, y_test = np.array(X_test), np.array(y_test)
# 
# 
# pytorch_train_dataset = Dataset(X_train,y_train)
# pytorch_test_dataset = Dataset(X_test,y_test)
# #train_loader, test_loader = impl_weighted_sampler(pytorch_train_dataset, pytorch_test_dataset, 12)
# train_loader, test_loader = DataLoader(pytorch_train_dataset, batch_size=12),DataLoader(pytorch_test_dataset, batch_size=12)
# model = tab_model(9,3)
# #model = tab_model(13,3)
# 
# ############# CST
# epochs = 300
# device = 'cuda:0'
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
# loss = nn.CrossEntropyLoss() # standard crossentropy loss
# ########
# 
# def run_epoch (loader, model, device ,loss_criterion, optimizer,train):
#     
#     running_loss = 0.0
#     running_corrects = 0.
# 
#     for inputs, labels in tqdm(loader):
#         if train:
#             inputs = Variable(inputs.type(torch.float32), requires_grad=True).to(device)
#             labels =  Variable(labels).type(torch.LongTensor).to(device)
#             optimizer.zero_grad()
# 
#         else:
#             inputs = inputs.type(torch.float32).to(device)
#             labels = labels.type(torch.LongTensor).to(device)
#         model = model.to(device)
#         outputs = model(inputs)
# 
#         #outputs = model(inputs)
#         
#         probs = nn.Softmax(dim=1)(outputs)
#         preds = torch.max(probs, 1)[1]
#         
#         loss = loss_criterion(outputs.squeeze(), labels)     
#         if train:
#             loss.backward()
#             optimizer.step()
#         else:
#             print("Preds",preds)
#             print("labels",labels)
# 
#             
#         running_loss += loss.item() * inputs.size(0)
#         running_corrects += torch.sum(preds == labels.data)
#     print("Comparision : .....................................")
#     print("Preds",preds)
#     print("labels",labels)
#     return running_loss/len(loader.dataset), running_corrects/len(loader.dataset)
# 
# 
# 
# for i in range(100):
#     r_loss,r_acc = run_epoch(train_loader,model,device,loss,optimizer,True)
#     print("Epoch {} Running loss {} and running accuracy {}" .format(i,r_loss,r_acc))
#     torch.save(model.state_dict(), "tabular_weights.pth")
# 
# r_loss,r_acc = run_epoch(test_loader,model,device,loss,optimizer,False)
# print("Test loss {} and Test accuracy {}" .format(r_loss,r_acc))   
# 
# 
# 
# =============================================================================




# =============================================================================
# lin_model = LinearRegression()
# lin_model.fit(X_train, y_train)
# # model evaluation for testing set
# y_test_predict = lin_model.predict(X_test)
# rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
# r2 = sklearn.metrics.r2_score(y_test, y_test_predict)
# 
# print("The model performance for testing set")
# print("--------------------------------------")
# print('RMSE is {}'.format(rmse))
# print('R2 score is {}'.format(r2))
# =============================================================================
class FusionNet_SEMul(nn.Module):
    """
    SE-approach
    """
    def __init__(self, model, meta_in=3, meta_out=512, dropout=0.3, hidden1 = 5, dropout1=0.5):
        super(FusionNet_SEMul, self).__init__()
        print(meta_in, hidden1, dropout, meta_out, dropout1)
        self.img_extractor = nn.Sequential(*list(model.children())[:-1], nn.Dropout(p=dropout)) # only the extractor part of the pretrained model 
        num_features = list(model.modules())[-1].in_features
        self.img_extractor.requires_grad_(False) # set the gradients false from the start
  
        self.metadata_extractor = nn.Sequential(
            nn.Linear(meta_in, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
          
            nn.Linear(hidden1, meta_out),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(meta_out),
            nn.ReLU(inplace=True) 
        )      

        self.fc = nn.Sequential(nn.Dropout(p=dropout1),nn.Linear(num_features, 1)) 
        
    def forward(self, img, meta): 
        img = self.img_extractor(img).flatten(1)
        print("img shape ",img.shape)
        
        meta = self.metadata_extractor(meta).flatten(1).sigmoid()  
        print("meta shape ",meta.shape)
        fusion = img*meta   
        x = self.fc(fusion).squeeze(1).sigmoid() # remove unnecessary dimension 
        return x 
    
mod = torchvision.models.resnet18(pretrained=True)
net = FusionNet_SEMul(model=mod)

img_buff = torch.rand(8,3,250,250)

meta = torch.rand(8,3)
out = net(img_buff,meta)










