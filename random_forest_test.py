# -*- coding: utf-8 -*-
"""
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pandas.plotting import table
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import sklearn
from torch.utils.data import Dataset

from torch.autograd import Variable
from tqdm import tqdm


class TabularMlp(nn.Module):
    def __init__(self,input_size,output_size):
        super(TabularMlp, self).__init__()
        hidden1 = 64
        hidden2 = 32
        self.processing_info = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True)
            )
        self.meta_fc = nn.Linear(32, output_size)
    def forward(self,X):
        m = self.processing_info(X)
        return self.meta_fc(m)
    
class TabDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]    

exc_dir = "..."

targets = ['RNG Number','Female','Bicuspid Aortic Valve',
           'History of Hypertension','age (yr)','bsa',
           'Ascending diameter (measured from surgical TEE) (mm)'
           ]
       
data = pd.read_excel(exc_dir).loc[:,targets]
y = pd.read_excel(exc_dir).loc[:,'Severity based on Diameter/BSA and Age']
#data = data.to_numpy()

#scaling age info
# =============================================================================
# fnames = ['953','3931']
# sub_sec = data[data['RNG Number'].isin(fnames)]
# sub_sec = sub_sec.reindex(index = fnames)
# print(sub_sec)
# =============================================================================
for ind in range(4,7):
    age_data = data.iloc[:,ind]
    age_data -= age_data.min()
    age_data /= age_data.max()
    data.iloc[:,ind] = age_data.astype('float')
         

test_rng_num = ['7252','2300','3901','5269','5722','6434','1384'] #Based on our folder train/test split
train_X = data[~ data['RNG Number'].isin(test_rng_num)].drop('RNG Number', axis=1).to_numpy()
test_X = data[data['RNG Number'].isin(test_rng_num)].drop('RNG Number', axis=1).to_numpy()


train_y = y[~ data['RNG Number'].isin(test_rng_num)].to_numpy()
train_y = np.where(train_y == -1, 0, train_y)
test_y = y[data['RNG Number'].isin(test_rng_num)].to_numpy()
test_y = np.where(test_y == -1, 0, test_y)

bs = 16
train_dataset = TabDataSet(train_X,train_y)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bs)

test_dataset = TabDataSet(test_X,test_y)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=bs)
model =  TabularMlp(6,3)


loss = nn.CrossEntropyLoss()  # standard crossentropy loss
optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
device = 'cpu'


for inputs,labels in train_loader:
    print(inputs.shape)
    print('labels shape',labels.shape)
    break




def run_epoch (loader, model, device ,loss_criterion, optimizer,train):
    
    running_loss = 0.0
    running_corrects = 0.

    for inputs, labels in tqdm(loader):
        if train:
            inputs = Variable(inputs.type(torch.float32), requires_grad=True).to(device)
            labels =  Variable(labels).type(torch.LongTensor).to(device)
            optimizer.zero_grad()

        else:
            inputs = inputs.type(torch.float32).to(device)
            labels = labels.type(torch.LongTensor).to(device)
        model = model.to(device)
        outputs = model(inputs)
        
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        
        loss = loss_criterion(outputs, labels)     
        if train:
            loss.backward()
            optimizer.step()
        else:
            print("Preds",preds)
            print("labels",labels)

            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    print("Comparision : .....................................")
    print("Preds",preds)
    print("labels",labels)
    return running_loss/len(loader.dataset), running_corrects/len(loader.dataset)



for i in range(100):
    r_loss,r_acc = run_epoch(train_loader,model,device,loss,optimizer,True)
    print("Epoch {} Running loss {} and running accuracy {}" .format(i,r_loss,r_acc))
    torch.save(model.state_dict(), "tabular_weightsoct22.pth")

r_loss,r_acc = run_epoch(test_loader,model,device,loss,optimizer,False)
print("Test loss {} and Test accuracy {}" .format(r_loss,r_acc))   

















####################### Random Forest experiment
# =============================================================================
# for depth in range(1, 15):
#     forest =RandomForestClassifier(max_depth=depth, n_estimators=500,
#                                                      random_state=0)
#     forest.fit(train_X, train_y)
# 
#     accuracy_train = sklearn.metrics.accuracy_score(train_y, forest.predict(train_X))
#     accuracy_test  = sklearn.metrics.accuracy_score(test_y, forest.predict(test_X))
#     
#     print("max depth ",depth, " training accuracy ",accuracy_train," testing accuracy ",accuracy_test)
# 
#     
# =============================================================================
####################### Displaying correlation
# =============================================================================
# correlations = data.corr()
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
#             square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
# plt.show();
# =============================================================================
