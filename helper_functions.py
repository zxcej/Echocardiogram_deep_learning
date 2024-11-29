# -*- coding: utf-8 -*-
"""

This file is for all the helper functions which are to be used in training

"""
import os
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torchvision.transforms.functional as F
import torch
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.utils.data.sampler as Sampler


from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import R3D_model
from network import LRCN_model
from network import C3D_model
from network import pixel_selector

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


import seaborn as sn
import pandas as pd


def compute_saliency_maps( X, y, model, device):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.
    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    
    # Make input tensor require gradient
    model.train()

    X.requires_grad_()
    
    saliency = None
    #forward pass
    scores = model(X)
    scores = (scores.gather(1, y.view(-1, 1)).squeeze())
    
    if len(scores.size()) ==0:
        scores = scores.unsqueeze(0)
    scores.backward((torch.FloatTensor([1.0]*scores.shape[0])).to(device))

    #saliency
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    model.eval()

    return saliency 
def show_saliency_maps( X, y, model,device):
    # Convert X and y from numpy arrays to Torch Tensors
    #X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    X_tensor = X
    y_tensor = y

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model,device)

    print(saliency.shape)
    print(X.shape)
    print(y.shape)
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.cpu().numpy()
    X = X.detach().cpu()
    X= X[:,0,:,:,:]
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        echo_img =  F.to_pil_image(X[i])
        echo_img = np.asarray(echo_img)
        plt.imshow(echo_img)
        plt.axis('off')
        #plt.title("Class: "+str([y[i].item()]) )
        plt.subplot(2, N, N + i + 1)
        print(saliency[i].transpose(1,2,0)*255)
        plt.imshow(saliency[i].transpose(1,2,0)*255, cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

def plot_results(epochs,losses,accs):
    """
        Args:
            epochs (int): number of epochs to compare to
            losses (list): list of losses throughout the
            range of epochs above
            accs (list): list of accuracies throughout the
            range of epochs above    
    """
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2,1)
    plt.plot(range(epochs),losses)
    plt.title("Epochs vs Training losses")
    plt.subplot(1, 2,2)
    plt.plot(range(epochs),accs)
    plt.title("Epochs vs Training Accuracy")

    
def show_grid(img_grid, sample_labels, title):
    if not isinstance(img_grid, list):
        img_grid = [img_grid]   
    plt.figure(figsize=(30,6))
    img_grid = img_grid[0].detach()
    img_grid = F.to_pil_image(img_grid)
    plt.imshow(np.asarray(img_grid))
    plt.axis('off')
    labels_titles = ""
    for it in sample_labels.data:
        labels_titles = labels_titles + ", "+str(it.item())
    title = title+str(labels_titles)
    plt.title( title)

    
def plot_sample_data(loader_batch,title,input_type='video',  echonet = False):
    sample_images, sample_labels = loader_batch[0],loader_batch[1] #choose inputs and their labels
    print("Sample labels shape",sample_labels)
    if echonet:
        sample_images = sample_images.permute((0,2,1,3,4))
    if input_type == 'frames':
        #now sample images looks like B,C,H,W , make_grid takes B,C,H,W
        sample_images = sample_images[:5,:,:,:] #display a grid of 5 pics
    else:        
        #now sample images looks like B,C,Temp,H,W , make_grid takes B,C,H,W
        sample_images = sample_images[:5,0,:,:,:] #display a grid of 5 pics
    sample_labels = sample_labels[:5]
    grid = torchvision.utils.make_grid(sample_images)
    show_grid(grid,sample_labels,title)

    
def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict  
def forward_train (loader, modelName, model, device ,loss_criterion, optimizer,echonet = False):
    model.train()

    running_loss = 0.0
    running_corrects = 0.0
    for inputs, labels,meta in tqdm(loader):
        if 'LRCN' in modelName or 'ResNet_att' in modelName:
            model.Lstm.reset_hidden_state()
        if modelName == 'r2plus1d_18' or  modelName == 'R3D':
            inputs = inputs.permute((0,2,1,3,4))
            
            
        inputs = Variable(inputs.type(torch.float32), requires_grad=True).to(device)
        meta = Variable(meta.type(torch.float32), requires_grad=True).to(device)
        labels = Variable(labels).type(torch.LongTensor).to(device)
        
        model = model.to(device)
        optimizer.zero_grad()
        outputs = model([inputs,meta])
        #outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        loss = loss_criterion(outputs, labels)     

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss/len(loader.dataset), running_corrects/len(loader.dataset)


def forward_test (loader, modelName, model, device ,loss_criterion,show_saliency = False,echonet = False):
    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    ytruth = []
    running_loss = 0.0
    running_corrects = 0.0
    tot_mini_vids = 0
    for inputs,labels,meta in tqdm(loader):
        model.eval()

        if 'LRCN' in modelName:
            model.Lstm.reset_hidden_state()
        if modelName == 'r2plus1d_18' or modelName == 'R3D':
            inputs = inputs.permute((0,2,1,3,4))
                
        if echonet:
             #inputs = inputs.permute((0,2,1,3,4))
             labels =  Variable(labels).to(device)
        else:
            labels = labels.type(torch.LongTensor).to(device)
        meta = meta.type(torch.float32).to(device)
        

        inputs = inputs.type(torch.float32).to(device)
    
        with torch.no_grad():
            outputs = model([inputs,meta])
            #outputs = model(inputs)
            
            
        s1 += outputs.sum()
        s2 += (outputs ** 2).sum()
        
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        print("preds",preds)
        print("LABELS",labels)
        loss = loss_criterion(outputs, labels)     
        
        yhat = yhat + preds.cpu().numpy().tolist()
        ytruth = ytruth + labels.cpu().numpy().tolist()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        tot_mini_vids += inputs.size(0)
        
        total += loss.item() * inputs.size(0)

        n += inputs.size(0)
        
    if show_saliency:
        show_saliency_maps(inputs,labels,model,device)
    plot_out_vs_labels(yhat,ytruth)    
    
    #return total / n, s2 / n - (s1 / n) ** 2
    return running_loss/len(loader.dataset), running_corrects/len(loader.dataset)
def plot_out_vs_labels(y_pred,y_true):

    #plt.figure()
    #plt.scatter(labels,outputs)
    
    #r2_score = sklearn.metrics.r2_score(labels, outputs)
    #print("R2 Score ",r2_score )
    # constant for classes
    ##PLOtting confusion matrix
    plt.figure()
    reordered_classes = ('Low Risk','Average Risk','High Risk') # from average,low,high
    cf_matrix = confusion_matrix(y_true, y_pred)

    cm2 = np.roll(cf_matrix, 1, axis=0)
    cm2 = np.roll(cm2, 1, axis=1)

    cm_display = ConfusionMatrixDisplay(cm2, display_labels=reordered_classes).plot()#, display_labels=classes
    plt.show()

    #df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                         #columns = [i for i in classes])
    #plt.figure(figsize = (12,7))
    #plt.xlabel("Predicted Labels")
    #sn.heatmap(df_cm, annot=True)
    plt.savefig('conf_matrix.png')
    
    
    
def forward_video_inference (loader,model,device):
    running_corrects = 0.0
    for inputs, labels in tqdm(loader):
        #Original input shape: B,Chan,timesteps,h,w 
        #labels shape = B,
        batches,c,t,_,_ = inputs.shape
        for b in range(batches):
            inputs = inputs[b].permute((1,0,2,3)).to(device)
            label = labels[b].type(torch.LongTensor).to(device)
            with torch.no_grad():
                outputs = model(inputs)
    
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            print("All Preds: ",preds)
            mode,_ = torch.mode(preds)
            print("mode of preds: ",mode)
            print("True label: ",label)
            if mode == label:
                running_corrects = running_corrects+1
        print ("Total correct is ",running_corrects," out of ",len(loader.dataset))
        
def impl_weighted_sampler(train_dataset,test_dataset, batch, sample_test = False):
    #train_target =  torch.tensor([samp[1].item() for samp in train_dataset])
    #test_target =  torch.tensor([samp[1].item() for samp in test_dataset])

    train_target =  torch.tensor([samp[1] for samp in train_dataset])
    test_target =  torch.tensor([samp[1] for samp in test_dataset])
    
    print('Previous target train class distribution 0/1/2: {}/{}/{}'.format(
        (train_target == 0).sum(), (train_target == 1).sum(), (train_target == 2).sum()))
    print('Previous target test class distribution 0/1/2: {}/{}/{}'.format(
        (test_target == 0).sum(), (test_target == 1).sum(), (test_target == 2).sum()))
    
    # Computing Training dataset Sampler 
    train_class_sample_count = torch.tensor(
        [(train_target == t).sum() for t in torch.unique(train_target, sorted=True)])
    train_weight = 1. / train_class_sample_count.float()
    train_samples_weight = torch.tensor([train_weight[t] for t in train_target])

    # finally the train sampler and loader
    train_sampler = Sampler.WeightedRandomSampler(train_samples_weight,
                                                  len(train_samples_weight))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch,  sampler=train_sampler)
    my_labels = {'0':0,'1':0,'2':0}
    for samp in train_dataloader:
        my_labels['0'] = my_labels['0']+(samp[1] == 0).sum()
        my_labels['1'] = my_labels['1']+(samp[1] == 1).sum()
        my_labels['2'] = my_labels['2']+(samp[1] == 2).sum()
        
    print("After Sampling train class distributions 0/1/2: {}/ {}/ {}".format(my_labels['0'], my_labels['1'], my_labels['2']))

    if sample_test:
        # Computing test dataset Sampler 
        test_class_sample_count = torch.tensor(
            [(test_target == t).sum() for t in torch.unique(test_target, sorted=True)])
        test_weight = 1. / test_class_sample_count.float()
        test_samples_weight = torch.tensor([test_weight[t] for t in test_target])
    
        # finally the train sampler and loader
        test_sampler = Sampler.WeightedRandomSampler(test_samples_weight, 
                                                     len(test_samples_weight))

        test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch, sampler=test_sampler)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=batch)
        

    return train_dataloader, test_dataloader

    
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)