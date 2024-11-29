import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import numpy as np 
import random
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import torch.utils.data.sampler as Sampler
from torchvision import transforms
from torchvision.models import resnet18
from dataloaders.dataset import VideoDataset
from dataloaders.vid_tab_dataset import VideoTabDataset
from dataloaders.echo_net_dataset import Echo
from dataloaders.single_frame_dataset import FrameDataset
from collections import OrderedDict

from network import R3D_model
from network import LRCN_model
from network import C3D_model
from network import LRCN_Tab
from network import tabular_network
from network import Res_Nl_Net

from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from helper_functions import forward_train,forward_test,plot_results,plot_sample_data, impl_weighted_sampler,init_normal

# =============================================================================
#                         DEFINING TRAINING PARAMETERS
# =============================================================================
# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
seed = 42
torch.manual_seed(seed)
exper_id = random.randint(0,10000)

useTest = True # See evolution of the test set when training
nTestInterval = 5 # Run on test set every nTestInterval epochs
nEpochs = 50
lr = 0.001 
bs = 16

datasetname = 'cardiac_mini_vids' # Options: cardiac_vids,cardiac_mini_vids,echonet,cardiac_mini_vids_Tab,cardiac_frames
modelName = 'LRCN' # Options: LRCN_Tab ,LRCN ,R3D  ,ResNet, r2plus1d_18,linear_tab,ResNet_att

exp_details = ""
save_weights_name = "att.pth" 

optimizername = 'adam'

cross_validation = False
load_model = False

lrcn_params = {
    "latent_dim" : 512, 
    "hidden_size" : 256,
    "lstm_layers" : 2,
    "bidirectional" : True}

if 'cardiac' in datasetname:
    num_classes = 3
elif datasetname == 'ucf101':
    num_classes = 101

# =============================================================================
#                         DEFINING SAVING PATHS
# =============================================================================

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
save_weights_dir = os.path.join(save_dir_root,
                                'model_weights',str(run_id)+ save_weights_name)

if load_model:
     read_weights_dir = os.path.join(save_dir_root,
                               'model_weights','tabular_weightsoct22.pth')

# =============================================================================
#                         DEFINING DATASET AND LOADERS
# =============================================================================

if datasetname == 'echonet':
    # Referring to EchonetDynamic Dataset and the original paper's dataset.
    # Returns training samples shaped Batch,Chan,Time(clip_len),H,W
    train_dataset, val_dataset = Echo(split="train"), Echo(split="val")
    test_dataset = Echo(split="test")
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=bs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=bs)
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    num_classes = 1

elif datasetname == 'cardiac_mini_vids_Tab':
    # Returns training samples shaped Batch,Time(clip_len),Chan,H,W
    train_dataset = VideoTabDataset(dataset='cardiac_mini_vids',split='train',clip_len=16)
    test_dataset = VideoTabDataset(dataset='cardiac_mini_vids',split='test',clip_len=16)
    train_dataloader , test_dataloader = impl_weighted_sampler(train_dataset,test_dataset, bs)
    #train_dataloader , test_dataloader = DataLoader(train_dataset, batch_size=bs),DataLoader(test_dataset, batch_size=bs)

elif datasetname == 'cardiac_mini_vids':
    # Returns training samples shaped Batch,Time(clip_len),Chan,H,W
    train_dataset = VideoDataset(dataset=datasetname, split='train',  clip_len=16)
    test_dataset = VideoDataset(dataset=datasetname, split='test',  clip_len=16)
    train_dataloader , test_dataloader = impl_weighted_sampler(train_dataset,test_dataset, bs)
    
elif datasetname == 'cardiac_vids':
    # Returns training samples shaped Batch,Time(clip_len),Chan,H,W
    train_dataset = VideoDataset(dataset=datasetname, split='train',  clip_len=36,preprocess=False)
    test_dataset = VideoDataset(dataset=datasetname, split='test',  clip_len=36,preprocess=False)
    
    train_dataloader , test_dataloader = impl_weighted_sampler(train_dataset,test_dataset, bs)
elif datasetname == 'cardiac_frames':
    # Returns training samples shaped Batch,Chan,H,W
    transform = transforms.Compose([
         transforms.Resize((255,255)),
         transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder( "...",transform)
    test_dataset = torchvision.datasets.ImageFolder( "...",transform)
    train_dataloader , test_dataloader = impl_weighted_sampler(train_dataset,test_dataset, bs)  
    
    
sample = next(iter(train_dataloader))
print("Data loader batches are with input shaped : ", sample[0].shape," and labels shape: ",sample[1].shape)
#plot_sample_data(sample,"Sample training Data",input_type='videos') # videos or frames

trainval_loaders = {'train': train_dataloader}#, 'val': val_dataloader
trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train']} #in ['train','val']

if cross_validation:
    k = 6
    print("Implementing %d fold cross validation "%(k))
    splits=KFold(n_splits=k,shuffle=True,random_state=seed)
    cross_dataset = ConcatDataset([train_dataset, test_dataset])
    
# =============================================================================
#                         DEFINING MODEL
# =============================================================================
saveName = modelName + '-' + datasetname

if modelName =='LRCN':
    model = LRCN_model.ConvLstm(lrcn_params['latent_dim'],lrcn_params['hidden_size'],
                    lrcn_params['lstm_layers'],lrcn_params['bidirectional'],num_classes,att_blocks = False)
    
elif modelName =='LRCN_Tab':
    model = LRCN_Tab.ConvLstmTab(lrcn_params['latent_dim'],lrcn_params['hidden_size'],
                    lrcn_params['lstm_layers'],lrcn_params['bidirectional'],n_class = 3,meta_categ = 8)
    
# =============================================================================
#     state_dict = torch.load(read_weights_dir)
#     new_state_dict = OrderedDict()
#     count1 = 0
#     for k,v in state_dict.items():
#         new_k = 'meta_model.'+k
#         if new_k in model.state_dict():
#             new_state_dict[new_k] = v
#             count1 = count1+1
#     print("Loaded %d Parameters"%(count1))    
#     model.load_state_dict(new_state_dict,strict=False)
# =============================================================================

elif modelName =='R3D':
    model = torchvision.models.video.r3d_18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
elif modelName =='r2plus1d_18':
    model =  torchvision.models.video.r2plus1d_18(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
elif modelName =='linear_tab':
    model =  tabular_network.tab_model(8, 3)

elif modelName =='C3D':
    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                    {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
elif modelName == 'ResNet':
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
elif modelName == 'ResNet_att':
    model = LRCN_model.ConvLstm(lrcn_params['latent_dim'],lrcn_params['hidden_size'],
                                lrcn_params['lstm_layers'],lrcn_params['bidirectional'],num_classes,att_blocks = True)
else:
    print("Model name written incorrectly")

model.to(device)
print('Experiment ID {} Training {} model on {} dataset...'.format(exper_id,modelName,datasetname))
# =============================================================================
#                         DEFINING LOSS and OPTIMIZER
# =============================================================================
criterion = nn.CrossEntropyLoss()  # standard crossentropy loss
#criterion = nn.MSELoss()
criterion.to(device)
if 'sgd' in optimizername:
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
elif 'adam' in optimizername:
    optimizer= torch.optim.Adam(model.parameters(),lr=lr)
if datasetname == 'echonet':
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9,weight_decay=5e-4)
    criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# =============================================================================
#                         TRAIN
# =============================================================================
def train_model(datasetname=datasetname, save_dir=save_dir, lr=lr,num_epochs=nEpochs,
   useTest=useTest, test_interval=nTestInterval, trainval_loaders= trainval_loaders,test_dataloader =test_dataloader,optimizer = optimizer,model = model):
    """Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    writer = SummaryWriter()
    phase ='train' # OR val
    all_losses = []
    all_accs = []

    for epoch in range(num_epochs):
        # each epoch has a training and validation step
        start_time = timeit.default_timer()
        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()
        epoch_loss,epoch_acc = forward_train(trainval_loaders[phase],
                        modelName,model,device,criterion,optimizer,echonet=False)
        if phase == 'train':
            writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            all_losses.append(epoch_loss)
            all_accs.append(epoch_acc.item())
        
        print("[{}] Epoch: {}/{} Loss: {} Acc: {} %".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc*100))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")
        
        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()
               
            epoch_loss,epoch_acc = forward_test(test_dataloader, modelName,
                                                model,device,criterion,echonet=False)
            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
           
            print("[test] Epoch: {}/{} Loss: {} Acc: {} %".format(epoch+1, nEpochs, epoch_loss, epoch_acc*100))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
    

    plot_results(num_epochs,all_losses,all_accs)
    torch.save(model.state_dict(), save_weights_dir)
    writer.add_text("Hyper Parameters", "Experiment id: "+str(exper_id)+ "ModelName: "+modelName + " Dataset: "+
       datasetname+" Optimizer: "+optimizername+" Learning_rate: "+str(lr)+" Batch_size: "+str(bs)+
       " seed: "+str(seed)+" Description: "+str(exp_details))   
    writer.close()
train_model()

