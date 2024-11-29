# -*- coding: utf-8 -*-

import os
import numpy as np
import io
import cv2
import torch
from torch.utils.data import Dataset
from mypath_file import Path
import matplotlib.pyplot as plt

class FrameDataset(Dataset):

    def __init__(self, dataset, split='train'):
        _, output_dir = Path.db_dir(dataset)
        self.data_root = output_dir
        folder = os.path.join(self.data_root, split)
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for vidname in os.listdir(os.path.join(folder, label)):
                for fname in os.listdir(os.path.join(folder, label,vidname)):
                    self.fnames.append(os.path.join(folder, label,vidname,fname))
                    labels.append(label)
        
        assert len(labels) == len(self.fnames)
        print('Number of {} Frames: {:d}'.format(split, len(self.fnames)))
        # Prepare a mapping between  the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        self.resize_height = 256 #437
        self.resize_width = 256 #630
        self.crop_size = 255
        if not os.path.exists('cardiac_labels.txt'):
            with open('cardiac_labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id+1) + ' ' + label + '\n')
        
    def __len__(self):
    	return len(self.fnames)
    
    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)
    
          
    def load_frames(self, file_dir):
   
        frame = np.array(cv2.imread(file_dir))

        if (frame.shape[0] != self.resize_height) or (frame.shape[1] != self.resize_width):
            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        return frame.astype('float32')
    
    def normalize(self, frame):
        means = [0.135,0.120,0.123] #Sample calculated 
        stds = [0.185,0.171 ,0.174]

        for  chan in range(3):
            frame[chan] = frame[chan]- means[chan]/stds[chan]

        return frame
    
    def to_tensor(self, buffer):
        return buffer.transpose((2, 0, 1))


    def get_all_target(self):
        labels = np.array([],dtype=np.longlong)
        for lbl in self.label_array:
            labels = np.append(labels,(lbl))
        return torch.from_numpy(labels)
                    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data_pth = "..."
    train_data = FrameDataset( data_pth, split='train')
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)


