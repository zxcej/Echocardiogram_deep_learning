# -*- coding: utf-8 -*-


import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import pandas

from sklearn.preprocessing import MinMaxScaler
from mypath_file import Path################################Path file is the text file in the parent directory

class VideoTabDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos frames]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
    EXAMPLE:
        train
            AverageRisk
                2668S_minivid_0
                    00000.jpg
                    00001.jpg
                2668S_minivid_1
                2668S_minivid_2
                
            LowRisk
            HighRisk
        test
            AverageRisk
            LowRisk
            HighRisk

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset , split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.dataset = dataset
        tab_file_dir = os.path.join(self.output_dir, 'ex_vivo_analysis_values2.0.xlsx')

        # The following three parameters are chosen as described in the paper section 4.1
        if 'cardiac' in self.dataset :
            self.resize_height = 256#256 #437
            self.resize_width = 256#256 #630
            self.crop_size = 255
        else:
            self.resize_height = 128#224 #128  
            self.resize_width = 171#316 #171  
            self.crop_size = 112
        
        mask_path = os.getcwd() + "\\dataloaders\\unspeckled_triangle.png"
        self.mask = (np.array(Image.open(mask_path)))[:,:,1]

        
        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        vid_rng_nums = []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
                
                vid_name = os.path.split(fname)[-1]
                rng_numb = (vid_name.split("_"))[0]
                if 'S' in rng_numb:
                    rng_numb =  rng_numb.replace("S", "")
                vid_rng_nums.append(rng_numb)
                
                
                

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))
        
        

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif  'cardiac' in self.dataset:
            if not os.path.exists('cardiac_labels.txt'):
                with open('cardiac_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')
        #THE FIRST COLUMN OF THE TABULAR DATA IS ALWAYS THE RNG NUMBER 
        self.tab_data = self.load_tabular_data(tab_file_dir,vid_rng_nums)


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])
        tab_info = self.get_tabular_data(index)
        buffer = self.normalize(buffer)
        return buffer, torch.from_numpy(labels), torch.from_numpy(tab_info)
        #return torch.from_numpy(tab_info),torch.from_numpy(labels)

    def check_integrity(self):
        
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False
        return True
    def load_tabular_data(self, exc_dir, vids_rng):
        targets = ['RNG Number','age (yr)','bsa','Systolic Pressure (mmHg)','Diastolic pressure (mmHg)','History of Hypertension','Diabetes (I/II)','Presents with NYHA Heart Failure Symptoms','Ascending diameter (measured from surgical TEE) (mm)']
        #targets = ['RNG Number','Female','Bicuspid Aortic Valve',
        #           'History of Hypertension','age (yr)','bsa','Systolic Pressure (mmHg)',
        #          'Ascending diameter (measured from surgical TEE) (mm)']
        data = pandas.read_excel(exc_dir).loc[:,targets].dropna().to_numpy() #now we have read all the tabular data file
        data[:,0] = (data[:,0].astype('str'))
        #let's filter just the ones we have videos for
# =============================================================================
#         print(vids_rng)
#         sub_data = data[data['RNG Number'].isin(vids_rng)]
#         print("sub data shape ",sub_data.head(1))
# 
#         sub_data = sub_data.reindex(index = vids_rng).drop('RNG Number', axis=1).to_numpy()
#         
#         #scaling age info
# =============================================================================
        scaler = MinMaxScaler()
        data[:,4:] = scaler.fit_transform(data[:,4:] )
        return data
    
    def get_tabular_data(self, indx):
        vid_name = os.path.split(self.fnames[indx])[-1]
        rng_numb = (vid_name.split("_"))[0]

        if 'S' in rng_numb:
            rng_numb =  rng_numb.replace("S", "")
        cur_sample_row = (np.where(self.tab_data[:,0]==rng_numb))[0].item()
        info = (self.tab_data[cur_sample_row,1:]).astype(np.float) ############## WE are simply storing the rng number but arent passing it to model
        #print("Range Number ",rng_numb," label ",self.label_array[indx],"info ",info)

        return info

        
    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            #os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            #train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, test = train_test_split(video_files, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            #val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            #if not os.path.exists(val_dir):
                #os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            #for video in val:
                #self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]


        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))
        if self.dataset == 'cardiac_mini_vids' :
            self.process_mini_video(video_filename, save_dir, capture)
        else:
            if not os.path.exists(os.path.join(save_dir, video_filename)):
                os.mkdir(os.path.join(save_dir, video_filename))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
            # Make sure splited video has at least 16 frames
            EXTRACT_FREQUENCY = 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1
                    if frame_count // EXTRACT_FREQUENCY <= 16:
                        EXTRACT_FREQUENCY -= 1
    
            count = 0
            i = 0
            retaining = True
            while (count < frame_count and retaining):
                retaining, frame = capture.read()
                if frame is None:
                    continue
                if count % EXTRACT_FREQUENCY == 0:
                    frame = self.apply_mask(frame,self.mask)             
                    cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                    i += 1
                count += 1
    
            # Release the VideoCapture once it is no longer needed
            capture.release()
        
    def apply_mask(self, frame, mask):
        """process frame and apply black pixel mask to erase medical annotations on the image sides
        Args:
            frame: image frame to be processed.
            mask: binary mask image to overlay to erase annotations 
        """
        ARCH = 485
        neg_ys ,neg_xes  = np.where(mask == 0 )
        pos_ys ,pos_xs  = np.where(mask != 0 )
        
        fil_y =neg_ys[ np.where(neg_ys<ARCH)]
        fil_x=neg_xes[ np.where(neg_ys<ARCH)]
        mask_neg_indices = (fil_y,fil_x)
        frame[mask_neg_indices] = 0
        
        ys ,ye = min(pos_ys), max(pos_ys)
        xs ,xe = min(pos_xs), max(pos_xs)
        #dimensions of frame become: 447,630,3
        frame = frame[ys-5:ye+30,xs+100:xe-100]        
        return frame
        
    def process_mini_video(self, video_filename, save_dir, capture):
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Make sure splited video has at least 16 frames
        MINI_VIDEO_LENGTH = 16
        
        if frame_count % MINI_VIDEO_LENGTH != 0:
            frame_count = frame_count - (frame_count % MINI_VIDEO_LENGTH)
            
        num_mini_vid = int(frame_count / MINI_VIDEO_LENGTH)
        print("Number of mini videos in ",video_filename, " is ",num_mini_vid)
        total_count = 0
        i =0
        retaining = True
        
        for vid_ind in range(num_mini_vid):
            if not os.path.exists(os.path.join(save_dir, video_filename+"_minivid_"+str(vid_ind))):
                os.mkdir(os.path.join(save_dir, video_filename+"_minivid_"+str(vid_ind)))
            
            
        while (total_count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue
            mini_vid_indx = total_count // MINI_VIDEO_LENGTH

            frame = self.apply_mask(frame,self.mask)             
            cv2.imwrite(filename=os.path.join(save_dir, video_filename+"_minivid_"+str(mini_vid_indx), '0000{}.jpg'.format(str(i))), img=frame)
            i += 1
            
            total_count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()
    
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        if 'cardiac' in self.dataset:
            means = [0.135,0.120,0.123] #Sample calculated 
            stds = [0.185,0.171 ,0.174]
        else:
            means = [0.485, 0.456, 0.406] #Image-Net values
            stds = [0.229, 0.224, 0.225]
        means = [0.135,0.120,0.123] #Sample calculated 
        stds = [0.185,0.171 ,0.174]
        transform_norm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(means, stds)
        ])
        #print("BUFFER SHAPE",buffer.shape)
        for i, frame in enumerate(buffer):
            frame = frame.transpose((1,2,0)) /255 #now shape 256,256,3 and manually div by 255 because toTensor doesnt scale float vals from 0-255 to 0-1
            img_normalized = transform_norm(frame)
            buffer[i] = img_normalized

            

        return buffer

    ##HELPER Function to understand pixel distribution; not needed for the class to function
    def plot_pix_distrib(self, img):
        img_np = np.array(img)
        plt.figure()
        plt.title("original Img")
        plt.imshow(img_np)
        print("org image shape",img_np.shape)
        ##################################PLOT BASIC IMAGE PIX DIST
        plt.figure()
        plt.hist(img_np.ravel(), bins=50, density=True)
        plt.xlabel("pixel values")
        plt.ylabel("relative frequency")
        plt.title("distribution of original img pixels")
        ##################################PLOT TENSOR IMAGE PIX DIST
        plt.figure()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        img_tr = transform(img)
        img_np = np.array(img_tr)
        # plot the pixel values
        plt.hist(img_np.ravel(), bins=50, density=True)
        plt.xlabel("pixel values")
        plt.ylabel("relative frequency")
        plt.title("distribution of tensor pixels")
        plt.figure()
        plt.title("tensor Img")
        print("tensor image shape",img_tr.shape)
        plt.imshow(np.array(img_tr.permute(1,2,0)))
        
        mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
        print(mean)
        print(std)
        ##################################PLOT Normalized IMAGE PIX DIST
        plt.figure()

        means = [0.135,0.120,0.123] #Sample calculated 
        stds = [0.185,0.171 ,0.174]
        
        transform_norm = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(means, stds)
        ])
 
 
        # get normalized image
        img_normalized = transform_norm(img_tr)
        print("norm image shape",img_normalized.shape)
        img_np = np.array(img_normalized)
        mean, std = img_normalized.mean([1,2]), img_normalized.std([1,2])

        print("Mean and Std of normalized image:")
        print("Mean of the image:", mean)
        print("Std of the image:", std)
        # plot the pixel values
        plt.hist(img_np.ravel(), bins=50, density=True)
        plt.xlabel("pixel values")
        plt.ylabel("relative frequency")
        plt.title("distribution of normalized pixels")
        plt.figure()
        plt.title("tensor normalized Img")
        plt.imshow(np.array(img_normalized.permute(1,2,0)))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, 3, self.resize_height, self.resize_width), np.dtype('float32')) #, np.dtype('float32')
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)) #.astype(np.float64)
            
            if (frame.shape[0] != self.resize_height) or (frame.shape[1] != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[i] = frame.transpose((2,0,1)) #to be shape 3,H,W
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        if 'cardiac' in self.dataset:
            #buffer = buffer[time_index:time_index + clip_len, :, :, :]
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
            buffer = buffer[time_index:time_index + clip_len,
                     height_index:height_index + crop_size,
                     width_index:width_index + crop_size, :]
        else:
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
            buffer = buffer[time_index:time_index + clip_len,
                     height_index:height_index + crop_size,
                     width_index:width_index + crop_size, :]
        

        return buffer

        
        
    def get_all_target(self):
        labels = np.array([],dtype=np.longlong)
        for lbl in self.label_array:
            labels = np.append(labels,(lbl))
        return torch.from_numpy(labels)
    
    
  