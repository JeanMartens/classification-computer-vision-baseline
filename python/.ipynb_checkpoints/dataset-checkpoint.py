import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2

from hyperparams import Hyperparams
from fastnumpyio import fastnumpyio

class PreprocessedDataset(Dataset):
    def __init__(self, metadata_df,normalise_tranform = None,train=True):
        self.metadata_df = metadata_df
        self.train = train
        self.normalise_transform = normalise_tranform
        self.resize = A.Resize(Hyperparams.img_shape[0], Hyperparams.img_shape[1],interpolation=cv2.INTER_LANCZOS4, always_apply=True)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Load input / input must be (Shape1, Shape2, channels)
        # input_path =str(self.metadata_df.loc[idx, 'input_path'])
        # input = fastnumpyio.load(input_path)
        input = np.random.random((Hyperparams.img_shape[0],Hyperparams.img_shape[1],Hyperparams.img_shape[2]))

        #Load label
        label = np.array(self.metadata_df.loc[idx, 'label'])

        #Load input weight
        weight = torch.tensor(self.metadata_df.loc[idx, 'weight'])

        #Augments 
        if self.train == True:
            transformed = Hyperparams.augment_transform(image=input)
            input = transformed["image"]

        #Resize
        resized = self.resize(image=input)
        input = resized["image"]

            
        #To tensor and (Shape1, Shape2, channels) -> (channels, Shape1, Shape2)
        input = torch.tensor(input).permute(2,0,1)
        label = torch.tensor(label)

        #Apply final transform (Usually Normalisation)
        if self.normalise_transform:
            input = self.normalise_transform(input)

        return input, label,weight