import albumentations as A
from torchvision import transforms
import numpy as np
import cv2

class Hyperparams:
    
    #Training Params
    lr = 1e-3
    num_epochs = 2
    batch_size_train = 4
    batch_size_valid = 4
    weight_decay = 1e-5
    img_shape = (28,28,1)


    #Model params
    encoder_name = "efficientnet_b0"


    #Folds params
    num_splits = 5
    splits_to_train = [1,2,3,4,5]
    splits_to_oof = [1,2,3,4,5]
    
    random_state = 19

    normalise_transform = transforms.Compose([
        transforms.Normalize(mean=(0.01247053), std=(0.15167077))
        ])

    augment_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
    ], p=0.6)


