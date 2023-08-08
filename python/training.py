import numpy as np
import pandas as pd 
import os
from jarviscloud import jarviscloud
from torch import nn
import torch

from train_valid import ModelTrainer
from fastnumpyio import fastnumpyio
from losses import *
from hyperparams import Hyperparams
from models import create_model

from stratified_kfold_loaders import *
from torch.cuda.amp import autocast

#Mock metadata
metadata = pd.DataFrame(np.random.randint(1, 5, size=(1000,3)), columns = ['input_path','label','weight'])

train_loaders, valid_loaders = kfold_loaders(
        metadata = metadata, 
        normalise_transform = Hyperparams.normalise_transform,
        batch_size_train = Hyperparams.batch_size_train, 
        batch_size_valid = Hyperparams.batch_size_valid,
        num_splits=Hyperparams.num_splits, 
        random_state=Hyperparams.random_state)

criterion = nn.CrossEntropyLoss()

training_instance = ModelTrainer(create_model, train_loaders,valid_loaders,criterion)


if __name__ == "__main__":
    training_instance.execute(Hyperparams.num_epochs, 
                              splits_to_train=Hyperparams.splits_to_train)
