import numpy as np
import os 
import torch
from torch import nn, optim
import torch.optim as optim
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import cv2
from accelerate import Accelerator, DistributedDataParallelKwargs,DistributedType
import accelerate.utils as accelerate_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt

from hyperparams import Hyperparams
from models import model_naming_function


class ModelTrainer:
    def __init__(self, model_func, trainloaders, valid_loaders, criterion):
        self.model_func = model_func
        self.trainloaders = trainloaders
        self.valid_loaders = valid_loaders
        self.criterion = criterion
        self.model_naming_function = model_naming_function

    def metric_function(self, y_pred, labels):
        return 1

    def seed_everything(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        accelerate_utils.set_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        
    def train(self, trainloader):
        self.model.train()
        self.accelerate.print('Training')
        train_running_loss = 0.0
        stream = tqdm(trainloader, total=len(trainloader), disable=not self.accelerate.is_local_main_process, )
        
        for i, data in enumerate(stream):
            inputs, labels, weights = data
            inputs, labels, weights = inputs.double(), labels.long(), weights.float()
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            loss = loss.sum(dim=0) * weights
            loss = loss.sum()
            train_running_loss += loss.item()
            self.accelerate.backward(loss)
            self.optimizer.step()
            outputs, labels = self.accelerate.gather_for_metrics((outputs, labels))
            
        epoch_loss = train_running_loss / (i+1)
        self.train_loss.append(epoch_loss)
        
        
    def validate(self, valid_loader):
        self.model.eval()
        self.accelerate.print('Validation')
        valid_running_loss = 0.0
        all_outputs = []
        all_labels = []
        stream = tqdm(valid_loader, total=len(valid_loader), disable=not self.accelerate.is_local_main_process, )
        
        with torch.no_grad():
            for i, data in enumerate(stream):
                inputs, labels ,weights = data
                inputs, labels, weights = inputs.double(), labels.long(), weights.float()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss.sum(dim=0) * weights
                loss = loss.sum()
                valid_running_loss += loss.item()


                outputs, labels = self.accelerate.gather_for_metrics((outputs, labels))

                all_outputs.append(outputs)
                all_labels.append(labels)
    
        epoch_loss = valid_running_loss / (i+1)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        global_metric = self.metric_function(all_outputs, all_labels)
    
        self.valid_loss.append(epoch_loss)
        self.valid_metric.append(global_metric)
        torch.cuda.empty_cache()

        self.last_metric = global_metric
        self.last_preds = all_outputs
        self.last_labels = all_labels

    def execute(self, num_epochs, splits_to_train):
        self.accelerate = Accelerator(
            mixed_precision="fp16",
            kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=False, find_unused_parameters=True)]
        )
        self.seed_everything(42)
        self.device = self.accelerate.device
        fold_count=0
        
        for trainloader, valid_loader in zip(self.trainloaders, self.valid_loaders):
            fold_count+=1
            if (fold_count in splits_to_train):
                self.accelerate.print(f"Fold {fold_count} of {splits_to_train}")
                self.model = self.model_func(self.accelerate)
                self.optimizer = optim.Adam(self.model.parameters(), lr=Hyperparams.lr, weight_decay=Hyperparams.weight_decay)
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=Hyperparams.num_epochs, eta_min=1e-6)
                self.model, self.optimizer, trainloader,valid_loader = self.accelerate.prepare(self.model, self.optimizer, trainloader,valid_loader)

                
                self.train_loss = []
                self.valid_loss = []
                self.train_acc = []
                self.valid_metric = []
    
                    
                for epoch in range(num_epochs):
                    self.accelerate.print(f"Epoch {epoch+1} of {num_epochs}")
                    self.train(trainloader)
                    self.validate(valid_loader)
                    self.accelerate.print(f"Training loss: {self.train_loss[-1]:.3f}")
                    self.accelerate.print(f"Validation loss: {self.valid_loss[-1]:.3f}, validation metric: {self.valid_metric[-1]:.3f}")
                    self.accelerate.print('-'*50)
                    unwrapped_model = self.accelerate.unwrap_model(self.model)
                    os.makedirs(f'running_models/fold_{fold_count}/', exist_ok=True)

                    model_path = f'running_models/fold_{fold_count}/'
                    model_name = self.model_naming_function(self.valid_metric[-1], epoch+1,Hyperparams)
                    
                    self.accelerate.save(unwrapped_model.state_dict(), model_path + model_name) 
                    self.scheduler.step()
    
                
                os.makedirs(f'running_preds/fold_{fold_count}/', exist_ok=True)
                torch.save(self.last_preds, f'running_preds/fold_{fold_count}/best_preds.pt')
                torch.save(self.last_labels, f'running_preds/fold_{fold_count}/best_labels.pt')

            else:
                self.accelerate.print(f"Fold {fold_count} ignored")
        print('TRAINING COMPLETE')