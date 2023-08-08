import torch
import numpy as np
from tqdm import tqdm
from jarviscloud import jarviscloud
import contextlib

from hyperparams import Hyperparams


class OOF_Evaluator:
    def __init__(self, splits_to_oof):
        self.splits_to_oof = splits_to_oof

    def metric_function(self, preds, labels):
        metric = preds.shape[0]
        return metric

    def calculate_oof_dice(self):
        all_preds = []
        all_labels = []

        for fold in self.splits_to_oof:
            preds = torch.load(f'running_preds/fold_{fold}/best_preds.pt',map_location = 'cpu')
            labels = torch.load(f'running_preds/fold_{fold}/best_labels.pt',map_location = 'cpu')

            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        global_metric = self.metric_function(all_preds, all_labels)
        print(f'global_metric = {global_metric}')


oof_evaluator = OOF_Evaluator(Hyperparams.splits_to_oof)
oof_evaluator.calculate_oof_dice()