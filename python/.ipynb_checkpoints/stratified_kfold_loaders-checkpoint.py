from sklearn.model_selection import KFold,StratifiedKFold
from dataset import PreprocessedDataset
from torch.utils.data import DataLoader


def kfold_loaders(metadata, normalise_transform = None, batch_size_train = 64, batch_size_valid = 64, num_splits=5, random_state=42):

    skfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    train_loaders = []
    valid_loaders = []

    for train_indices, valid_indices in kf.split(metadata):
        train_metadata = metadata.iloc[train_indices].reset_index(drop=True)
        valid_metadata = metadata.iloc[valid_indices].reset_index(drop=True)

        train_dataset = PreprocessedDataset(train_metadata, normalise_transform, train=True)
        valid_dataset = PreprocessedDataset(valid_metadata, normalise_transform, train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
    
    return train_loaders, valid_loaders

