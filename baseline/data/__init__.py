from .sep28 import Sep28K
from .uclass import Uclass
from .fluencybank import FluencyBank
from torchaudio.transforms import MelSpectrogram
from sklearn.model_selection import train_test_split
import torch
import numpy as np

available_datasets = {
    'uclass': Uclass,
    'fluencybank': FluencyBank,
    'sep28k': Sep28K
}

def get_dataset(cfg, split='train'):
    return available_datasets[cfg.data.name](**cfg.data, split=split)

def get_dataloaders(cfg):
    
    # transforms = MelSpectrogram(win_length=400, hop_length=160, n_mels=40)
    train_dataset = get_dataset(cfg, split='train')
    val_dataset = get_dataset(cfg, split='val')
    test_dataset = get_dataset(cfg, split='test')

    #  print the y distribution for each split
    # print(f"Train: {np.unique(train_dataset.label, return_counts=True, axis=0)}")
    # print(f"Val: {np.unique(val_dataset.label, return_counts=True)}")
    # print(f"Test: {np.unique(test_dataset.label, return_counts=True)}")
    print('******train distribution****\n'+'\n'.join([f'{(np.unique((train_dataset.label>2).float()[:,i], return_counts=True)[1])}' for i in range(train_dataset.label.shape[1])]))
    print('******val distribution****\n'+'\n'.join([f'{(np.unique((val_dataset.label>2).float()[:,i], return_counts=True)[1])}' for i in range(val_dataset.label.shape[1])]))
    print('******test distribution****\n'+'\n'.join([f'{(np.unique((test_dataset.label>2).float()[:,i], return_counts=True)[1])}' for i in range(test_dataset.label.shape[1])]))
    #  get weights for loss function
    weights = [(np.unique((train_dataset.label>2).float()[:,i], return_counts=True)[1]/len(train_dataset))[0] for i in range(train_dataset.label.shape[1])]
    print('******weights for the BCE loss****\n',weights)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.solver.batch_size,
        shuffle=True,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.solver.batch_size,
        shuffle=False,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, weights
