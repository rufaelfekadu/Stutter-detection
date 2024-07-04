from .sep28 import Sep28K
from torchaudio.transforms import MelSpectrogram
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def get_dataloaders(cfg):
    
    transforms = MelSpectrogram(win_length=400, hop_length=160, n_mels=40)
    dataset = Sep28K(cfg.data_path, cfg.label_path, ckpt_path=cfg.data_ckpt, transforms=transforms)


    train_idx, val_idx = train_test_split(np.arange(0, len(dataset)), test_size=0.1, random_state=cfg.seed) 
    val_idx, test_idx = train_test_split(val_idx, test_size=0.35, random_state=cfg.seed)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    #  print the y distribution for each split
    print(f"Train: {np.unique(train_dataset.dataset.label_per_type[train_idx], return_counts=True)}")
    print(f"Val: {np.unique(val_dataset.dataset.label_per_type[val_idx], return_counts=True)}")
    print(f"Test: {np.unique(test_dataset.dataset.label_per_type[test_idx], return_counts=True)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
