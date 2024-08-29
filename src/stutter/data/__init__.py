from .sep28 import Sep28K
from .uclass import Uclass
from .fluencybank import FluencyBank, FluencyBankSlow, FluencyBankYOHO  
from torchaudio.transforms import MelSpectrogram
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd

available_datasets = {
    'uclass': Uclass,
    'fluencybank': FluencyBank,
    'sep28k': Sep28K,
    'fluencybankyoho': FluencyBankYOHO
}

def get_dataset(cfg):
    splits = ['train', 'val', 'test']
    ds = available_datasets[cfg.data.name](**cfg.data)
    datasets = {}
    for split in splits:
        idx = np.where(ds.split == splits.index(split))[0]
        print(f'{split} dataset size: {len(idx)}')
        datasets[split] = torch.utils.data.Subset(ds, idx)
    if not 'yoho'in cfg.data.name:
        print_dataset_stats(datasets)
    return datasets

def print_dataset_stats(dataset):
    to_print = {}
    print('Dataset stats:')
    for i, data in dataset.items():
        labels = (data.dataset.label[data.indices]>2).float()
        label_counts = torch.sum(labels, axis=0)
        to_print[i] = dict(zip(data.dataset.label_columns, label_counts.numpy().tolist()))
    print(pd.DataFrame(to_print).T)


def get_dataloaders(cfg):
    
    datasets = get_dataset(cfg)
    #  get weights for loss function
    # weights = [(np.unique((train_dataset.label>=2).float()[:,i], return_counts=True)[1]/len(train_dataset))[0] for i in range(train_dataset.label.shape[1])]
    # print('******weights for the BCE loss****\n',weights)

    train_loader = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=cfg.solver.batch_size,
        shuffle=True,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets['val'],
        batch_size=cfg.solver.batch_size,
        shuffle=False,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets['test'],
        batch_size=len(datasets['test']),
        shuffle=False,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    
    from stutter.utils.misc import setup_exp
    from stutter.config import cfg
    
    cfg.data.name = 'fluencybankyoho'
    cfg.data.root = 'datasets/fluencybank/ds_sentence/reading/A3'
    cfg.data.label_path = 'datasets/fluencybank/new_annotations/reading_split.json'
    cfg.output_dir = 'outputs/fluencybank'

    setup_exp(cfg)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    for batch in train_loader:
        print(batch['mel_spec'].shape, batch['label'].shape)
        break
    for batch in val_loader:
        print(batch['mel_spec'].shape, batch['label'].shape)
        break
    for batch in test_loader:
        print(batch['mel_spec'].shape, batch['label'].shape)
        break
    print('done')