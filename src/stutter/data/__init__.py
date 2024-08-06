from .sep28 import Sep28K
from .uclass import Uclass
from .fluencybank import FluencyBank, FluencyBankSlow
from torchaudio.transforms import MelSpectrogram
from sklearn.model_selection import train_test_split
import torch
import numpy as np

available_datasets = {
    'uclass': Uclass,
    'fluencybank': FluencyBank,
    'sep28k': Sep28K
}

def get_dataset(cfg):
    splits = ['train', 'val', 'test']
    ds = available_datasets[cfg.data.name](**cfg.data)
    datasets = {}
    for split in splits:
        idx = np.where(ds.split == splits.index(split))[0]
        print(f'{split} dataset size: {len(idx)}')
        datasets[split] = torch.utils.data.Subset(ds, idx)
    return datasets


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