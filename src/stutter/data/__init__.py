from .sep28 import Sep28K
from .fluencybank import SEDataset, ClassificationDataset 
from .hf_data import HuggingFaceDataset
from torchaudio.transforms import MelSpectrogram
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

available_datasets = {
    'sed': SEDataset,
    'classification': ClassificationDataset,
    'sep28k': Sep28K,
    'hf': HuggingFaceDataset,
}

def collate_fn(batch):
    # Find the maximum length of audio features in the batch
    max_len = max(len(x['audio']) for x in batch)
    
    # Pad each audio feature array to the maximum length and convert to tensor
    for x in batch:
        x['audio_features'] = torch.tensor(
            np.pad(x['audio'], (0, max_len - len(x['audio'])), mode='constant', constant_values=0),
            dtype=torch.float32
        )
    
    # Stack the tensors for each key in the batch
    collated_batch = {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys() if k != 'file_path'}
    
    return collated_batch

def get_dataset(cfg):
    splits = ['train', 'val', 'test']
    ds = available_datasets[cfg.data.name](**cfg.data)
    datasets = {}
    for split in splits:
        idx = np.where(ds.split == splits.index(split))[0]
        print(f'{split} dataset size: {len(idx)}')
        datasets[split] = torch.utils.data.Subset(ds, idx)
    
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

    train_loader = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=cfg.solver.batch_size,
        shuffle=True,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets['val'],
        batch_size=cfg.solver.batch_size+1,
        shuffle=False,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets['test'],
        batch_size=cfg.solver.batch_size,
        shuffle=False,
        num_workers=cfg.solver.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    
    from stutter.utils.misc import setup_exp
    from stutter.config import cfg
    
    cfg.data.name = 'sed'
    cfg.data.root = 'datasets/fluencybank/ds_15/interview/clips/feature'
    cfg.data.split_file = 'datasets/fluencybank/our_annotations/interview_split.json'
    cfg.data.label_path = 'datasets/fluencybank/ds_15/interview/label/sed'
    cfg.data.cache_dir = 'datasets/fluencybank/ds_15/interview/fluencynak.pt'
    cfg.data.win_length = 400
    cfg.data.n_mfcc = 13
    cfg.data.n_mels = 40
    cfg.data.hop_length = 160
    cfg.data.n_frames = 15



    # setup_exp(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    for batch in train_loader:
        print(batch['audio'].shape, batch['label'].shape)
        break
    for batch in val_loader:
        print(batch['audio'].shape, batch['label'].shape)
        break
    for batch in test_loader:
        print(batch['audio'].shape, batch['label'].shape)
        break
    print('done')