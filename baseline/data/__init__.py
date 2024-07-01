from .sep28 import Sep28K
from torchaudio.transforms import MelSpectrogram
from sklearn.model_selection import train_test_split
import torch

def get_dataloaders(cfg):
    transform = MelSpectrogram(win_length=400, hop_length=160, n_mels=40)
    dataset = Sep28K(cfg.data_path, cfg.label_path, transform=transform)

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=cfg.seed)
    val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=cfg.seed)

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
