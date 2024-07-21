import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('baseline/')
from utils import load_audio_files


class FluencyBank(Dataset):
    __acceptable_params = ['root', 'label_path', 'ckpt']
    def __init__(self, transforms=None, save=True, split='val', **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]

        self.transform = transforms
        self.label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        self.ckpt = f'{self.ckpt}/{split}.pt' if self.ckpt else f'{self.name}_{split}.pt'

        if os.path.isfile(self.ckpt):
            print("************ Loading Cached Dataset ************")
            self.data, self.label = torch.load(self.ckpt)
         
        else:
            print("************ Loading Dataset ************")
            self._load_data(split=split)
            if save:
                torch.save((self.data, self.label), self.ckpt)
    
    
    def _load_data(self, split='train'):
        data_path = self.root
        df = pd.read_csv(self.label_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']).rjust(3,'0'), f"{row['Show']}_{str(row['EpId']).rjust(3,'0')}_{row['ClipId']}.wav"), axis=1)
        
        # Split the data into train, val, and test
        if not 'split' in df.columns:
            unique_clips = df['EpId'].unique()
            print("unique files: ", len(unique_clips))
            # np.random.shuffle(unique_clips) 
            df['split'] = df['EpId'].apply(lambda x: 'train' if x in unique_clips[:25] else 'val' if x in unique_clips[25:29] else 'test')
            print(df['split'].value_counts())
        df = df[df['split'] == split].reset_index(drop=True)

        self.data, failed = load_audio_files(df)
        df = df.drop(failed).reset_index(drop=True)
        self.label = torch.tensor(df[self.label_columns].values, dtype=torch.float32)

        del df
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'mel_spec': self.data[idx],
            'label': self.label[idx]
        }
        # return self.data[idx], self.label_fluent[idx], self.label_ccc[idx], self.label_per_type[idx]

if __name__ == "__main__":
    label_path = 'datasets/fluencybank/fluencybank_labels.csv'
    data_path = 'datasets/fluencybank/clips/'
    ck_path = 'datasets/fluencybank/dataset.pt'
    train_transforms = torchaudio.transforms.MelSpectrogram(win_length=400, hop_length=160, n_mels=257)
    dataset = FluencyBank(root=data_path, ckpt=ck_path, label_path=label_path, transforms=train_transforms)
    print(dataset[0])