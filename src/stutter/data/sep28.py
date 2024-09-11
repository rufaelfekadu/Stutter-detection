import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from stutter.utils.data import load_audio_files

class ScaleTransform:
    def __init__(self, scaler=MinMaxScaler()):
        self.scaler = scaler
    def __call__(self, x):
        # flatten the input
        n, c, l = x.shape
        x = x.reshape(n, -1)
        x = self.scaler.transform(x)
        return x.reshape(n, c, l)
    def fit(self, x):
        n, c, l = x.shape
        x = x.reshape(n, -1)
        self.scaler.fit(x)
        return self
    
class Sep28kNew(Dataset):

    __acceptable_params = ['root', 'label_path', 'ckpt', 'name', 'n_mels', 'win_length', 'hop_length', 'n_fft']

    def __init__(self, transforms=None, save=True, **kwargs):
        super(Sep28kNew, self).__init__()
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]

        self.length = 3
        self.transform = transforms
        self.scaler = ScaleTransform(MinMaxScaler())

        self.label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        self.ckpt = f'{self.ckpt}/{self.name}.pt' if self.ckpt else f'{self.name}.pt'

        if os.path.isfile(self.ckpt):
            print("************ Loading Cached Dataset ************")
            self.mel_spec, self.f0, self.label, self.split = torch.load(self.ckpt)
         
        else:
            print("************ Loading Dataset ************")
            self._load_data(**kwargs)
            if save:
                torch.save((self.mel_spec, self.f0, self.label, self.split), self.ckpt)
                # torch.save((self.data, self.label), self.ckpt)
    
    def _load_data(self, **kwargs):
        
        data_path = self.root
        df = pd.read_csv(self.label_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']).rjust(3,'0'), f"{row['Show']}_{str(row['EpId']).rjust(3,'0')}_{row['ClipId']}.wav"), axis=1)
        
        if not 'split' in df.columns:
            unique_clips = df['EpId'].unique()
            df['split'] = df['EpId'].apply(lambda x: 'train' if x in unique_clips[:int(0.8*len(unique_clips))] else 'val' if x in unique_clips[int(0.8*len(unique_clips)):int(0.9*len(unique_clips))] else 'test')
        
        self.mel_specs, self.f0, failed = load_audio_files(df, **kwargs)
        # remove failed files
        print(f"Failed to load {len(failed)} files")
        df = df.drop(failed).reset_index(drop=True)
        self.mel_specs = np.stack([x for x in self.mel_specs if x is not None])
        self.f0 = np.stack([x for x in self.f0 if x is not None])
               
        # scale the mel_specs
        train_idx = df[df['split'] == 'train'].index
        self.scaler.fit(self.mel_specs[train_idx])
        self.mel_spec = self.scaler(self.mel_specs)

        df['split'] = df['split'].apply(lambda x: 0 if x == 'train' else 1 if x == 'val' else 2)
        self.split = df['split'].values
        
        self.label = torch.tensor(df[self.label_columns].values, dtype=torch.float32)
        self.mel_spec = torch.tensor(self.mel_spec, dtype=torch.float32)
        self.f0 = torch.tensor(self.f0, dtype=torch.float32)

        del df
    
    def __len__(self):
        return len(self.mel_spec)

    def __getitem__(self, idx):
        return {
            'mel_spec': self.mel_spec[idx].squeeze(0),
            # 'f0': self.f0[idx],
            'label': self.label[idx]
        }


class Sep28K(Dataset):
    __acceptable_params = ['root', 'label_path', 'ckpt']
    def __init__(self, transforms=None, save=True, split='train', **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]

        self.ckpt = f'{self.ckpt}/{split}.pt' if self.ckpt else f'{self.name}_{split}.pt'
        self.transform = transforms
        self.label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        
        if os.path.isfile(self.ckpt):
            print("************ Loading Cached Dataset ************")
            self.data, self.label= torch.load(self.ckpt)
            
        else:
            print("************ Loading Dataset ************")
            self._load_data()
            if save:
                torch.save((self.data, self.label), self.ckpt)
    
    
    def _load_data(self, split='train'):
        data_path = self.root
        df = pd.read_csv(self.label_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']), f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"), axis=1)

        # Split the data into train, val, and test
        if not 'split' in df.columns:
            df['split'] = 'train'
            df.loc[train_test_split(df.index, test_size=0.11, random_state=42)[1], 'split'] = 'temp'
            df.loc[train_test_split(df[df['split'] == 'temp'].index, test_size=0.7, random_state=42)[1], 'split'] = 'val'
            df['split'] = df['split'].replace('temp', 'test')

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
            'label': self.label[idx],
        }
        # return self.data[idx], self.label_fluent[idx], self.label_ccc[idx], self.label_per_type[idx]

if __name__ == "__main__":
    label_path = 'datasets/sep28k/SEP-28k_labels_new.csv'
    data_path = 'datasets/sep28k/clips/'
    ck_path = 'datasets/sep28k/dataset.pt'
    dataset = Sep28K(root=data_path, ckpt=ck_path, label_path=label_path)
    print(dataset[0])