import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchaudio
import pandas as pd
import numpy as np
import os
import librosa
from stutter.utils.data import load_audio_files
from stutter.utils.data import logmelfilterbank
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

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

class FluencyBank(Dataset):

    __acceptable_params = ['root', 'label_path', 'ckpt', 'name', 'n_mels', 'win_length', 'hop_length', 'n_fft']

    def __init__(self, transforms=None, save=True, **kwargs):
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

class FluencyBankSlow(Dataset):
    __acceptable_params = ['root', 'label_path', 'ckpt', 'name', 'n_mels', 'win_length', 'hop_length', 'n_fft', 'n_frames', 'sr']
    def __init__(self, transforms=None, split='train', **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]

        self.length = (self.n_frames*self.sr + self.win_length) // self.hop_length - 1

        # self.transform = torchaudio.transforms.MelSpectrogram(win_length=self.win_length, hop_length=self.hop_length, n_mels=self.n_mels, sample_rate=self.sr)
        self.label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']

        self.label, self.data_paths = self.load_data()
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def load_data(self):

        data_path = self.root
        df = pd.read_csv(self.label_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']).rjust(3,'0'), f"{row['Show']}_{str(row['EpId']).rjust(3,'0')}_{row['ClipId']}.wav"), axis=1)
        
        if not 'split' in df.columns:
            unique_clips = df['EpId'].unique()
            print("unique files: ", len(unique_clips))
            # np.random.shuffle(unique_clips) 
            df['split'] = df['EpId'].apply(lambda x: 'train' if x in unique_clips[:int(0.8*len(unique_clips))] else 'val' if x in unique_clips[int(0.8*len(unique_clips)):int(0.9*len(unique_clips))] else 'test')
            print(df['split'].value_counts())
        
        self.split = df['split'].values

        return df[self.label_columns].values, df['file_path'].values
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # load the audio file
        # waveform, sample_rate = torchaudio.load(self.data_paths[idx], format='wav')
        waveform, sample_rate = librosa.load(self.data_paths[idx], sr=self.sr)
        # mel_spec = logmelfilterbank(waveform, sample_rate, fft_size=self.n_mels, hop_size=self.hop_length, win_length=self.win_length)
        y = np.pad(y, (0, 48000 - len(y)), constant_values=0)
        return {
            'audio': waveform.squeeze(0),
            'label': self.label[idx]
        }
    
# get all none values from list

if __name__ == "__main__":
    kwargs = {
        'root': 'datasets/fluencybank/new_clips/',
        'label_path': 'outputs/fluencybank/fluencybank_labels_new_split.csv',
        'ckpt': 'outputs/fluencybank/',
        'name': 'fluencybank',
        'n_mels': 40,
        'win_length': 400,
        'hop_length': 160,
    }
    dataset = FluencyBank(**kwargs)
    print(dataset[0])