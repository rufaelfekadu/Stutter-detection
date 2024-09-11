import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchaudio
import pandas as pd
import numpy as np
import os
import librosa
from stutter.utils.data import load_audio_files, logmelfilterbank, aggregate_labels
from stutter.utils.annotation import LabelMap
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoFeatureExtractor, AutoProcessor
from glob import glob
import soundfile as sf
import json

encoders = {
    "whisper-l": "openai/whisper-large-v3",
    "whisper-b": "openai/whisper-base",
    "wav2vec2-b": "facebook/wav2vec2-base-960h"
}
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
    
def shrink_tensor(tensor, target_length):
    tensor = tensor.permute(0, 2, 1)
    shrunk_tensor = F.interpolate(tensor, size=target_length, mode='linear', align_corners=False)
    shrunk_tensor = shrunk_tensor.permute(0,2,1)  
    return shrunk_tensor

class FluencyBankYOHO(Dataset):
    __acceptable_params = ['root', 'label_path', 'ckpt', 'name', 'n_mels', 'win_length', 'hop_length', 'n_fft']

    def __init__(self, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        self.audio_splits = glob(f'{self.root}/clips/audio/*/*.wav')
        self.text_splits = glob(f'{self.root}/label/*/sed/*.txt')
        # assert len(self.audio_splits) == len(self.text_splits), f"Number of audio {len(self.audio_splits)} and text {len(self.text_splits)} splits do not match"
        self.transform = AutoFeatureExtractor.from_pretrained("openai/whisper-base", cache_dir="./outputs/")
        self.label_map = LabelMap()
        with open(self.label_path, 'r') as f:
            split_data = json.load(f)

        self.split = np.zeros(len(self.audio_splits))
        for i, audio_path in enumerate(self.audio_splits):
            audio_file = audio_path.split('/')[-2]
            if audio_file in split_data['train']:
                self.split[i] = 0
            elif audio_file in split_data['val']:
                self.split[i] = 1
            else:
                self.split[i] = 2

    def __len__(self):
        return len(self.audio_splits)

    def __getitem__(self, idx):
        audio_path = self.audio_splits[idx]
        audio_file = audio_path.split('/')[-2]
        sample_id = audio_path.split('/')[-1].split('.')[0]
        txt_path = f"{self.root}/label/{audio_file}/sed/{sample_id}.txt"
        wav,sr = sf.read(audio_path)
        audio_features = self.transform(wav,sampling_rate=sr)
        labels = np.loadtxt(txt_path, delimiter=' ')
        if len(labels) == 0:
            labels = np.zeros((22, 15))
        else:
            labels = labels.reshape(-1, 15)
            labels = np.pad(labels, ((0, 22 - len(labels)), (0, 0)), 'constant', constant_values=0)
        labels = torch.tensor(labels, dtype=torch.float32)
        audio_features = torch.tensor(audio_features['input_features'], dtype=torch.float32)

        return {
            "mel_spec": audio_features,
            "file_path": txt_path,
            "label": labels
        }

class FluencyBankSed(Dataset):

    __acceptable_params = ['root', 'label_path', 'cache_dir', 'name', 'split_file', 'encoder_name']

    def __init__(self, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        if self.encoder_name not in encoders:
            raise ValueError(f"Invalid encoder name {self.encoder_name}")
        
        self.data = torch.load(self.label_path)
        self.label = self.data['labels']
        self.label_paths = self.data['label_paths']
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.label = shrink_tensor(self.label, 1249)
        self.label_map = LabelMap()
        self.class_imbalance = torch.sum(self.label, dim=(0,1))/ (self.label.shape[0]*self.label.shape[1])
        print(f"Class Imbalance: {self.class_imbalance}")
        print(f"Feature extractor: {self.encoder_name}")
        self.transform = AutoFeatureExtractor.from_pretrained(encoders[self.encoder_name], cache_dir=self.cache_dir, return_tensors="pt")

        self.soft_labels = torch.zeros_like(self.label, dtype=torch.float32)
        self.soft_labels = (torch.sum(self.label, dim=1, keepdim=True) > 0).float().squeeze(1)

        self.label_map = LabelMap()
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)

        self.split = np.zeros(len(self.label_paths))
        for i, audio_path in enumerate(self.label_paths):
            audio_file = audio_path.split('/')[-3]
            if audio_file in split_data['train']:
                self.split[i] = 0
            elif audio_file in split_data['val']:
                self.split[i] = 1
            else:
                self.split[i] = 2

    def __getitem__(self, idx):
        file_name = self.label_paths[idx].split('/')[-1]
        media_file = file_name.split('_')[0]
        clip_id = file_name.split('_')[1]
        audio_file_path = f"{self.root}/{media_file}/{media_file}_{clip_id}.wav"
        wav,sr = sf.read(audio_file_path)
        audio_features = self.transform(wav,sampling_rate=sr)
        
        return {
            'mel_spec': audio_features['input_values'][0],
            'label': self.label[idx],
            'file_path': self.label_paths[idx]
        }
    
    def __len__(self):
        return len(self.label_paths)
    
class FluencyBank(Dataset):

    __acceptable_params = ['root', 'label_path', 'annotator', 'split_file', 'ckpt', 'cache_dir', 'name', 'n_mels', 'win_length', 'hop_length', 'n_fft', 'sr']

    def __init__(self, transforms=None, save=True, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]

        self.length = 3
        self.transform = transforms
        self.scaler = ScaleTransform(MinMaxScaler())
        self.label_map = LabelMap()

        self.cache_path = f'{self.cache_dir}/{self.name}_{self.annotator}.pt' if self.ckpt else f'{self.name}_{self.annotator}.pt'

        if os.path.isfile(self.cache_path):
            print("************ Loading Cached Dataset ************")
            self.mel_spec, self.f0, self.label, self.split = torch.load(self.cache_path)
         
        else:
            print("************ Loading Dataset ************")
            self._load_data(**kwargs)
            if save:
                torch.save((self.mel_spec, self.f0, self.label, self.split), self.cache_path)
                # torch.save((self.data, self.label), self.ckpt)

    def _load_data(self, **kwargs):
        
        data_path = self.root
        df = pd.read_csv(self.label_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['media_file'], f"{row['file_name']}.wav"), axis=1)

        #  read the split file
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        df['split'] = df['media_file'].apply(lambda x: 0 if x in split_data['train'] else 1 if x in split_data['val'] else 2)
        self.split = df['split'].values
        
        kwargs['n_frames'] = int(((df['end'] - df['start']).max()/1000)*self.sr)
        mel_specs, self.f0, failed = load_audio_files(df, **kwargs)

        # remove failed files
        print(f"Failed to load {len(failed)} files")
        df = df.drop(failed).reset_index(drop=True)
        mel_specs = np.stack([x for x in mel_specs if x is not None])
        f0 = np.stack([x for x in self.f0 if x is not None])
               
        # scale the mel_specs
        # train_idx = df[df['split'] == 'train'].index
        # self.scaler.fit(self.mel_specs[train_idx])
        # self.mel_spec = self.scaler(self.mel_specs)

        # df['split'] = df['split'].apply(lambda x: 0 if x == 'train' else 1 if x == 'val' else 2)
        
        self.label = torch.tensor(df[self.label_map.core].values, dtype=torch.float32)
        self.mel_spec = torch.tensor(mel_specs, dtype=torch.float32)
        self.f0 = torch.tensor(f0, dtype=torch.float32)

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
    def __init__(self, transforms=None, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]

        self.length = (self.n_frames*self.sr + self.win_length) // self.hop_length - 1

        # self.transform = torchaudio.transforms.MelSpectrogram(win_length=self.win_length, hop_length=self.hop_length, n_mels=self.n_mels, sample_rate=self.sr)
        self.label_columns = ['SR','ISR','MUR','P','B', 'no_stutter']

        self.label, self.data_paths = self.load_data()
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def load_data(self):

        data_path = self.root
        df = pd.read_csv(self.label_path).reset_index(drop=True)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['media_file'] ,f"{row['media_file']}_{row['clip_id']}.wav"), axis=1)
        df.drop(['split'], axis=1, inplace=True)
        # one if all the labels are 0
        df['no_stutter'] = df.apply(lambda row: 1 if row[self.label_columns[:-1]].sum() == 0 else 0, axis=1)
        if not 'split' in df.columns:
            print("adding split column")
            df['split'] = 'train'
            # read split file
            with open('datasets/fluencybank/our_annotations/reading_split.json') as f:
                split_data = json.load(f)
            val_files = split_data['val']
            test_files = split_data['test']
            df.loc[df['media_file'].isin(val_files), 'split'] = 'val'
            df.loc[df['media_file'].isin(test_files), 'split'] = 'test'

        df['split'] = df['split'].apply(lambda x: 0 if x == 'train' else 1 if x == 'val' else 2)
        self.split = df['split'].values

        return df[self.label_columns].values, df['file_path'].values
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        if idx<0 or idx>=len(self.data_paths):
            raise IndexError(f"Index {idx} out of bounds")
        # load the audio file
        # waveform, sample_rate = torchaudio.load(self.data_paths[idx], format='wav')
        waveform, sample_rate = librosa.load(self.data_paths[idx], sr=self.sr)
        mel_spec = logmelfilterbank(waveform, sample_rate, num_mels=self.n_mels, hop_size=self.hop_length, win_length=self.win_length)
        try:
            mel_spec = np.pad(mel_spec, (self.length - len(mel_spec), 0), constant_values=0)
        except Exception as e:
            # clip the mel_spec to length
            mel_spec = mel_spec[:self.length,:]
            
        # to torch tensor
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        return {
            'mel_spec': mel_spec,
            'label': self.label[idx]
        }
    
# get all none values from list

if __name__ == "__main__":
    kwargs = {
        'root': 'datasets/fluencybank/ds_label/reading/A1/clips/audio',
        'label_path': 'datasets/fluencybank/ds_label/reading/A1/total_label.csv',
        'split_file': 'datasets/fluencybank/our_annotations/reading_split.json',
        'ckpt': 'outputs/fluencybank/',
        'cache_dir': 'outputs/fluencybank/fluencybank.pt',
        'name': 'fluencybank',
        'n_mfcc': 13,
        'n_mels': 40,
        'win_length': 400,
        'hop_length': 160,
        'n_frames': 5,
        'sr': 16000
    }
    # dataset = FluencyBankYOHO(**kwargs)
    # for i in range(10):
    #     print(dataset[i]['mel_spec']['input_features'].shape, dataset[i]['label'])

    dataset = FluencyBank(**kwargs)
    for i in range(10):
        print(dataset[i]['mel_spec'].shape, dataset[i]['label'])