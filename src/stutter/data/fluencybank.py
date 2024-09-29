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

class SEDataset(Dataset):
    __acceptable_params = ['name', 'root', 'label_path', 'cache_dir', 'split_file', 'n_mels', 'win_length', 'hop_length', 'n_fft', 'sr', 'n_frames']
    
    def __init__(self, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        self.label_map = LabelMap()
        class_names = ['SR','ISR','MUR','P','B', 'V']
        self.class_names = [self.label_map.description[i] for i in class_names]
        try:
            self.label, self.data = torch.load(self.cache_dir)
            print("************ Loading Cached Data ************")
        except:
            print("************ Preparing Data ************")
            self.label = self.prep_label()
            self.data = self.prep_data()
            torch.save((self.label, self.data), self.cache_dir)

        assert len(self.data) == len(self.label), f"Data and label length do not match {len(self.data)} != {len(self.label)}"
        self.file_names = list(self.data.keys())
        # read the split file
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        self.split = np.zeros(len(self.data))
        for i, audio_file in enumerate(self.data.keys()):
            file_name = audio_file.split('_')[0]
            if file_name in split_data['train']:
                self.split[i] = 0
            elif file_name in split_data['val']:
                self.split[i] = 1
            else:
                self.split[i] = 2

    def prep_data(self):
        data_path = self.root
        audio_files = glob(f'{data_path}/*/*.npy')
        audio_files = [x.split('/')[-1].split('.')[0] for x in audio_files]
        data = {}
        for audio_file in audio_files:
            file_name = audio_file.split('_')[0]
            audio = np.load(f'{data_path}/{file_name}/{audio_file}.npy')
            mel_spec = torch.tensor(audio, dtype=torch.float32)
            data[audio_file] = mel_spec
        return data
        
    def prep_label(self):
        label_size = (self.n_frames*self.sr + self.win_length) // self.hop_length - 1
        # read all the text files in the label path
        label_files = glob(f'{self.label_path}/*/*.txt')
        labels = {}
        for label_file in label_files:
            l = torch.zeros(label_size, len(self.class_names))
            with open(label_file, 'r') as f:
                temp = f.readlines()
                temp = [x.strip().split(',') for x in temp]
                temp = [[float(x[0]), float(x[1]), x[2]] for x in temp]
                for t in temp:
                    start = int((t[0]*self.sr - self.win_length) // self.hop_length)
                    end = int((t[1]*self.sr - self.win_length) // self.hop_length)
                    try:
                        label = self.class_names.index(t[2])
                    except ValueError:
                        continue
                    l[start:end, label] = 1
            file_name = label_file.split('/')[-1].split('.')[0]
            labels[file_name] = l
        return labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        audio_features = self.data[file_name]
        label = self.label[file_name]
        return {
            'audio': audio_features,
            'label': label,
            'fname': file_name
        }

class ClassificationDataset(Dataset):
    __acceptable_params = ['name', 'root', 'label_path', 'cache_dir', 'split_file', 'encoder_name', 'n_mels', 'win_length', 'hop_length', 'n_fft', 'sr', 'n_frames']

    def __init__(self, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        self.label_map = LabelMap()
        class_names = ['SR', 'ISR', 'MUR', 'P', 'B', 'V']
        self.class_names = [self.label_map.description[i] for i in class_names]

        try:
            self.label, self.data = torch.load(self.cache_dir)
            print("************ Loading Cached Data ************")
        except:
            print("************ Preparing Data ************")
            self.label = self.prep_label()
            self.data = self.prep_data()
            torch.save((self.label, self.data), self.cache_dir)
    
        assert len(self.data) == len(self.label), f"Data and label length do not match {len(self.data)} != {len(self.label)}"
        self.file_names = list(self.data.keys())
        # read the split file
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        self.split = np.zeros(len(self.data))
        for i, audio_file in enumerate(self.file_names):
            file_name = audio_file.split('_')[0]
            if file_name in split_data['train']:
                self.split[i] = 0
            elif file_name in split_data['val']:
                self.split[i] = 1
            else:
                self.split[i] = 2

    def prep_data(self):
        data_path = self.root
        audio_files = glob(f'{data_path}/*/*.npy')
        # audio_files = [x.split('/')[-1].split('.')[0] for x in audio_files]
        data = {}
        for audio_file in audio_files:
            audio = np.load(audio_file)
            audio = torch.tensor(audio, dtype=torch.float32)
            file_name = audio_file.split('/')[-1].split('.')[0]
            data[file_name] = audio
        return data
    
    def prep_label(self):
        # read the label file
        label_files = glob(f'{self.label_path}/*/*.txt')
        labels = {}
        for label_file in label_files:
            with open(label_file, 'r') as f:
                l = torch.zeros(len(self.class_names))
                temp = f.readlines()
                temp = [x.strip().split(',') for x in temp]
                for t in temp:
                    try:
                        label = self.class_names.index(t[2])
                    except:
                        continue
                    l[label] = 1
                file_name = label_file.split('/')[-1].split('.')[0]
                labels[file_name] = l
        return labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        audio_features = self.data[file_name]
        label = self.label[file_name]
        return {
            'audio': audio_features,
            'label': label,
            'fname': file_name
        }

if __name__ == "__main__":
    kwargs = {
        'root': 'datasets/fluencybank/ds_15/interview/clips/feature',
        'label_path': 'datasets/fluencybank/ds_15/interview/label/sed',
        'split_file': 'datasets/fluencybank/our_annotations/interview_split.json',
        'annotator': 'A1',
        'cache_dir': 'outputs/fluencybank/fluencybank_sed.pt',
        'name': 'fluencybank',
        'n_mfcc': 13,
        'n_mels': 40,
        'win_length': 400,
        'hop_length': 160,
        'n_frames': 15,
        'sr': 16000
    }
    dataset = SEDataset(**kwargs)
    for i in range(10):
        print(dataset[i]['audio'].shape, dataset[i]['label'], dataset[i]['fname'])