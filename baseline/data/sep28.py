import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split


class Sep28K(Dataset):
    __acceptable_params = ['root', 'label_path', 'ckpt', 'win_length', 'hop_length', 'n_mels']
    def __init__(self, transforms=None, save=True, split='train', **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]

        self.ckpt = f'{self.ckpt}/{split}.pt' if self.ckpt else f'{self.name}_{split}.pt'

        self.transform = transforms
        self.mel_func = MelSpectrogram(win_length=400, hop_length=160, n_mels=40)
        self.label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        
        if os.path.isfile(self.ckpt):
            print("************ Loading Cached Dataset ************")
            self.data, self.label_fluent, self.label_ccc, self.label_per_type = torch.load(self.ckpt)
            
        else:
            print("************ Loading Dataset ************")
            df = self._load_data()
            df = df[df['split'] == split].reset_index(drop=True)

            self.data, failed = self.load_audio_files(df)
            df = df.drop(failed).reset_index(drop=True)

            self.label_fluent = torch.tensor(df['label_fluent'].values, dtype=torch.long)
            self.label_per_type = torch.tensor(df['label_per_type'].values, dtype=torch.long)
            self.label_ccc = torch.tensor(df[self.label_columns].values, dtype=torch.float32)
            del df
            if save:
                torch.save((self.data, self.label_fluent, self.label_ccc, self.label_per_type), self.ckpt)
    
    def _load_audio_file(self, row):
        audio_path = row['file_path']
        try:
            waveform, sample_rate = torchaudio.load(audio_path, format='wav')
            mel_spec = self.mel_func(waveform)
            if mel_spec.shape[-1] < 301:
                print(f'Padding {audio_path} with {301 - mel_spec.shape[-1]}')
                mel_spec = torch.cat([mel_spec, torch.zeros(1,40, 301 - mel_spec.shape[-1])], dim=2)
            return (row.name, mel_spec)  # Return the index and the mel_spec
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")
            return (row.name, None)

    def load_audio_files(self, df):
        mel_specs = [None] * len(df)  # Preallocate list with None
        failed = []
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and get future objects
            futures = [executor.submit(self._load_audio_file, row) for _, row in df.iterrows()]
            for future in tqdm(as_completed(futures)):
                index, mel_spec = future.result()
                if mel_spec is not None:
                    mel_specs[index] = mel_spec.squeeze(0)  # Place mel_spec at its original index
                else:
                    failed.append(index)
        # Filter out None values in case of errors
        mel_specs = [spec for spec in mel_specs if spec is not None]
        return mel_specs, failed
    
    def _load_data(self):
        data_path = self.root
        df = pd.read_csv(self.label_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']), f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"), axis=1)
        df['label_fluent'] = df['NoStutteredWords'].map(lambda x: 1 if x >=2 else 0)
        df['label_per_type'] = df[self.label_columns].apply(lambda row: len(self.label_columns) if row.max() <= 1 else self.label_columns.index(row.idxmax()), axis=1)
       
        # Split the data into train, val, and test
        df['split'] = 'train'
        df.loc[train_test_split(df.index, test_size=0.1, random_state=42)[1], 'split'] = 'temp'
        df.loc[train_test_split(df[df['split'] == 'temp'].index, test_size=0.7, random_state=42)[1], 'split'] = 'test'
        df['split'] = df['split'].replace('temp', 'val')

        return df[['file_path', 'split', 'label_fluent', 'label_per_type']+self.label_columns]
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'mel_spec': self.data[idx],
            'label_fluent': self.label_fluent[idx],
            'label_ccc': self.label_ccc[idx],
            'label_per_type': self.label_per_type[idx]
        }
        # return self.data[idx], self.label_fluent[idx], self.label_ccc[idx], self.label_per_type[idx]

if __name__ == "__main__":
    label_path = 'datasets/sep28k/SEP-28k_labels_new.csv'
    data_path = 'datasets/sep28k/clips/'
    ck_path = 'datasets/sep28k/dataset.pt'
    train_transforms = MelSpectrogram(win_length=400, hop_length=160, n_mels=257)
    dataset = Sep28K(root=data_path, ckpt=ck_path, label_path=label_path, transforms=train_transforms)
    print(dataset[0])