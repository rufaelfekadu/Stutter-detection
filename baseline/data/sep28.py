import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class Sep28K(Dataset):
    def __init__(self, root, label_path, ckpt_path='dataset.pt', transforms=None, save=True):
        
        self.transform = transforms
        self.root = root
        self.labels_path = label_path
        self.ckpt_path = ckpt_path
        self.label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        if os.path.isfile(ckpt_path):
            print("************ Loading Cached Dataset ************")
            self.data, self.label_fluent, self.label_ccc, self.label_per_type = torch.load(ckpt_path)
            
        else:
            print("************ Loading Dataset ************")
            df = self._load_data()
            self.data = self.load_audio_files(df)
            self.label_fluent = torch.tensor(df['label_fluent'].values, dtype=torch.long)
            self.label_per_type = torch.tensor(df['label_per_type'].values, dtype=torch.long)
            self.label_ccc = torch.tensor(df[self.label_columns].values, dtype=torch.float32)
            if save:
                torch.save((self.data, self.label_fluent, self.label_ccc, self.label_per_type), ckpt_path)
    
    def _load_audio_file(self, row):
        audio_path = row['file_path']
        try:
            waveform, sample_rate = torchaudio.load(audio_path, format='wav')
            mel_spec = self.transform(waveform) if self.transform else waveform
            if mel_spec.shape[-1] < 301:
                mel_spec = torch.cat([mel_spec, torch.zeros(1,40, 301 - mel_spec.shape[-1])], dim=2)

            return (row.name, mel_spec)  # Return the index and the mel_spec
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")
            return (row.name, None)

    def load_audio_files(self, df):
        mel_specs = [None] * len(df)  # Preallocate list with None
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and get future objects
            futures = [executor.submit(self._load_audio_file, row) for _, row in df.iterrows()]
            for future in tqdm(as_completed(futures)):
                index, mel_spec = future.result()
                if mel_spec is not None:
                    mel_specs[index] = mel_spec.squeeze(0)  # Place mel_spec at its original index

        # Filter out None values in case of errors
        mel_specs = [spec for spec in mel_specs if spec is not None]
        return mel_specs
    
    def _load_data(self):
        data_path = self.root
        df = pd.read_csv(self.labels_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']), f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"), axis=1)
        df['label_fluent'] = df['NoStutteredWords'].map(lambda x: 1 if x >=2 else 0)
        df['label_per_type'] = df[self.label_columns].apply(lambda row: len(self.label_columns) if row.max() <= 1 else self.label_columns.index(row.idxmax()), axis=1)
        #  remove samples with ambigiues labels
        df = df[~(df['label_per_type']==len(self.label_columns))]
        df = df.reset_index(drop=True)
        return df[['file_path', 'label_fluent', 'label_per_type']+self.label_columns]
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label_fluent[idx], self.label_ccc[idx], self.label_per_type[idx]

if __name__ == "__main__":
    label_path = 'datasets/sep28k/SEP-28k_labels_new.csv'
    data_path = 'datasets/sep28k/clips/'
    train_transforms = transforms.MelSpectrogram(win_length=400, hop_length=160, n_mels=40)
    dataset = Sep28K(root=data_path, label_path=label_path, transforms=train_transforms)
    print(dataset[0])