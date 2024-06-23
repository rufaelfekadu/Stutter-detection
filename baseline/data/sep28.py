import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd

class Sep28K(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.transform = transforms
        self.data = self._load_data(csv_file)

    def _load_data(self, csv_file):
        data_path = '/Users/apple/Documents/Projects/RA/baseline/dataset/clips/'
        df = pd.read_csv(csv_file)
        label_columns = df.columns[5:]  
        df['FilePath'] = data_path + df['Show'] + '/' + df['EpId'].astype(str) + '/' +  df['Show'] + '_' + df['EpId'].astype(str) + '_' + df['ClipId'] .astype(str) + '.wav'
        df[label_columns]
        df['Labels'] = df[label_columns].values.tolist()
        return df[['FilePath', 'Labels']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['FilePath']
        waveform, sample_rate = torchaudio.load(audio_path, format='wav')
        mel_spec = self.transform(waveform) if self.transform else waveform
        label_vector = row['Labels']
        return mel_spec, label_vector

if __name__ == "__main__":
    dataset = Sep28K(csv_file='/Users/apple/Documents/Projects/RA/baseline/dataset/SEP-28k_labels.csv', 
                     transforms=transforms.MelSpectrogram(win_length=400, hop_length=160))
    mel_spec, label_vector = dataset[0]
    print(mel_spec.shape, label_vector)