import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Sep28K(Dataset):
    def __init__(self, root, label_path, transforms=None, split='train', random_state=42):
        self.transform = transforms
        self.root = root
        self.labels_csv = label_path
        self.data = self._load_data()

    def _load_data(self):
        data_path = self.root
        df = pd.read_csv(self.labels_csv)
        label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']), f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"), axis=1)
        df['label_fluent'] = df['NoStutteredWords'].map(lambda x: 0 if x >=2 else 1)
        df['label_per_type'] = df[label_columns].apply(lambda row: -1 if row.max() <= 1 else label_columns.index(row.idxmax()), axis=1)
        df['label_ccc'] = df[label_columns].values.tolist()
        return df[['file_path', 'label_fluent', 'label_per_type', 'label_ccc']]
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['file_path']
        waveform, sample_rate = torchaudio.load(audio_path, format='wav',)
        mel_spec = self.transform(waveform) if self.transform else waveform
        label_vector = row['label_ccc']
        label = row['label_fluent']
        if mel_spec.shape[-1] < 301:
            mel_spec = torch.cat([mel_spec, torch.zeros(1,40, 301 - mel_spec.shape[-1])], dim=2)

        return mel_spec, torch.tensor(label, dtype=torch.long), torch.tensor(label_vector, dtype=torch.float32), torch.tensor(row['label_per_type'], dtype=torch.long)

if __name__ == "__main__":
    label_path = '../datasets/sep28k/SEP-28k_labels.csv'
    data_path = '../datasets/sep28k/clips/'
    train_transforms = transforms.MelSpectrogram(win_length=400, hop_length=160, n_mels=257)
    dataset = Sep28K(root=data_path, labels_csv=label_path, transforms=train_transforms)
    mel_spec, label_vector = dataset[0]
    print(mel_spec.shape, label_vector)