import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchaudio
import pandas as pd
import os
from stutter.utils.misc import load_audio_files


class FluencyBank(Dataset):
    __acceptable_params = ['root', 'label_path', 'ckpt']
    def __init__(self, transforms=None, save=True, split='train', **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]

        self.transform = transforms
        self.label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        self.ckpt = f'{self.ckpt}/{split}.pt' if self.ckpt else f'{self.name}_{split}.pt'

        if os.path.isfile(self.ckpt):
            print("************ Loading Cached Dataset ************")
            self.mel_spec, self.f0, self.label = torch.load(self.ckpt)
         
        else:
            print("************ Loading Dataset ************")
            self._load_data(split=split)
            if save:
                torch.save((self.mel_spec, self.f0, self.label), self.ckpt)
                # torch.save((self.data, self.label), self.ckpt)
    
    
    def _load_data(self, split='train'):
        data_path = self.root
        df = pd.read_csv(self.label_path)
        df['file_path'] = df.apply(lambda row: os.path.join(data_path, row['Show'], str(row['EpId']).rjust(3,'0'), f"{row['Show']}_{str(row['EpId']).rjust(3,'0')}_{row['ClipId']}.wav"), axis=1)
        
        # Split the data into train, val, and test
        if not 'split' in df.columns:
            unique_clips = df['EpId'].unique()
            print("unique files: ", len(unique_clips))
            # np.random.shuffle(unique_clips) 
            df['split'] = df['EpId'].apply(lambda x: 'train' if x in unique_clips[:int(0.8*len(unique_clips))] else 'val' if x in unique_clips[int(0.8*len(unique_clips)):int(0.9*len(unique_clips))] else 'test')
            print(df['split'].value_counts())
        
        df = df[df['split'] == split].reset_index(drop=True)

        # df = df[df[self.label_columns].apply(lambda x: x>=2).any(axis=1)].reset_index(drop=True)

        self.mel_spec, self.f0, failed = load_audio_files(df)


        print(f"Failed to load {len(failed)} files")
        df = df.drop(failed).reset_index(drop=True)

        # to tensor
        self.mel_spec = torch.tensor(self.mel_spec, dtype=torch.float32)
        self.f0 = torch.tensor(self.f0, dtype=torch.float32)
        self.label = torch.tensor(df[self.label_columns].values, dtype=torch.float32)

        del df
    
    
    def __len__(self):
        return len(self.mel_spec)

    def __getitem__(self, idx):
        return {
            'mel_spec': self.mel_spec[idx],
            'f0': self.f0[idx],
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