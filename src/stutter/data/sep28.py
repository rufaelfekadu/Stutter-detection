import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from stutter.utils.misc import load_audio_files


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