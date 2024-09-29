import torch
from datasets import Dataset
from transformers import VivitImageProcessor
from stutter.utils.data import read_video_pyav, sample_frame_indices, make_video_dataframe, prepare_hf_dataset_video
from stutter.utils.annotation import LabelMap
import pandas as pd
import numpy as np
import json
from glob import glob
import soundfile as sf

from transformers import AutoFeatureExtractor, AutoProcessor
from datasets import load_dataset, Dataset

class HuggingFaceDataset():
    def __init__(self, manifest_file, annotator, data_root, aggregate,label_category,clip_len:int =10, num_proc:int=2, split:str="train"):
        self.manifest_file = manifest_file
        self.annotator = annotator
        self.data_root = data_root
        self.aggregate = aggregate
        self.clip_len = clip_len
        self.label_category = label_category
        self.num_proc = num_proc
        self.split = split
        self.processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400", cache_dir="/tmp/")
        self.dataset = Dataset.from_pandas(make_video_dataframe(self.manifest_file,self.annotator,root=self.data_root,aggregate=self.aggregate, split=self.split))
    
    # def __len__(self):
    #     return len(self.dataset)
    def prepare_dataset(self):    
        dataset = self.dataset.map(prepare_hf_dataset_video, fn_kwargs={'label_type':self.label_category, 'processor':self.processor, 'clip_len':self.clip_len} , 
                                   batched=False, num_proc=self.num_proc, remove_columns=self.dataset.column_names)
        dataset = dataset.class_encode_column("labels")
        shuffled_dataset = dataset.shuffle(seed=42)
        # if self.split == "train":
        #     shuffled_dataset = shuffled_dataset.train_test_split(test_size=0.1)
        return shuffled_dataset

class HuggingFaceDataset(Dataset):
    __acceptable_params = ['label_path', 'encoder_name', 'cache_dir', 'split_file', 'ckpt']
    def __init__(self, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        self.class_names = ['SR','ISR','MUR','P','B', 'V']
        self.label_map = LabelMap()
        # Attempt to load the dataset from the Hugging Face Hub
        try:
            dataset = load_dataset(self.ckpt, cache_dir=self.cache_dir)
            super().__init__(data=dataset.data_)  # Call parent constructor with loaded dataset
            print(f"Loaded dataset '{self.cache_dir}' from the Hub")
        except Exception as e:
            print(f"Failed to load dataset '{self.cache_dir}' from the Hub: {e}")
            # Call the prep_data function and build the dataset from the returned dictionary
            self.audio_processor = AutoProcessor.from_pretrained(self.encoder_name, cache_dir=self.cache_dir)
            data_dict = self.prep_data()
            super().__init__(data=data_dict)  
            self.push_to_hub(self.ckpt)
        
        self.split = np.zeros(len(self.dataset))
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        for i, audio_file in enumerate(self.dataset['file_name']):
            if audio_file in split_data['train']:
                self.split[i] = 0
            elif audio_file in split_data['val']:
                self.split[i] = 1
            else:
                self.split[i] = 2

    def prep_data(self):

        audio_files = glob(f"{self.root}/**/*.wav", recursive=True)
        data_dict = {'audio': [], 'labels': []}
        for audio_file in audio_files:
            audio, sr = sf.read(audio_file)
            audio = self.audio_processor(audio, sr)['input_values'][0]
            data_dict['audio'].append(audio)
            label_file = audio_file.replace('.wav', '.txt').replace('clips', 'label').replace('audio', 'sed')
            data_dict['labels'].append(self.prep_label(label_file))
        return data_dict
    
    @staticmethod
    def prep_label(class_names, label_file):
        with open(label_file, 'r') as f:
            l = torch.zeros(len(class_names))
            temp = f.readlines()
            temp = [x.strip().split(',') for x in temp]
            for t in temp:
                label = class_names.index(t[2])
                l[label] = 1   
        return l


if __name__ == "__main__":
    kwargs = {
        'label_path': 'data/labels',
        'encoder_name': 'facebook/wav2vec2-base-960h',
        'cache_dir': 'data/cache',
        'split_file': 'data/split.json'
    }

