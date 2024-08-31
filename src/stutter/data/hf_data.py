import torch
from datasets import Dataset
from transformers import VivitImageProcessor
from stutter.utils.data import read_video_pyav, sample_frame_indices, make_video_dataframe, prepare_hf_dataset_video

class VivitVideoData():
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
        
    
if __name__ == "__main__":
    annotator = "A3"
    label_type="secondary_event"
    manifest_file = "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_5/reading/total_df.csv"
    data_root = "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_5/reading/clips/video"

    dataset = VivitVideoData(manifest_file, annotator, data_root, aggregate=True, label_category=label_type)
    dataset = dataset.prepare_dataset()

