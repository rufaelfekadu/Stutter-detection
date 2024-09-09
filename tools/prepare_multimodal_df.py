from datasets import Dataset
from tqdm import tqdm

from transformers import  VivitImageProcessor, AutoFeatureExtractor
from stutter.utils.data import read_video_pyav, sample_frame_indices, make_video_dataframe, prepare_hf_dataset_video

feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks", cache_dir="/tmp/")
processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400", cache_dir="/tmp/")


for annotator in tqdm(["sad"], total=3, leave=False):
    num_frames = 10
    split = "train"
    manifest_file =f"/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_label/reading/{annotator}/total_label.csv"
    data_root = f"/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_label/reading/{annotator}/clips/video"

    df = make_video_dataframe(manifest_file, annotator, data_root, aggregate=False, split=split)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(prepare_hf_dataset_video,
                        fn_kwargs={"label_type": 'stutter_category', 'processor' : processor, 
                                    'extractor': feature_extractor}, 
                        num_proc=4, writer_batch_size=100)

    dataset.save_to_disk(f"outputs/fluencybank/dataset/stutter_hf/label_split/{annotator}_multimodal_train")