"/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/vivit/ds_5_None_secondary_event"

import numpy as np
import torch

from tqdm import tqdm
from stutter.data.hf_data import VivitVideoData
from stutter.utils.metrics import compute_video_classification_metrics
from transformers import VivitImageProcessor, VivitForVideoClassification

annotator = "A1"
num_frames = 10
split_strategy = "ds_5"
label_type="secondary_event"
manifest_file = "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_5/reading/total_df.csv"
data_root = "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_5/reading/clips/video"

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")



for annotator in ['A2', "A3"]:
    test_data = VivitVideoData(manifest_file, annotator, data_root, aggregate=True, label_category=label_type, num_proc=1, split="test")
    test_data = test_data.prepare_dataset()
    model = VivitForVideoClassification.from_pretrained(f"/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/vivit/ds_5_{annotator}_secondary_event/best")
    model.to("cuda")
    preds = []
    labels = []
    print(f"Evaluating {annotator}...")
    for id, example in tqdm(enumerate(test_data)):
        # breakpoint()
        inputs = example['pixel_values']
        with torch.no_grad():
            outputs = model(torch.tensor(example['pixel_values']).unsqueeze(0).to("cuda"))
            logits = outputs.logits
        preds.append(logits.argmax(-1).item())
        labels.append(example['labels'])
        # print(model.config.id2label[predicted_label])
        
    from datasets import load_metric
    acc = load_metric("accuracy")
    f1 = load_metric("f1")

    print(f"Accuracy: {acc.compute(predictions=preds, references=labels)}")
    print(f"F1: {f1.compute(predictions=preds, references=labels)}")