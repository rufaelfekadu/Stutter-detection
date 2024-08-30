import torch
from transformers import Trainer, TrainingArguments, AdamW, VivitConfig, VivitForVideoClassification

from stutter.data.hf_data import VivitVideoData
from stutter.utils.metrics import compute_video_classification_metrics

device = "cuda"
CACHE_DIR = "/tmp/"

annotator = "A3"
num_frames = 10
split_strategy = "ds_5"
label_type="secondary_event"
manifest_file = "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_5/reading/total_df.csv"
data_root = "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/datasets/fluencybank/ds_5/reading/clips/video"

# print("=================Loading data=====================")
# dataset = VivitVideoData(manifest_file, annotator, data_root, aggregate=True, label_category=label_type, num_proc=4)

# print("=================Preparing Dataset=====================")
# dataset = dataset.prepare_dataset()


# print("=================Initialize Model=====================")

# def initialise_model(shuffled_dataset, num_frames = 10, video_size = [10,224,224] ):
#     labels = shuffled_dataset['train'].features['labels'].names
#     config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
#     config.num_classes=len(labels)
#     config.id2label = {str(i): c for i, c in enumerate(labels)}
#     config.label2id = {c: str(i) for i, c in enumerate(labels)}
#     config.num_frames=num_frames
#     config.video_size=video_size

#     model = VivitForVideoClassification.from_pretrained(
#                 "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/vivit/ds_5_A1_secondary_event",
#                 ignore_mismatched_sizes=True,
#                 config=config,cache_dir=CACHE_DIR).to(device)
#     return model 

# model = initialise_model(dataset, num_frames = num_frames, video_size = [10,224,224])

# training_args = TrainingArguments(
#     output_dir=f"./results/{split_strategy}_{annotator}",         
#     num_train_epochs=1,             
#     per_device_train_batch_size=20, 
#     gradient_accumulation_steps=2,  
#     per_device_eval_batch_size=10,    
#     learning_rate=5e-05,            
#     weight_decay=0.01,              
#     logging_dir="./logs",           
#     logging_steps=10,                
#     seed=42,                       
#     evaluation_strategy="steps",    
#     eval_steps=10,                   
#     warmup_steps=int(0.1 * 20),      
#     optim="adamw_torch",          
#     lr_scheduler_type="linear",      
#     fp16=True,    
#     metric_for_best_model="accuracy",
#     load_best_model_at_end=True,
#     report_to='wandb',
#     run_name=f"{split_strategy}_{annotator}"
#     # auto_find_batch_size = True                   
# )

# optimizer = AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)


# trainer = Trainer(
#     model=model,                      
#     args=training_args, 
#     train_dataset=dataset["train"],      
#     eval_dataset=dataset["test"],       
#     optimizers=(optimizer, None), 
#     compute_metrics = compute_video_classification_metrics   
# )

# print("=================Begin Training=====================")
# train_results = trainer.train()
# trainer.save_model(f"./results/{split_strategy}_{annotator}/checkpoint_best")
# trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
# trainer.save_state()


# print("=================Begin Evaluation=====================")
# annotator = "Gold"
# test_data = VivitVideoData(manifest_file, annotator, data_root, aggregate=True, label_category=label_type, num_proc=4, split="test")
# test_data = test_data.prepare_dataset()
# config = VivitConfig.from_pretrained("/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/vivit/ds_5_A2_secondary_event")
# model = VivitForVideoClassification.from_pretrained(
#                 "/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/vivit/ds_5_A2_secondary_event",
#                 ignore_mismatched_sizes=True,
#                 config=config,cache_dir=CACHE_DIR).to(device)
# training_args = TrainingArguments("test_trainer", max_steps=1, per_device_eval_batch_size=10, eval_steps=1)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=test_data,
#     eval_dataset=test_data,
#     compute_metrics=compute_video_classification_metrics
# ,
# )

# # print("=================Preparing Dataset=====================")
# # test_data = dataset.prepare_dataset()
# trainer.evaluate(test_data)
