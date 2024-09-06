import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AdamW, VivitConfig, VivitForVideoClassification, Wav2Vec2Processor

from stutter.data.hf_data import VivitVideoData
from stutter.utils.metrics import compute_video_classification_metrics


import numpy as np
from .trainer import Trainer
from stutter.utils.meters import AverageMeter
from stutter.models.vivit import VivitForStutterClassification
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_metric
from sklearn.metrics import f1_score
acc = load_metric("accuracy")
f1 = load_metric("f1")

VIDEO_STUTTER_CLASSES = ['V', 'FG', 'HM']
class VivitForStutterTrainer(Trainer):
    
    def get_model(self):
        model = VivitForStutterClassification(self.cfg)
        optim = torch.optim.SGD(model.parameters(), lr=self.cfg.solver.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=5e-6)
        
        return model, optim, scheduler, None
    
    def get_dataloaders(self):
        # TODO: Fix this function
        df = load_from_disk("outputs/fluencybank/dataset/stutter_hf/ds_5_multimodal_train").remove_columns(['input_values', 'attention_mask'])
        df = df.train_test_split(test_size=0.1, seed=42, shuffle=True)
        dataset = df['train']
        val_dataset = df['test']
        test_dataset =load_from_disk("outputs/fluencybank/dataset/stutter_hf/ds_5_multimodal_test").remove_columns(['input_values', 'attention_mask'])
        dataset.set_format(type='torch', columns=[ 'pixel_values','labels'])
        val_dataset.set_format(type='torch', columns=[ 'pixel_values','labels'])
        test_dataset.set_format(type='torch', columns=[ 'pixel_values', 'labels'])
             
        train_loder = DataLoader(dataset, batch_size=self.cfg.solver.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False)

        return train_loder, val_loader, test_loader
    
    def compute_t1_metrics(self, y_pred, y_true):
        accuracy  = acc.compute(predictions=np.argmax(y_pred, axis=1), references=y_true)['accuracy']
        f1_score = f1.compute(predictions=np.argmax(y_pred, axis=1), references=y_true)['f1']
        return {"accuracy": accuracy, "f1_t1": f1_score}

    def compute_t2_metrics(self, y_pred, y_true):
        y_pred = (y_pred.sigmoid() > 0.5)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_any = f1_score((torch.sum(y_true,dim=1)>0).float(), (torch.sum(y_pred,dim=1)>0).float(), average='macro')
        metrics = {}
        for i,classes in enumerate(VIDEO_STUTTER_CLASSES):
            metrics[classes] = f1_score(y_true[:,i], y_pred[:,i])
        return {'accuracy': (y_pred==y_true.bool()).float().mean().item(), 
                'f1_weighted': f1_weighted, 'f1_macro': f1_macro, 'f1_any':f1_any, **metrics}
    
    def parse_batch_train(self, batch):
        image = batch['pixel_values'].to(self.device)
        y = batch['labels'].to(self.device)
        return image, y
    
    def train_step(self, batch):
        image, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, labels=y)
        return {
                'loss': loss.item()
        }
    
    def val_step(self, batch):
        image, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, labels=y)
        if self.cfg.task == 't1':
            metrics = self.compute_t1_metrics(logits.cpu(), y.cpu())
            metrics['loss'] = loss.item()
        else:
            metrics = self.compute_t2_metrics(logits.cpu(), y.cpu())
            metrics['loss'] = loss.item()
        return metrics

    def test_step(self, batch):
        image, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, labels=y)
        if self.cfg.task == 't1':
            metrics = self.compute_t1_metrics(logits.cpu(), y.cpu())
            metrics['loss'] = loss.item()
        else:
            metrics = self.compute_t2_metrics(logits.cpu(), y.cpu())
            metrics['loss'] = loss.item()
        print(*metrics)
        return metrics
    
    def _init_meters(self):
        self.val_meters['accuracy'] = AverageMeter('val_accuracy', writer=self.logger)
        self.val_meters['loss'] = AverageMeter('val_loss', writer=self.logger)
        if self.cfg.task == 't1':
            self.val_meters['f1_t1'] = AverageMeter('val_f1_t1', writer=self.logger)
        else:
            self.val_meters['f1_weighted'] = AverageMeter('val_f1_weighted', writer=self.logger)
            self.val_meters['f1_macro'] = AverageMeter('val_f1_macro', writer=self.logger)
            self.val_meters['f1_any'] = AverageMeter('val_f1_any', writer=self.logger)
            self.val_meters['lr'] = AverageMeter('lr', writer=self.logger)
        
            for classes in VIDEO_STUTTER_CLASSES:
                self.val_meters[classes] = AverageMeter(f'val_{classes}_f1', writer=self.logger)
            
        self.train_meters['loss'] = AverageMeter('train_loss', writer=self.logger)

        self.test_meters['accuracy'] = AverageMeter('test_accuracy', writer=self.logger)
        self.test_meters['loss'] = AverageMeter('test_loss', writer=self.logger)
        if self.cfg.task == 't1':
            self.test_meters['f1_t1'] = AverageMeter('test_f1_t1', writer=self.logger)
        else:
            self.test_meters['f1_weighted'] = AverageMeter('test_f1_weighted', writer=self.logger)
            self.test_meters['f1_macro'] = AverageMeter('test_f1_macro', writer=self.logger)
            self.test_meters['f1_any'] = AverageMeter('test_f1_any', writer=self.logger)
            for classes in VIDEO_STUTTER_CLASSES:
                self.test_meters[classes] = AverageMeter(f'test_{classes}_f1', writer=self.logger)
         
        




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
