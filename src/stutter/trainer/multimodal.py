import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .trainer import Trainer
from stutter.models.multimodal import MultiModalClassification
from stutter.utils.meters import AverageMeter
from transformers import TrainingArguments, Wav2Vec2Config, VivitConfig
from datasets import load_from_disk, Dataset, concatenate_datasets, load_metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
acc = load_metric("accuracy")
f1 = load_metric("f1")

from typing import Optional

STUTTER_CLASSES = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM']

def collate_fn(batch):
    max_len = max([len(x['input_values'][0]) for x in batch])
    for x in batch:
        x['input_values'] = F.pad(x['input_values'],(0, max_len - x['input_values'][0].shape[0]), "constant", 0)
        x['attention_mask'] = F.pad(x['attention_mask'],(0, max_len - x['attention_mask'][0].shape[0]), "constant", 0)
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}

class MultiModalTrainer(Trainer):

    def _init_meters(self):
        self.val_meters['accuracy'] = AverageMeter('val_accuracy', writer=self.logger)
        self.val_meters['f1_weighted'] = AverageMeter('val_f1_weighted', writer=self.logger)
        self.val_meters['f1_macro'] = AverageMeter('val_f1_macro', writer=self.logger)
        self.val_meters['f1_any'] = AverageMeter('val_f1_any', writer=self.logger)
        self.val_meters['lr'] = AverageMeter('lr', writer=self.logger)
        self.val_meters['loss'] = AverageMeter('val_loss', writer=self.logger)
        
        for classes in STUTTER_CLASSES:
            self.val_meters[classes] = AverageMeter(f'val_{classes}_f1', writer=self.logger)
        
        self.train_meters['loss'] = AverageMeter('train_loss', writer=self.logger)

        self.test_meters['accuracy'] = AverageMeter('test_accuracy', writer=self.logger)
        self.test_meters['f1_weighted'] = AverageMeter('test_f1_weighted', writer=self.logger)
        self.test_meters['f1_macro'] = AverageMeter('test_f1_macro', writer=self.logger)
        self.test_meters['f1_any'] = AverageMeter('test_f1_any', writer=self.logger)
        self.test_meters['loss'] = AverageMeter('test_loss', writer=self.logger)
        for classes in STUTTER_CLASSES:
            self.test_meters[classes] = AverageMeter(f'test_{classes}_f1', writer=self.logger)

    def get_model(self):
        # TODO: Fix this function
        wavconfig = Wav2Vec2Config.from_pretrained("superb/wav2vec2-base-superb-ks")
        vivitconfig = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        vivitconfig.num_frames=10
        vivitconfig.video_size=[10,224,224]

        model = MultiModalClassification(self.cfg.model.output_size, wavconfig, vivitconfig)
        optim = torch.optim.SGD(model.parameters(), lr=self.cfg.solver.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=5e-6)

        return model, optim, scheduler, None
    
    def get_dataloaders(self):
        # TODO: Fix this function
        # df = load_from_disk("outputs/fluencybank/dataset/stutter_hf/ds_5_multimodal_train")
        # test_dataset = load_from_disk("outputs/fluencybank/dataset/stutter_hf/label_split/Gold_multimodal_test")
        df = load_from_disk(f"{self.cfg.data.label_path}train")
        test_dataset = load_from_disk("outputs/fluencybank/dataset/stutter_hf/label_split/Gold_multimodal_test")
        
        if self.cfg.tasks[0] == 't1':
            if self.cfg.data.annotation != 'any':
                ann_index = STUTTER_CLASSES.index(self.cfg.data.annotation)
                print(f"Label is {self.cfg.data.annotation} at index {ann_index}")
                df = df.map(lambda x: {'labels': x['labels'][ann_index]}, num_proc=8)
                test_dataset = test_dataset.map(lambda x: {'labels': x['labels'][ann_index]}, num_proc=8)
            else:
                df = df.map(lambda x: {'labels': max(x['labels'])}, num_proc=8)
                test_dataset = test_dataset.map(lambda x: {'labels': max(x['labels'])}, num_proc=8)   
            class_counts = df['labels'].count(0), df['labels'].count(1)
            print(f"Class counts are {class_counts}")
            minority_class = 0 if class_counts[0] < class_counts[1] else 1

            # Filter out samples to undersample the majority class
            minority_class_samples = df.filter(lambda example: example['labels'] == minority_class, num_proc=8, writer_batch_size=100)
            majority_class_samples = df.filter(lambda example: example['labels'] != minority_class, num_proc=8, writer_batch_size=100)

            # Randomly undersample the majority class to the same size as the minority class
            majority_class_samples = majority_class_samples.shuffle(seed=42).select(range(len(minority_class_samples)))

            # Combine minority and undersampled majority class
            undersampled_dataset = concatenate_datasets([minority_class_samples, majority_class_samples])
            del minority_class_samples, majority_class_samples
            # Shuffle the final undersampled dataset
            df = undersampled_dataset.shuffle(seed=42)
            class_counts = undersampled_dataset['labels'].count(0), undersampled_dataset['labels'].count(1)
            print(f"Undersampled Class counts are {class_counts}")
            # df = undersampled_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
            
        
        df = df.train_test_split(test_size=0.1, seed=42, shuffle=True)
        dataset = df['train']
        val_dataset = df['test']
        dataset.set_format(type='torch', columns=[ 'pixel_values','input_values', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=[ 'pixel_values','input_values', 'attention_mask', 'labels'])
        test_dataset.set_format(type='torch', columns=[ 'pixel_values','input_values', 'attention_mask', 'labels'])
             
        train_loder = DataLoader(dataset, batch_size=self.cfg.solver.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loder, val_loader, test_loader
    
    def compute_metrics(self, y_pred, y_true):
        y_pred = (y_pred.sigmoid() > 0.5)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_any = f1_score((torch.sum(y_true,dim=1)>0).float(), (torch.sum(y_pred,dim=1)>0).float(), average='macro')
        metrics = {}
        for i,classes in enumerate(STUTTER_CLASSES):
            metrics[classes] = f1_score(y_true[:,i], y_pred[:,i])
        return {'accuracy': (y_pred==y_true.bool()).float().mean().item(), 
                'f1_weighted': f1_weighted, 'f1_macro': f1_macro, 'f1_any':f1_any, **metrics}

    def compute_t1_metrics(self, y_pred, y_true):
        y_pred = torch.round(y_pred).cpu().numpy()
        accuracy  = acc.compute(predictions=y_pred, references=y_true)['accuracy']
        f1_score = f1.compute(predictions=y_pred, references=y_true)['f1']
        return {"accuracy": accuracy, f"f1_{self.cfg.data.annotation}": f1_score}

    def compute_t2_metrics(self, y_pred, y_true):
        y_pred = (y_pred.sigmoid() > 0.5)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_any = f1_score((torch.sum(y_true,dim=1)>0).float(), (torch.sum(y_pred,dim=1)>0).float(), average='macro')
        metrics = {}
        for i,classes in enumerate(STUTTER_CLASSES):
            metrics[classes] = f1_score(y_true[:,i], y_pred[:,i])
        return {'accuracy': (y_pred==y_true.bool()).float().mean().item(), 
                'f1_weighted': f1_weighted, 'f1_macro': f1_macro, 'f1_any':f1_any, **metrics}
    
    def parse_batch_train(self, batch):
        image = batch['pixel_values'].to(self.device)
        audio = batch['input_values'].squeeze(1).to(self.device)
        attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
        if self.cfg.tasks[0] == 't1':
            y = batch['labels'].to(self.device).float()
        else:
            y = batch['labels'].to(self.device)
        return image, audio, attention_mask, y
    
    def train_step(self, batch):
        image, audio, attention_mask, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, input_values=audio, attention_mask=attention_mask, labels=y)
        loss.backward()
        self.optimizer.step()
        return {
                'loss': loss.item()
        }
    
    def val_step(self, batch):
        image, audio, attention_mask, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, input_values=audio, attention_mask=attention_mask, labels=y)
        if self.cfg.tasks[0] == 't1':
            metrics = self.compute_t1_metrics(logits, y.cpu())
            metrics['loss'] = loss.item()
        else:
            metrics = self.compute_t2_metrics(logits.cpu(), y.cpu())
            metrics['loss'] = loss.item()
        return metrics

    def test_step(self, batch):
        image, audio, attention_mask, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, input_values=audio, attention_mask=attention_mask, labels=y)
        return loss, logits, y
    
    # def after_test(self):
    #     preds = torch.cat(self.preds, axis=0)
    #     labels = torch.cat(self.labels, axis=0).cpu().numpy()
    #     preds = (preds.sigmoid() > 0.5).cpu().numpy()
    #     cm = confusion_matrix(labels, preds)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    #     # log the confusion matrix
    #     # self.logger.add_figure(f'confusion_matrix', disp.figure_)
    #     for i, s in enumerate(STUTTER_CLASSES):
    #         cm = confusion_matrix(labels[:,i], preds[:,i])
    #         disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    #         # log the confusion matrix
    #         self.logger.add_figure(f'confusion_matrix_{s}', disp.figure_)
    
    def validate(self):
        self.stage = 'val'
        self.model.eval()
        self._reset_meters(self.val_meters)

        with torch.no_grad():
            total_loss = 0
            for batch in self.val_loader:
                losses = self.val_step(batch)
                total_loss += losses['loss']
                for key, loss in losses.items():
                    self._update_meter(self.val_meters, f'{key}', loss)

            
            # log the learning rate
            self.val_meters['lr'].update(self.scheduler.get_last_lr()[0])
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss/len(self.val_loader))
            else:
                self.scheduler.step()

            self._write_meters(self.val_meters)
            total_loss /= len(self.val_loader)

            if total_loss < self.best_val_loss:
                self.best_val_loss = total_loss
                self.best_epoch = self.epoch
                self.save_model('best_checkpoint.pt')
                self.patience = self.cfg.solver.es_patience
                print(f"Saving best model at epoch {self.epoch} with loss {self.best_val_loss}")
            else:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Early Stopping at epoch {self.epoch} \nbest epoch at {self.best_epoch} with loss {self.best_val_loss}")
                    return True
        
        # self.after_validation()
        self.model.train()
        return False
    # def test(self, loader=None, name='test'):
    #     loader = loader or self.test_loader
    #     self.stage = 'test'
    #     # self.load_model()
    #     self.model.eval()
    #     self._reset_meters(self.test_meters)
    #     self.preds = []
    #     self.labels = []    
    #     with torch.no_grad():
    #         for batch in tqdm(loader, desc='Evaluating Test Set', total=len(loader)):
    #             loss, logits, labels = self.test_step(batch)
    #             self.preds.append(logits)
    #             self.labels.append(labels)
    #     logits = torch.cat(self.preds, axis=0)
    #     # del self.preds
    #     y = torch.cat(self.labels, axis=0)              
    #     # del self.labels 
    #     metrics = self.compute_metrics(logits.cpu(), y.cpu())
    #     metrics['loss'] = loss.item()
    #     for key_metric, val in metrics.items():
    #         print(key_metric, val)
    #         self._update_meter(self.test_meters, f'{key_metric}', val)
            
    #     self._write_meters(self.test_meters)
    #     self.after_test()
    
    def test(self, loader=None, name='test'):
        loader = loader or self.test_loader
        self.stage = 'test'
        # self.load_model()
        self.model.eval()
        self._reset_meters(self.test_meters)
        self.preds = []
        self.labels = []    
        with torch.no_grad():
            for batch in tqdm(loader, desc='Evaluating Test Set', total=len(loader)):
                loss, logits, labels = self.test_step(batch)
                self.preds.append(logits)
                self.labels.append(labels)
        logits = torch.cat(self.preds, axis=0)
        del self.preds
        y = torch.cat(self.labels, axis=0)              
        del self.labels 
        if self.cfg.tasks[0] == 't1':
            metrics = self.compute_t1_metrics(logits, y.cpu())
            metrics['loss'] = loss.item()
        else:
            metrics = self.compute_t2_metrics(logits.cpu(), y.cpu())
            metrics['loss'] = loss.item()
        for key_metric, val in metrics.items():
            print(key_metric, val)
            self._update_meter(self.test_meters, f'{key_metric}', val)
            
        self._write_meters(self.test_meters)
        self.after_test()