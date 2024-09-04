
from .trainer import Trainer
from stutter.models.multimodal import MultiModalClassification
from stutter.utils import AverageMeter
from transformers import TrainingArguments, Wav2Vec2Config, VivitConfig
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

STUTTER_CLASSES = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM']

class MultiModalTrainer(Trainer):

    def _init_meters(self):
        self.val_meters['accuracy'] = AverageMeter('val_accuracy', writer=self.logger)
        self.val_meters['f1_weighted'] = AverageMeter('val_f1_weighted', writer=self.logger)
        self.val_meters['f1_macro'] = AverageMeter('val_f1_macro', writer=self.logger)
        self.val_meters['f1_any'] = AverageMeter('val_f1_any', writer=self.logger)
        
        for classes in STUTTER_CLASSES:
            self.val_meters[classes] = AverageMeter(f'val_{classes}_f1', writer=self.logger)
        
        self.train_meters['loss'] = AverageMeter('train_loss', writer=self.logger)

        self.test_meters['accuracy'] = AverageMeter('test_accuracy', writer=self.logger)
        self.test_meters['f1_weighted'] = AverageMeter('test_f1_weighted', writer=self.logger)
        self.test_meters['f1_macro'] = AverageMeter('test_f1_macro', writer=self.logger)
        self.test_meters['f1_any'] = AverageMeter('test_f1_any', writer=self.logger)
        for classes in STUTTER_CLASSES:
            self.test_meters[classes] = AverageMeter(f'test_{classes}_f1', writer=self.logger)

    def get_model(self):
        # TODO: Fix this function
        wavconfig = Wav2Vec2Config.from_pretrained("superb/wav2vec2-base-superb-ks")
        vivitconfig = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        vivitconfig.num_frames=10
        vivitconfig.video_size=[10,224,224]

        model = MultiModalClassification(8, wavconfig, vivitconfig)
        optim = torch.optim.SGD(model.parameters(), lr=self.cfg.solver.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=5e-6)

        return model, optim, scheduler, None
    
    def get_dataloaders(self):
        # TODO: Fix this function
        df = load_from_disk("/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/dataset/stutter_hf/ds_5_multimodal_train")
        df = df.train_test_split(test_size=0.1, seed=42, shuffle=True)
        dataset = df['train']
        val_dataset = df['test']
        test_dataset =load_from_disk("/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/dataset/stutter_hf/ds_5_multimodal_test")
        dataset = dataset.map(lambda x: {"input_values": torch.tensor(x["input_values"][0]), 'attention_mask':torch.tensor(x['attention_mask'][0]), "labels": torch.tensor(x["labels"])})
        val_dataset = val_dataset.map(lambda x: {"input_values": torch.tensor(x["input_values"][0]), 'attention_mask':torch.tensor(x['attention_mask'][0]), "labels": torch.tensor(x["labels"])})
        test_dataset = test_dataset.map(lambda x: {"input_values": torch.tensor(x["input_values"][0]), 'attention_mask':torch.tensor(x['attention_mask'][0]), "labels": torch.tensor(x["labels"])})
        
        train_loder = DataLoader(dataset, batch_size=self.cfg.solver.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False)

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

    def parse_batch_train(self, batch):
        x = batch['input_values'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        y = batch['labels'].to(self.device)
        return x, y, attention_mask
    
    def train_step(self, batch):
        x, y, attention_mask = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=x, input_values=x, attention_mask=attention_mask, labels=y)
        return {
                'loss': loss
        }
    
    def val_step(self, batch):
        x, y, attention_mask = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=x, input_values=x, attention_mask=attention_mask, labels=y)
        metrics = self.compute_metrics(logits, y)
        metrics['val_loss'] = loss
        return metrics

    def test_step(self, batch):
        x, y, attention_mask = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=x, input_values=x, attention_mask=attention_mask, labels=y)
        metrics = self.compute_metrics(logits, y)
        metrics['loss'] = loss
        return metrics

