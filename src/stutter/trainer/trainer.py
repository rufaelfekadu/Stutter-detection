import os
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
from tqdm import tqdm

from stutter.utils.annotation import LabelMap
from stutter.data import get_dataloaders
from stutter.models import build_model
from stutter.utils.meters import LossMeter, AverageMeter
from stutter.utils.metrics import f1_score_per_class, f1_score_, weighted_accuracy, multilabel_EER, binary_acc, binary_f1, iou_metric
from stutter.utils.loss import build_loss, CCCLoss
import librosa
import matplotlib.pyplot as plt
from transformers import AutoModelForAudioClassification, TrainingArguments
from transformers import Trainer as HuggingFaceTrainer


metric_bank={
    'acc': lambda pred, y: (torch.argmax(pred, dim=1) == y).sum().item() / y.size(0),
    'f1': f1_score_per_class,
    'f1_macro': f1_score_,
    'wacc': weighted_accuracy,
    'eer': multilabel_EER,
    'binary_acc': binary_acc,
    'binary_f1': binary_f1,
    'iou': iou_metric,
}

class BaseTrainer(object):

    def __init__(self, device, logger=None):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.device = device
        self.logger = logger
        self.train_meters = {}
        self.test_meters = {}
        self.val_meters = {}
        self.tasks = None
        self.stage = None
        self.epoch = 0
        self.best_epoch = 0

        
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    
class Trainer(BaseTrainer):

    def __init__(self, cfg, logger=None, metrics=['acc', 'f1', 'wacc', 'eer']):
        super(Trainer, self).__init__(cfg.solver.device, logger)

        self.cfg = cfg
        self.tasks = cfg.tasks

        self.metrics = metrics
        for metric in self.metrics:
            setattr(self, metric, metric_bank[metric])

        self._init_meters()
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(cfg)

        # build model
        self.model, self.optimizer, self.scheduler = build_model(cfg)
        self.model = self.model.to(self.device)
        self.criterion = build_loss(cfg)

        self.best_val_loss = float('inf')
        self.patience = cfg.solver.es_patience
        print(self.model)
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     self.model = torch.nn.DataParallel(self.model)

    def _init_meters(self):
        for key in self.tasks:
            self.train_meters[f'{key}_train_loss'] = LossMeter(name=f'{key}_train_loss', writer=self.logger)
            self.val_meters[f'{key}_val_loss'] = LossMeter(name=f'{key}_val_loss', writer=self.logger)
            for metric in self.metrics:
                self.test_meters[f'{key}_test_{metric}'] = AverageMeter(name=f'{key}_test_{metric}', writer=self.logger)
                self.test_meters[f'{key}_val_{metric}'] = AverageMeter(name=f'{key}_val_{metric}', writer=self.logger)
                self.test_meters[f'{key}_train_{metric}'] = AverageMeter(name=f'{key}_train_{metric}', writer=self.logger)

        self.val_meters['lr'] = LossMeter(name='lr', writer=self.logger)

    def _reset_meters(self, meters):
        for meter in meters.values():
            meter.reset()
    
    def _write_meters(self, meters):
        for meter in meters.values():
            meter.write()
    
    def _update_meter(self, meters, key, val):
        if key in meters:
            meters[key].update(val)

    def train(self):
        self.stage = 'train'
        es = False
        for epoch in range(self.cfg.solver.epochs):

            self.model.train()
            self._reset_meters(self.train_meters)
            tq_obj = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
            for i, batch in enumerate(tq_obj):
                self.optimizer.zero_grad()
                losses = self.train_step(batch)
                for key, loss in losses.items():
                    self._update_meter(self.train_meters, f'{key}_train_loss', loss)
                    tq_obj.set_postfix({ key: loss })

                if i % self.cfg.solver.log_steps == 0:
                    self._write_meters(self.train_meters)

                if i % self.cfg.solver.eval_steps == 0:
                    es = self.validate()
                    if es: break
            # self._write_meters(self.train_meters)

            if es: break

            self.epoch += 1
    
    def validate(self):
        self.stage = 'val'
        tasks = self.tasks
        self.model.eval()
        self._reset_meters(self.val_meters)

        with torch.no_grad():
            total_loss = 0
            for batch in self.val_loader:
                losses = self.val_step(batch)
                total_loss += sum(losses.values())
                for key, loss in losses.items():
                    self._update_meter(self.val_meters, f'{key}_val_loss', loss)

            # log the learning rate
            self.val_meters['lr'].update(self.scheduler.get_last_lr()[0])
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss/len(self.val_loader))
            else:
                self.scheduler.step()
            # self.scheduler.step(self.val_meters[f'{key}_val_loss'].avg)
            # self.scheduler.step(self.epoch)

            self._write_meters(self.val_meters)
            #  Early Stopping
            # total_loss = sum([self.val_meters[f'{key}_val_loss'].avg for key in tasks])
            if total_loss < self.best_val_loss:
                self.best_val_loss = total_loss
                self.best_epoch = self.epoch
                self.save_model('best_checkpoint.pt')
                self.patience = self.cfg.solver.es_patience
            else:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Early Stopping at epoch {self.epoch} \nbest epoch at {self.best_epoch} with loss {self.best_val_loss}")
                    return True
        self.model.train()
        return False

    def test(self, loader=None, name='test'):
        loader = loader or self.test_loader
        self.stage = 'test'
        self.model.eval()
        self._reset_meters(self.test_meters)

        with torch.no_grad():
            for batch in loader:
                metrics = self.test_step(batch)
                for key_task, metric in metrics.items():
                    for key_metric, val in metric.items():
                        self._update_meter(self.test_meters, f'{key_task}_{name}_{key_metric}', val)
                
        self._write_meters(self.test_meters)
        self.after_test()
        
    def after_test(self):
        pass
    
    def save_model(self, path):
        path = os.path.join(self.cfg.output.checkpoint_dir, path)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='best_checkpoint.pt'):
        path = os.path.join(self.cfg.output.checkpoint_dir, path)
        print(f'Loading model from {path}')
        self.model.load_state_dict(torch.load(path))

    def compute_metrics(self, pred, y):
        vals = {}
        for metric in self.metrics:
            vals[metric] = getattr(self, metric)(pred, y)
        return vals
    
    def parse_batch_train(self, batch):
        # x = torch.cat([batch['mel_spec'], batch['f0']], dim=1)
        x = batch['mel_spec']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def parse_batch_test(self, batch):
        raise NotImplementedError
    
    def train_step(self, x, y):
        raise NotImplementedError
    
    def val_step(self, x):
        raise NotImplementedError
    
    def test_step(self, x):
        raise NotImplementedError
    
    def inference(self, x):
        raise NotImplementedError
    
class MTLTrainer(Trainer):

    def __init__(self, cfg, logger=None, metrics=['acc']):
        super(MTLTrainer, self).__init__(cfg, logger, metrics = metrics)
        # assert len(self.tasks) > 1, 'Specify more than 1 task for MTL'

        self.num_classes = cfg.model.output_size
        self.test_preds = []
        self.test_labels = []
        
    def train_step(self, batch):

        x, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        loss1, loss2 = torch.tensor(0), torch.tensor(0)

        if 't1' in self.tasks:
            y_t1 = (y>=2).float()
            y_t1 = (torch.sum(y[:, :-1], dim=1)>0).long()
            loss1 = self.criterion['t1'](pred_t1, y_t1)

        if 't2' in self.tasks:
            if  not isinstance(self.criterion['t2'], CCCLoss):
                y_t2 = (y>=2).float()
            loss2 = self.criterion['t2'](pred_t2, y_t2)

        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()

        return{
            't1': loss1.item(),
            't2': loss2.item()
        }
    
    def val_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        loss1, loss2 = torch.tensor(0), torch.tensor(0)
        if 't1' in self.tasks:
            y_t1 = (y>=2).float()
            y_t1 = (torch.sum(y[:, :self.num_classes-1], dim=1)>0).long()
            loss1 = self.criterion['t1'](pred_t1, y_t1)
        if 't2' in self.tasks:
            if  not isinstance(self.criterion['t2'], CCCLoss):
                y_t2 = (y>=2).float()
            loss2 = self.criterion['t2'](pred_t2, y_t2)
        return{
            't1': loss1.item(),
            't2': loss2.item()
        }
    
    def test_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        metrics_1, metrics_2 = {}, {}
        if 't1' in self.tasks:
            y_t1 = (y>=2).int()
            y_t1 = (torch.sum(y[:, :self.num_classes-1], dim=1)>0).int()
            pred_t1 = torch.argmax(pred_t1, dim=1)
            metrics_1 = self.compute_metrics(pred_t1, y_t1)
            self.test_preds.append(pred_t1.cpu().numpy())
            self.test_labels.append(y_t1.cpu().numpy())

        if 't2' in self.tasks:
            y_t2 = (y>=2).int()
            pred_t2 = (torch.sigmoid(pred_t2)>=0.5).int()
            metrics_2 = self.compute_metrics(pred_t2, y_t2)
            self.test_preds.append(pred_t2.cpu().numpy())
            self.test_labels.append(y_t2.cpu().numpy())

        return {
            't1': metrics_1,
            't2': metrics_2
        }
    
    def inference(self, batch):
        tasks = self.tasks
        X, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(X, tasks=tasks)
        return {
            't1': pred_t1,
            't2': pred_t2
        }

    # def after_test(self):
    #     preds = np.concatenate(self.test_preds, axis=0)
    #     labels = np.concatenate(self.test_labels, axis=0)
    #     num_classes = self.cfg.model.output_size
    #     cm = confusion_matrix(labels, preds)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    #     # log the confusion matrix
    #     self.logger.add_figure(f'confusion_matrix_t1', disp.figure_)
    
    def after_test(self):
        preds = np.concatenate(self.test_preds, axis=0)
        labels = np.concatenate(self.test_labels, axis=0)
        for i in range(self.num_classes):
            cm = confusion_matrix(labels[:,i], preds[:,i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
            # log the confusion matrix
            self.logger.add_figure(f'confusion_matrix_{i}', disp.figure_)

class YohoTrainer(Trainer):
    def __init__(self, cfg, logger=None, metrics=['acc']):
        super(YohoTrainer, self).__init__(cfg, logger, metrics = metrics)
        self.criterion = self.criterion['t2']
        self.test_preds = []
        self.test_labels = []
        self.num_classes = 11

    def parse_batch_train(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def train_step(self, batch):
        x, y = self.parse_batch_train(batch)

        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return{
            't2': loss.item()
        }

    # def test_step(self, batch):

    #     x, y = self.parse_batch_train(batch)
    #     preds = self.model(x.squeeze(1))
    #     self.test_preds.append(preds)
    #     self.test_labels.append(y)
    #     metrics = self.compute_metrics(preds, y)

    #     return {
    #         't2': metrics
    #     }

    def test(self, loader=None, name='test'):
        # generate the reference txt_files
        loader = loader or self.test_loader
        self.stage = 'test'
        self.model.eval()
        self._reset_meters(self.test_meters)

        for batch in loader:
            x, y = self.parse_batch_train(batch)
            preds = self.model(x.squeeze(1))
            self.test_preds.append(preds)
            self.test_labels.append(y)
        
        self.test_preds
        

    def val_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        return{
            't2': loss.item()
        }
    
    def after_test(self):
        label_map = LabelMap()
        preds = torch.concat(self.test_preds, axis=0)
        labels = torch.concat(self.test_labels, axis=0)

        preds[:,:,2:-1] = (torch.sigmoid(preds[:,:,2:-1]) >= 0.5).int()
        
        label_time = labels[:,:,0:2]
        # visualize the predictions and the label plot the time span and label
        breakpoint()
        for i in range(5):
            events = preds[i,:,:]
            plt.figure()
            for j in range(events.size(0)):
                if (events[j,2:-1] >=0.5)== 1:
                    plt.plot(events[j,0:2], [1, 1], color='-ro', label=label_map.strfromlabel(events[j,2:]))
            
            ground_truth = labels[i,:]
            for j in range(ground_truth.size(0)):
                if ground_truth[j] == 1:
                    plt.plot(label_time[j,0:2], [2, 2], color='-bo', label=label_map.strfromlabel(ground_truth[j,2:]))
         
class Wave2vecTrainer(Trainer):
    def __init__(self, cfg, logger=None, metrics=['acc']):
        super(Wave2vecTrainer, self).__init__(cfg, logger, metrics = metrics)
        
        self.model = AutoModelForAudioClassification.from_pretrained(cfg.model.name)
        self.num_classes = cfg.model.output_size
        self.test_preds = []
        self.test_labels = []
    
    def train(self):
        self.hug_trainer.train()
    
    def test(self):
        self.hug_trainer.evaluate()
        

trainer_registery = {
    'mtl': MTLTrainer,
    'yoho': YohoTrainer,
}

def build_trainer(cfg, logger=None, metrics=['f1']):
    trainer = trainer_registery[cfg.setting](cfg, logger, metrics)
    return trainer