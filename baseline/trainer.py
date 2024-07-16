import os
import torch
from utils import AverageMeter, LossMeter, weighted_accuracy, EER, f1_score_, f1_score_per_class
from tqdm import tqdm
from data import get_dataloaders
from models import build_model

metric_bank={
    'acc': lambda pred, y: (torch.argmax(pred, dim=1) == y).sum().item() / y.size(0),
    'f1': f1_score_,
    'wacc': weighted_accuracy,
    'eer': EER
}

class BaseTrainer(object):

    def __init__(self, criterion, device, logger=None):
        self.model = None
        self.optimizer = None
        self.criterion = criterion
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

    def __init__(self, cfg, criterion, logger=None, metrics=['acc', 'f1', 'wacc', 'eer']):
        super(Trainer, self).__init__(criterion, cfg.solver.device, logger)

        self.cfg = cfg
        self.tasks = list(self.criterion.keys())

        self.metrics = metrics
        for metric in self.metrics:
            setattr(self, metric, metric_bank[metric])

        self._init_meters()
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(cfg)

        # build model
        self.model = build_model(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.solver.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[34, 58, 72], gamma=0.4)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40, eta_min=0)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.solver.lr_step, gamma=cfg.solver.lr_decay)

        # init best val loss for early stoping
        self.best_val_loss = float('inf')
        self.patience = cfg.solver.es_patience
        
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     self.model = torch.nn.DataParallel(self.model)
        
        # makedir for saving checkpoints
        os.makedirs(cfg.output.save_dir, exist_ok=True)

    def _init_meters(self):
        for key in self.tasks:
            self.train_meters[f'{key}_train_loss'] = LossMeter(name=f'{key}_train_loss', writer=self.logger)
            self.val_meters[f'{key}_val_loss'] = LossMeter(name=f'{key}_val_loss', writer=self.logger)
            for metric in self.metrics:
                self.test_meters[f'{key}_test_{metric}'] = AverageMeter(name=f'{key}_test_{metric}', writer=self.logger)
                self.test_meters[f'{key}_val_{metric}'] = AverageMeter(name=f'{key}_val_{metric}', writer=self.logger)
                self.test_meters[f'{key}_train_{metric}'] = AverageMeter(name=f'{key}_train_{metric}', writer=self.logger)

    def _reset_meters(self, meters):
        for meter in meters.values():
            meter.reset()
    
    def _write_meters(self, meters):
        for meter in meters.values():
            meter.write()
    
    def _update_meters(self, meters, val):
        if not isinstance(val, list):
            val = [val]
        for key, value in zip(self.tasks, val):
            meters[f'{key}_{self.stage}_loss'].update(value)

    def train(self):
        self.stage = 'train'
        for epoch in range(self.cfg.solver.epochs):
            self.model.train()
            self._reset_meters(self.train_meters)

            tq_obj = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
            for batch in tq_obj:
                losses = self.forward_backward(batch)
                if not isinstance(losses, tuple):
                    losses = [losses]
                for key, loss in zip(self.tasks, losses):
                    self.train_meters[f'{key}_train_loss'].update(loss)
                    tq_obj.set_postfix({ key: loss })

            

            self._write_meters(self.train_meters)

            # validation
            if not self.validate(): 
                self.save_model('last_checkpoint.pt')
                break

            self.epoch += 1

    def _inference(self, batch):
        _,y = self.parse_batch_test(batch)
        preds =  self.model_inference(batch)

        if self.stage == 'val':
            losses = self.criterion(preds, y)
            self._update_meters(self.val_meters, losses)

        elif self.stage == 'test':
            metrics = self.compute_metrics(preds, y)

        return preds, losses, metrics
    
    def validate(self):
        self.stage = 'val'
        tasks = self.tasks
        self.model.eval()
        self._reset_meters(self.val_meters)

        with torch.no_grad():
            for batch in self.val_loader:
                _, losses, _ = self.model_inference(batch)
                if not isinstance(losses, tuple):
                    losses = [losses]
                for key, loss in zip(tasks, losses):
                    self.val_meters[f'{key}_val_loss'].update(loss)

            self.scheduler.step(self.val_meters[f'{key}_val_loss'].avg)
            # self.scheduler.step(self.epoch)

            self._write_meters(self.val_meters)
            #  Early Stopping
            total_loss = sum([self.val_meters[f'{key}_val_loss'].avg for key in tasks])
            if total_loss < self.best_val_loss:
                self.best_val_loss = total_loss
                self.best_epoch = self.epoch
                self.save_model('best_checkpoint.pt')
                self.patience = self.cfg.solver.es_patience
            else:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Early Stopping at epoch {self.epoch} \nbest epoch at {self.best_epoch} with loss {self.best_val_loss}")
                    return False
        return True

    def test(self, loader=None, name='test'):
        loader = loader or self.test_loader
        self.stage = 'test'
        tasks = self.tasks
        self.model.eval()
        self._reset_meters(self.test_meters)

        with torch.no_grad():
            for batch in loader:
                _, _, metrics = self.model_inference(batch)
                if not isinstance(metrics, tuple):
                    metrics = [metrics]
                for key_task, metric in zip(tasks, metrics):
                    for key_metric, val in metric.items():
                        self.test_meters[f'{key_task}_{name}_{key_metric}'].update(val)
                    
        self._write_meters(self.test_meters)

    def save_model(self, path):
        path = os.path.join(self.cfg.output.checkpoint_dir, path)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='best_checkpoint.pt'):
        path = os.path.join(self.cfg.output.checkpoint_dir, path)
        self.model.load_state_dict(torch.load(path))

    def compute_metrics(self, pred, y):
        vals = {}
        for metric in self.metrics:
            vals[metric] = getattr(self, metric)(pred, y)
        return vals
    
    def parse_batch_train(self, batch):
        raise NotImplementedError
    
    def parse_batch_test(self, batch):
        raise NotImplementedError
    
    def forward_backward(self, x, y):
        raise NotImplementedError
    
    def model_inference(self, x):
        raise NotImplementedError
    
class MTLTrainer(Trainer):

    def __init__(self, cfg, model, optimizer, criterion, logger=None, metrics=['acc']):
        super(MTLTrainer, self).__init__(cfg, model, optimizer, criterion, logger, metrics = metrics)
        assert len(self.tasks) > 1, 'Specify more than 1 task for MTL'
        
    def forward_backward(self, batch):
        device = self.device
        tasks = self.tasks
        loss_t1, loss_t2 = self.criterion['t1'], self.criterion['t2']
        x, y_t1, y_t2, y = batch
        x, y_t1, y_t2, y = x.to(device), y_t1.to(device), y_t2.to(device), y.to(device)

        self.optimizer.zero_grad()
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        loss1, loss2 = torch.tensor(0), torch.tensor(0)

        if 't1' in tasks:
            loss1 = loss_t1(pred_t1, y_t1)
            
        if 't2' in tasks:
            loss2 = loss_t2(pred_t2, y_t2)

        loss = loss1 + loss2

        loss.backward()
        self.optimizer.step()

        return loss1.item(), loss2.item()
    
    def model_inference(self, batch):
        tasks = self.tasks
        X, y_t1, y_t2, _ = batch
        X, y_t1, y_t2 = X.to(self.device), y_t1.to(self.device), y_t2.to(self.device)
        pred_t1, pred_t2 = self.model(X, tasks=tasks)
        loss1, loss2 = torch.tensor(0), torch.tensor(0)

        if 't1' in tasks:
            loss1 = self.criterion['t1'](pred_t1, y_t1)
            if self.stage == 'test':
                metrics_1 = self.compute_metrics(pred_t1, y_t1)
            
        if 't2' in tasks:
            loss2 = self.criterion['t2'](pred_t2, y_t2)
            if self.stage == 'test':
                metrics_2 = self.compute_metrics(pred_t2, y_t2)
        
        return (pred_t1, pred_t2), (loss1.item(), loss2.item()), (metrics_1, metrics_2)

class STLTrainer(Trainer):
    
    def __init__(self, cfg, criterion, logger=None, metrics=['acc', 'f1', 'wacc', 'eer']):
        super(STLTrainer, self).__init__(cfg, criterion, logger, metrics)

        if isinstance(self.criterion, dict):
            self.tasks = list(self.criterion.keys())
            self.task = self.tasks[0]
        else:
            raise ValueError('Criterion should be a dictionary specify task type as key and loss function as value')
        
        assert len(self.tasks) == 1, 'Number of tasks should be 1 for STL'

        self.criterion = self.criterion[self.tasks[0]]
    
    def parse_batch_train(self, batch):
        x = batch['mel_spec']
        y = batch['label_per_type']
        y_t1 = batch['label_fluent']
        y_t2 = batch['label_ccc']
        return x.to(self.device), y.to(self.device), y_t1.to(self.device), y_t2.to(self.device) 
    
    def parse_batch_test(self, batch):
        x = batch['mel_spec']
        y = batch['label_per_type']
        return x.to(self.device), y.to(self.device)
    
    def forward_backward(self, batch):
        x, y, y_t1, y_t2 = self.parse_batch_train(batch)
        self.optimizer.zero_grad()
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)

        if self.task == 't1':
            loss = self.criterion(pred_t1, y_t1)
        elif self.task == 't2':
            loss = self.criterion(pred_t2, y)
        else:
            raise ValueError('Task not found')
        
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def model_inference(self, batch):
        # X, y_t1, y_t2, y = batch
        # X, y_t1, y_t2, y = X.to(self.device), y_t1.to(self.device), y_t2.to(self.device), y.to(self.device)
        X, y, y_t1, y_t2 = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(X, tasks=self.tasks)
        metrics = {}
        if self.task == 't1':
            loss = self.criterion(pred_t1, y_t1)
            if self.stage == 'test':
                metrics = self.compute_metrics(pred_t1, y_t1)
        else:
            loss = self.criterion(pred_t2, y)
            if self.stage == 'test':
                metrics = self.compute_metrics(pred_t2, y_t2)

        return   (pred_t1, pred_t2), loss.item(), metrics        