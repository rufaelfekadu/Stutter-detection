
import torch
from utils import AverageMeter
from tqdm import tqdm
from data import get_dataloaders
from sklearn.metrics import f1_score

class BaseTrainer(object):

    def __init__(self, model, optimizer, criterion, device, logger=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.train_meters = {}
        self.test_meters = {}
        self.val_meters = {}
        
    
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    

class Trainer(BaseTrainer):

    def __init__(self, cfg, model, optimizer, criterion, device, logger=None):
        super(Trainer, self).__init__(model, optimizer, criterion, device, logger)
        self.cfg = cfg
        self._init_meters()
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(cfg)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            self.model = torch.nn.DataParallel(self.model)

    def _init_meters(self):

        for key in self.cfg.tasks:
            self.train_meters[f'{key}_train_loss'] = AverageMeter(name=f'{key}_train_loss', writer=self.logger)
            self.val_meters[f'{key}_val_loss'] = AverageMeter(name=f'{key}_val_loss', writer=self.logger)
            self.test_meters[f'{key}_test_acc'] = AverageMeter(name='test_acc', writer=self.logger)


    def _reset_meters(self, meters):
        for meter in meters.values():
            meter.reset()
    
    def _write_meters(self, meters):
        for meter in meters.values():
            meter.write()

    def train(self):

        for epoch in range(self.cfg.epochs):
            self.model.train()
            self._reset_meters(self.train_meters)

            tq_obj = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=True)
            for batch in tq_obj:
                losses = self.forward_backward(batch)
                for key, loss in zip(self.cfg.tasks, losses):
                    self.train_meters[f'{key}_train_loss'].update(loss)
                tq_obj.set_postfix(t1_loss=f'{self.train_meters["t1_train_loss"].avg:.2f}', t2_loss=f'{self.train_meters["t2_train_loss"].avg:.2f}')             
            for meter in self.meters.values():
                meter.write(epoch)

            # validation
            self.validate()
    
    def validate(self):

        tasks = self.cfg.tasks
        self.model.eval()

        self._reset_meters(self.val_meters)

        loss_t1, loss_t2 = self.criterion['t1'], self.criterion['t2']

        with torch.no_grad():
            for batch in self.val_loader:
                X, y_t1, y_t2, _ = batch
                X, y_t1, y_t2 = X.to(self.device), y_t1.to(self.device), y_t2.to(self.device)
                pred_t1, pred_t2 = self.model(X, tasks=tasks)
                
                if 't1' in tasks:
                    loss1 = loss_t1(pred_t1, y_t1)
                    self.val_meters['t1_val_loss'].update(loss1.item())

                if 't2' in tasks:
                    loss2 = loss_t2(pred_t2, y_t2)
                    self.val_meters['t2_val_loss'].update(loss2.item())

        self._write_meters(self.val_meters)

    def test(self, test_loader=None):

        test_loader = test_loader if test_loader else self.test_loader

        tasks = self.cfg.tasks
        self.model.eval()
        self._reset_meters(self.test_meters)

        with torch.no_grad():
            for batch in test_loader:
                X, y_t1, y_t2, y = batch
                X, y_t1, y_t2, y = X.to(self.device), y_t1.to(self.device), y_t2.to(self.device), y.to(self.device)
                pred_t1, pred_t2 = self.model(X, tasks=tasks)
                
                if 't1' in tasks:
                    pred_t1 = torch.argmax(pred_t1, dim=1)
                    acc = (pred_t1 == y_t1).sum().item()
                    self.meters['t1_test_acc'].update(acc, X.size(0))
                if 't2' in tasks:
                    pred_t2 = torch.argmax(pred_t2, dim=1)
                    acc_fluency = (pred_t2 == y).sum().item()
                    self.meters['t2_test_acc'].update(acc_fluency, X.size(0))

            self.meters['t1_test_acc'].write()
            self.meters['t2_test_acc'].write()

        return self.meters['t1_test_acc'].avg, self.meters['t2_test_acc'].avg

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def get_model(self):
        return self.model

    def forward_backward(self, x, y):
        raise NotImplementedError
    
    def model_inference(self, x):
        raise NotImplementedError
    

class SimpleTrainer(Trainer):

    def __init__(self, cfg, model, optimizer, criterion, device, logger=None):
        super(SimpleTrainer, self).__init__(cfg, model, optimizer, criterion, device, logger)
        self.tasks = cfg.tasks

        assert len(self.tasks) == len(self.criterion), 'Number of tasks and criterion should be same'
        
    
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
    
    def model_inference(self, x):
        device = self.device
        x = x.to(device)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)

        return pred_t1, pred_t2
