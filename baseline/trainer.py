
import torch
from utils import AverageMeter
from tqdm import tqdm
from data import get_dataloaders

class BaseTrainer(object):

    def __init__(self, model, optimizer, criterion, device, logger=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.meters = {}
        self._init_meters()
        
    def _init_meters(self):

        for key in self.criterion.keys():
            self.meters[f'{key}_train_loss'] = AverageMeter(name=f'{key}_train_loss', writer=self.logger)
            self.meters[f'{key}_val_loss'] = AverageMeter(name=f'{key}_val_loss', writer=self.logger)

        self.meters[f'{key}_train_acc'] = AverageMeter(name=f'{key}_train_acc', writer=self.logger)
        self.meters[f'{key}_val_acc'] = AverageMeter(name=f'{key}_val_acc', writer=self.logger)
        self.meters['test_acc'] = AverageMeter(name='test_acc', writer=self.logger)
    
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
        
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(cfg)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            self.model = torch.nn.DataParallel(self.model)

    
    def train(self):

        task_type = self.cfg.task_type
        device = self.device
        loss_t1, loss_t2 = self.criterion['t1'], self.criterion['t2']
        
        for epoch in range(self.cfg.epochs):
            self.model.train()
            self.meters['t1_train_loss'].reset()
            self.meters['t2_train_loss'].reset()

            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=None):
                x, y_t1, y_t2, y = batch
                x, y_t1, y_t2, y = x.to(device), y_t1.to(device), y_t2.to(device), y.to(device)

                self.optimizer.zero_grad()
                pred_t1, pred_t2 = self.model(x, task_type=task_type)
                loss1, loss2 = 0, 0

                if task_type == 'mtl':
                    loss1 = loss_t1(y_t1, pred_t1)
                    loss2 = loss_t2(y_t2, pred_t2)
                    loss = loss1 + loss2
                elif task_type == 't1':
                    loss = loss_t1(y_t1, pred_t1)
                else:
                    loss = loss_t2(y_t2, pred_t2)

                loss.backward()
                self.optimiser.step()

                self.meters['t1_train'].update(loss1.item())
                self.meters['t2_train'].update(loss2.item())

            if self.logger:
                self.meters['t1_train'].write(epoch)
                self.meters['t2_train'].write(epoch)
    
    def test(self, test_loader):
        self.model.eval()
        self.meters['test_acc'].reset()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                self.meters['test_acc'].update((predicted == labels).sum().item() / labels.size(0), labels.size(0))
        if self.logger:
            self.meters['test_acc'].write()
        
        return self.meters['test_acc'].avg

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def get_model(self):
        return self.model
    
