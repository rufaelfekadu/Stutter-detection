
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
        self.meters = {}
        self._init_meters()
        
    def _init_meters(self):

        for key in self.criterion.keys():
            self.meters[f'{key}_train_loss'] = AverageMeter(name=f'{key}_train_loss', writer=self.logger)
            self.meters[f'{key}_val_loss'] = AverageMeter(name=f'{key}_val_loss', writer=self.logger)

        self.meters[f'{key}_train_acc'] = AverageMeter(name=f'{key}_train_acc', writer=self.logger)
        self.meters[f'{key}_val_acc'] = AverageMeter(name=f'{key}_val_acc', writer=self.logger)
        self.meters['t1_test_acc'] = AverageMeter(name='test_acc', writer=self.logger)
        self.meters['t2_test_acc'] = AverageMeter(name='test_acc', writer=self.logger)
        self.meters['test_f1'] = AverageMeter(name='test_f1', writer=self.logger)
    
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

            tq_obj = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=True)
            for batch in tq_obj:
                x, y_t1, y_t2, y = batch
                x, y_t1, y_t2, y = x.to(device), y_t1.to(device), y_t2.to(device), y.to(device)

                self.optimizer.zero_grad()
                pred_t1, pred_t2 = self.model(x.squeeze(1), task_type=task_type)
                loss1, loss2 = 0, 0

                if task_type == 'mtl':
                    loss1 = loss_t1(pred_t1, y_t1)
                    loss2 = loss_t2(pred_t2, y_t2)
                    loss = loss1 + loss2
                elif task_type == 't1':
                    loss = loss_t1(pred_t1, y_t1)
                else:
                    loss = loss_t2(pred_t2, y_t2)

                loss.backward()
                self.optimizer.step()

                self.meters['t1_train_loss'].update(loss1.item())
                self.meters['t2_train_loss'].update(loss2.item())
                #  update text of tqdm object with loss values approximated to 2 decimal places
                tq_obj.set_postfix(t1_loss=f'{self.meters["t1_train_loss"].avg:.2f}', t2_loss=f'{self.meters["t2_train_loss"].avg:.2f}')
            
            self.meters['t1_train_loss'].write(epoch)
            self.meters['t2_train_loss'].write(epoch)

            # validation
            
            self.validate()
    
    def validate(self):

        task_type = self.cfg.task_type
        self.model.eval()
        self.meters['t1_val_loss'].reset()
        self.meters['t2_val_loss'].reset()

        loss_t1, loss_t2 = self.criterion['t1'], self.criterion['t2']

        with torch.no_grad():
            for batch in self.val_loader:
                X, y_t1, y_t2, _ = batch
                X, y_t1, y_t2 = X.to(self.device), y_t1.to(self.device), y_t2.to(self.device)
                pred_t1, pred_t2 = self.model(X.squeeze(1), task_type=task_type)
                
                if task_type == 'mtl' or task_type == 't1':
                    loss1 = loss_t1(pred_t1, y_t1)
                    self.meters['t1_val_loss'].update(loss1.item())

                loss2 = loss_t2(pred_t2, y_t2)
                self.meters['t2_val_loss'].update(loss2.item())

        self.meters['t1_val_loss'].write()

    def test(self, model, test_loader):
        task_type = self.cfg.task_type
        model.eval()
        self.meters['t1_test_acc'].reset()
        self.meters['t2_test_acc'].reset()
        self.meters['test_f1'].reset()

        with torch.no_grad():
            for batch in test_loader:
                X, y_t1, y_t2, y = batch
                pred_t1, pred_t2 = model(X.squeeze(1), task_type=task_type)
                
                if task_type == 'mtl' or task_type == 't1':
                    pred_t1 = torch.argmax(pred_t1, dim=1)
                    acc = (pred_t1 == y_t1).sum().item()
                    self.meters['t1_test_acc'].update(acc, X.size(0))

                pred_t2 = torch.argmax(pred_t2, dim=1)
                acc_fluency = (pred_t2 == y).sum().item()
                self.meters['t2_test_acc'].update(acc_fluency, X.size(0))

                if not task_type == 't1':
                    f1 = f1_score(y.item().numpy(), pred_t2.item().numpy(), average='weighted')
                    self.meters['test_f1'].update(f1, X.size(0))
            
            self.meters['t2_test_acc'].write()

        return self.meters['t1_test_acc'].avg, self.meters['t2_test_acc'].avg

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def get_model(self):
        return self.model
    
