import os
import torch
from tqdm import tqdm

from stutter.data import get_dataloaders
from stutter.models import build_model
from stutter.utils import AverageMeter, metric_registery, make_video_dataframe, extract_mfcc

from stutter.loss import build_loss

from torch.utils.data import DataLoader
from datasets import  Dataset

# metric_bank={
#     'acc': lambda pred, y: (torch.argmax(pred, dim=1) == y).sum().item() / y.size(0),
#     'f1': f1_score_per_class,
#     'f1_macro': f1_score_,
#     'wacc': weighted_accuracy,
#     'eer': multilabel_EER,
#     'binary_acc': binary_acc,
#     'binary_f1': binary_f1,
#     'iou': iou_metric,
#     'video_metrics': compute_video_classification_metrics,
# }


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
        self.global_step = 0
        self.best_epoch = 0

    def _init_meter(self, meters, key, set='train'):
        if key not in meters:
            meters[key] = AverageMeter(name=f'{set}_{key}', writer=self.logger)

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
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    
class Trainer(BaseTrainer):

    def __init__(self, cfg, logger=None, resume_from_checkpoint=False):
        super(Trainer, self).__init__(cfg.solver.device, logger)


        self.cfg = cfg
        self.tasks = cfg.tasks

        self.validate_on = cfg.solver.validate_on
        self.validate_mode = cfg.solver.validate_mode # ['min', 'max']
        self.comparator = lambda curr, best: curr < best if self.validate_mode == 'min' else curr > best

        self.metrics = cfg.solver.metrics

        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()

        # build model
        self.model, self.optimizer, self.scheduler, self.criterion = self.get_model()
        self.model = self.model.to(self.device)

        self.best_val = float('inf') if self.validate_mode == 'min' else float('-inf')
        self.patience = cfg.solver.es_patience

        # if len(cfg.solver.device) > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     self.model = torch.nn.DataParallel(self.model)

        print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'Number of total parameters: {sum(p.numel() for p in self.model.parameters())}')

        # assert eval_steps % log_steps == 0, 'eval_steps should be divisible by log_steps'
        if self.cfg.solver.eval_steps > len(self.train_loader):
            self.cfg.solver.eval_steps = len(self.train_loader)
            print(f'eval_steps is greater than the number of batches in the train_loader. Setting eval_steps to {len(self.train_loader)}')

        if self.cfg.solver.log_steps > len(self.train_loader):
            self.cfg.solver.log_steps = len(self.train_loader)
            print(f'log_steps is greater than the number of batches in the train_loader. Setting log_steps to {len(self.train_loader)}')

        if resume_from_checkpoint:
            self.load_model()

        self.sanity_check()

        assert self.validate_on in self.val_meters, f'{self.validate_on} not in validation meters'

    def sanity_check(self):
        print('Running Sanity Check')
        # get one element from train_loader
        for batch in self.train_loader:
            train_outputs = self.train_step(batch)
            for key, val in train_outputs.items():
                self._init_meter(self.train_meters, key, set='train')
            break

        # get one element from val_loader
        for batch in self.val_loader:
            val_outputs = self.val_step(batch)
            for key, val in val_outputs.items():
                self._init_meter(self.val_meters, key, set='val')
            break

        for batch in self.test_loader:
            test_outputs = self.test_step(batch)
            for key, val in test_outputs.items():
                self._init_meter(self.test_meters, key, set='test')
            break

        self._init_meter(self.val_meters, 'lr')

        print('Sanity Check Passed')

    def get_dataloaders(self):
        return get_dataloaders(self.cfg)

    def get_model(self):
        return *build_model(self.cfg), build_loss(self.cfg)

    def train(self):
        self.stage = 'train'
        es = False
        for epoch in range(self.cfg.solver.epochs):

            self.model.train()
            self._reset_meters(self.train_meters)
            tq_obj = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=True)
            for i, batch in enumerate(tq_obj):
                self.optimizer.zero_grad()
                losses = self.train_step(batch)
                for key, loss in losses.items():
                    self._update_meter(self.train_meters, f'{key}', loss)
                    
                tq_obj.set_postfix({key: round(meter.avg[0],2) for key, meter in self.train_meters.items()})

                self.global_step += 1

                if self.global_step % self.cfg.solver.log_steps == 0:
                    self._write_meters(self.train_meters)

                if self.global_step % self.cfg.solver.eval_steps == 0:
                    es = self.validate()
                    if es: break


            if es: break

            self.epoch += 1
    
    def validate(self):
        self.stage = 'val'
        self.model.eval()
        self._reset_meters(self.val_meters)

        with torch.no_grad():
            for batch in self.val_loader:
                losses = self.val_step(batch)
                for key, loss in losses.items():
                    self._update_meter(self.val_meters, f'{key}', loss)

            self.val_meters['lr'].update(self.scheduler.get_last_lr()[0])

            self._write_meters(self.val_meters)

            # get the metric to validate on
            curr_loss = sum(self.val_meters[self.validate_on].avg)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(curr_loss)
            else:
                self.scheduler.step()

            if self.comparator(curr_loss, self.best_val):
                self.best_val = curr_loss
                self.best_epoch = self.epoch
                self.save_model('best_checkpoint.pt')
                self.patience = self.cfg.solver.es_patience
            else:
                self.patience -= 1
                if self.patience == 0:
                    print(f"Early Stopping at epoch {self.epoch} \nbest epoch at {self.best_epoch} with loss {self.best_val}")
                    return True
        
        # self.after_validation()
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
                for key, val in metrics.items():
                    self._update_meter(self.test_meters, f'{key}', val)
        self._write_meters(self.test_meters)
        self.after_test()
        
    def after_test(self):
        pass
    
    def save_model(self, path):
        path = os.path.join(self.cfg.output.checkpoint_dir, path)
        to_save = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_epoch': self.best_epoch,
            'best_val': self.best_val
        }
        torch.save(to_save, path)

    def load_model(self, path='best_checkpoint.pt'):
        path = os.path.join(self.cfg.output.checkpoint_dir, path)
        try:
            print(f'Loading model from {path}')
            state_dict = torch.load(path)
        except:
            print('No checkpoint found')
            return
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.epoch = state_dict['epoch']
        self.best_epoch = state_dict['best_epoch']
        self.best_val= state_dict['best_val']
        
    def compute_metrics(self, pred, y):
        vals = {}
        for metric in self.metrics:
            vals[metric] = metric_registery[metric](pred, y)
        return vals
    
    def parse_batch_train(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def parse_batch_test(self, batch):
        raise NotImplementedError
    
    def train_step(self, batch):
        raise NotImplementedError
    
    def val_step(self, batch):
        raise NotImplementedError
    
    def test_step(self, batch):
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
    
    def get_dataloaders(self):
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True, nb_workers=8) 
        PRIMARY_EVENT = ['SR', 'ISR', 'MUR', 'P', 'B'] #, "primary_event"
        
        train_df = make_video_dataframe(self.cfg.data.label_path,self.cfg.data.annotator, self.cfg.data.root, True, extension=".wav")
        train_df['labels'] = train_df[PRIMARY_EVENT].apply(lambda x: x.values, axis=1)
        train_df = train_df[["file_name", "labels"]]

        train_df = Dataset.from_pandas(train_df).train_test_split(test_size=0.1, seed=42, shuffle=True)
        train_df = train_df.map(lambda x: {'mfcc': extract_mfcc(x["file_name"])}, num_proc=8)
        
        test_df = make_video_dataframe(self.cfg.data.label_path,"Gold", self.cfg.data.root, True,split="test", extension=".wav")
        test_df['labels'] = test_df[PRIMARY_EVENT].apply(lambda x: x.values, axis=1)
        test_df = test_df[["file_name", "labels"]]

        test_df = Dataset.from_pandas(test_df)
        test_df = test_df.map(lambda x: {'mfcc': extract_mfcc(x["file_name"])}, num_proc=8)
        # breakpoint()cont
        train_df['train'].set_format(type='torch', columns=[ 'mfcc', 'labels'])
        train_df['test'].set_format(type='torch', columns=[ 'mfcc', 'labels'])
        test_df.set_format(type='torch', columns=[ 'mfcc', 'labels'])
        train_loader = DataLoader(train_df['train'], batch_size=self.cfg.solver.batch_size, shuffle=True)
        val_loader = DataLoader(train_df['test'], batch_size=self.cfg.solver.batch_size, shuffle=False)
        test_loader = DataLoader(test_df, batch_size=self.cfg.solver.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def parse_batch_train(self, batch):
        # x = torch.cat([batch['mel_spec'], batch['f0']], dim=1)
        x = batch['mfcc']
        y = batch['labels']
        return x.to(self.device), y.to(self.device) 
    
    def train_step(self, batch):

        x, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        loss1, loss2 = torch.tensor(0), torch.tensor(0)

        if 't1' in self.tasks:
            # y_t1 = (y>=2).float()
            # y_t1 = (torch.sum(y[:, :-1], dim=1)>0).long()
            loss1 = self.criterion['t1'](pred_t1.squeeze(), y[:,0])

        if 't2' in self.tasks:
            # if  not isinstance(self.criterion['t2'], CCCLoss):
            #     y_t2 = (y>=2).float()
            loss2 = self.criterion['t2'](pred_t2, y)

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
            # y_t1 = (y>=2).float()
            # y_t1 = (torch.sum(y[:, :self.num_classes-1], dim=1)>0).long()
            loss1 = self.criterion['t1'](pred_t1.squeeze(), y[:,0])
        if 't2' in self.tasks:
            # if  not isinstance(self.criterion['t2'], CCCLoss):
            #     y_t2 = (y>=2).float()
            loss2 = self.criterion['t2'](pred_t2, y)
        return{
            't1': loss1.item(),
            't2': loss2.item()
        }
    
    def test_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        metrics_1, metrics_2 = {}, {}
        if 't1' in self.tasks:
            # y_t1 = (y>=2).int()
            # y_t1 = (torch.sum(y[:, :self.num_classes-1], dim=1)>0).int()
            # pred_t1 = torch.argmax(pred_t1, dim=1)
            
            pred_t1 = (torch.sigmoid(pred_t1)>=0.5).int()
            metrics_1 = self.compute_metrics(pred_t1, y[:,0])
            self.test_preds.append(pred_t1.cpu().numpy())
            self.test_labels.append(y[:,0].cpu().numpy())

        if 't2' in self.tasks:
            # y_t2 = (y>=2).int()
            metrics_2 = self.compute_metrics(pred_t2, y)
            self.test_preds.append(pred_t2.cpu().numpy())
            self.test_labels.append(y.cpu().numpy())

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
    
