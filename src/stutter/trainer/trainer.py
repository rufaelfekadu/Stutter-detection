import os
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
from tqdm import tqdm

from stutter.utils.annotation import LabelMap
from stutter.data import get_dataloaders
from stutter.data.hf_data import VivitVideoData
from stutter.models import build_model
from stutter.models.vivit import VivitForStutterClassification
from stutter.utils.meters import LossMeter, AverageMeter
from stutter.utils.metrics import f1_score_per_class, f1_score_, weighted_accuracy, multilabel_EER, binary_acc, binary_f1, iou_metric,compute_video_classification_metrics
from stutter.utils.loss import build_loss, CCCLoss
from stutter.utils.data import deconstruct_labels
from stutter.utils.misc import plot_sample
from stutter.utils.data import make_video_dataframe, extract_mfcc
import librosa
import matplotlib.pyplot as plt
from transformers import AutoModelForAudioClassification, TrainingArguments, VivitForVideoClassification, VivitConfig, AdamW
from transformers import Trainer as HuggingFaceTrainer
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_metric, Dataset, concatenate_datasets
from sklearn.metrics import f1_score
acc = load_metric("accuracy")
f1 = load_metric("f1")

metric_bank={
    'acc': lambda pred, y: (torch.argmax(pred, dim=1) == y).sum().item() / y.size(0),
    'f1': f1_score_per_class,
    'f1_macro': f1_score_,
    'wacc': weighted_accuracy,
    'eer': multilabel_EER,
    'binary_acc': binary_acc,
    'binary_f1': binary_f1,
    'iou': iou_metric,
    'video_metrics': compute_video_classification_metrics,
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
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()

        # build model
        self.model, self.optimizer, self.scheduler, self.criterion = self.get_model()
        # breakpoint()
        self.model = self.model.to(self.device)

        self.best_val_loss = float('inf')
        self.best_f1 = 0
        self.patience = cfg.solver.es_patience
        print(self.model)

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

    def get_dataloaders(self):
        return get_dataloaders(self.cfg)

    def get_model(self):
        return *build_model(self.cfg), build_loss(self.cfg)

    def _init_meters(self):
        for key in self.tasks:
            self.train_meters[f'{key}'] = LossMeter(name=f'{key}_train_loss', writer=self.logger)
            self.val_meters[f'{key}'] = LossMeter(name=f'{key}_val_loss', writer=self.logger)
            self.test_meters[f'{key}'] = LossMeter(name=f'{key}_test_loss', writer=self.logger)
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
                    self._update_meter(self.train_meters, f'{key}', loss)
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
        self.model.eval()
        self._reset_meters(self.val_meters)

        with torch.no_grad():
            total_loss = 0
            for batch in self.val_loader:
                losses = self.val_step(batch)
                total_loss += sum(losses.values())
                for key, loss in losses.items():
                    self._update_meter(self.val_meters, f'{key}', loss)

            # log the learning rate
            self.val_meters['lr'].update(self.scheduler.get_last_lr()[0])
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss/len(self.val_loader))
            else:
                self.scheduler.step()

            self._write_meters(self.val_meters)

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
        
        # self.after_validation()
        self.model.train()
        return False

    def after_validation(self):
        pass

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
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        # log the confusion matrix
        self.logger.add_figure(f'confusion_matrix', disp.figure_)
        # for i in range(self.num_classes):
        #     cm = confusion_matrix(labels[:,i], preds[:,i])
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        #     # log the confusion matrix
        #     self.logger.add_figure(f'confusion_matrix_{i}', disp.figure_)

class SedTrainer2(Trainer):

    def __init__(self, cfg, logger=None, metrics=['acc']):
        super(SedTrainer2, self).__init__(cfg, logger, metrics = metrics)
        self.criterion = self.criterion['t2']
        self.test_mfcc = []
        self.test_preds = []
        self.test_labels = []
        self.test_fnames = []
        self.test_encoder_outputs = []
        self.val_mfcc = []
        self.val_outputs = []
        self.val_preds = []
        
        self.num_classes = 5

        # print the number of trainable parameters

    def hook_fn(self, module, input, output):
        self.test_encoder_outputs.append(output)

    def parse_batch_train(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def parse_batch_test(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        fname = batch['file_path']
        return x.to(self.device), y.to(self.device), fname
    
    def train_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return{
            't2': loss.item()
        }
    
    def val_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        self.val_outputs.append(y)
        self.val_preds.append(pred)
        self.val_mfcc.append(x)
        return{
            't2': loss.item()
        }

    def test(self, loader=None, name='test'):
        # generate the reference txt_files
        loader = loader or self.test_loader
        self.stage = 'test'
        self.model.eval()
        self._reset_meters(self.test_meters)
        handle = self.model.encoder.register_forward_hook(self.hook_fn)
        with torch.no_grad():
            for batch in loader:

                x, y, fname = self.parse_batch_test(batch)
                preds = self.model(x.squeeze(1).squeeze(1), output_attentions=True)
                test_loss = self.criterion(preds, y)
                self.test_mfcc.append(x)
                self.test_preds.append(preds)
                self.test_fnames.append(fname)
                self.test_labels.append(y)
                self._update_meter(self.test_meters, f't2_test_loss', test_loss.item())
        
        self._write_meters(self.test_meters)
        handle.remove()
        self.after_test()

    def after_test(self):
        mfcc = torch.cat(self.test_mfcc, axis=0)
        preds = torch.concat(self.test_preds, axis=0) # (N, 22, 2+num_classes)
        fnames = self.test_fnames[0]
        y = torch.cat(self.test_labels, axis=0)
        preds_mask = (torch.sigmoid(preds) >= 0.5).int()

        # write the predictions to a file
        for i,fname in tqdm(enumerate(fnames), desc='Writing predictions', total=len(fnames)):
            pred_fname = fname.replace('_ref.txt', '_pred.txt')
            pred = preds_mask[i,:,:]
            with open(pred_fname, 'w') as f:
                events = deconstruct_labels(pred, clip_duration=30, sr=16000, smooth=3)
                for event in events:
                    f.write(f'{event[0]},{event[1]},{event[2]}\n')
        
        # plot some of the encoder outputs
        output = torch.concat(self.test_encoder_outputs, axis=0)
        preds = torch.nn.functional.interpolate(preds.unsqueeze(0), size=(output.size(1), output.size(2)), mode='nearest').squeeze(0)
        preds_mask = torch.nn.functional.interpolate(preds_mask.float().unsqueeze(0), size=(output.size(1), output.size(2)), mode='nearest').squeeze(0)
        resized_y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(output.size(1), output.size(2)), mode='nearest').squeeze(0)
        for i in range(5):
            fig, ax = plt.subplots(5, figsize=(20,10))
            normalized_output = (output[i] - output[i].min()) / (output[i].max() - output[i].min())
            # normalized_mfcc = (mfcc[i] - mfcc[i].min()) / (mfcc[i].max() - mfcc[i].min())

            # ax[0].imshow(normalized_mfcc.cpu().numpy().T, aspect='auto', cmap='inferno')
            # ax[0].set_title('MFCC Features')

            ax[1].imshow(output[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[1].set_title('wav2vec2 Features')

            ax[2].imshow(preds[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[2].set_title('logits')
        
            ax[3].imshow(preds_mask[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[3].set_title('Predictions')

            ax[4].imshow(resized_y[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[4].set_title('Ground Truth')
            
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])

            # Add a single color bar
            cbar = fig.colorbar(ax[1].images[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            plt.savefig(f'outputs/encoder_output_{i}.png')
            self.logger.add_figure(f'test/encoder_output_{i}', fig)
                
class SedTrainer(Trainer):
    def __init__(self, cfg, logger=None, metrics=['acc']):
        super(SedTrainer, self).__init__(cfg, logger, metrics = metrics)
        self.criterion = self.criterion['t2']
        self.test_preds = []
        self.test_labels = []
        self.test_fnames = []
        self.num_classes = 5

    def parse_batch_train(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def parse_batch_test(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        fname = batch['file_path']
        return x.to(self.device), y.to(self.device), fname
    
    def train_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return{
            't2': loss.item()
        }
    
    def val_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        return{
            't2': loss.item()
        }
    
    def test(self, loader=None, name='test'):
        # generate the reference txt_files
        loader = loader or self.test_loader
        self.stage = 'test'
        self.model.eval()
        self._reset_meters(self.test_meters)

        with torch.no_grad():
            for batch in loader:
                x, y, fname = self.parse_batch_test(batch)
                preds = self.model(x.squeeze(1))
                test_loss = self.criterion(preds, y)
                self.test_preds.append(preds)
                self.test_fnames.append(fname)
                self._update_meter(self.test_meters, f't2_test_loss', test_loss.item())
        
        self._write_meters(self.test_meters)
        self.after_test()

    def after_test(self):
        label_map = LabelMap()
        preds = torch.concat(self.test_preds, axis=0) # (N, 22, 2+num_classes)
        fnames = self.test_fnames[0]
        preds[:,:,2:] = (torch.sigmoid(preds[:,:,2:]) >= 0.5).int()
        # write the predictions to a file
        for i,fname in enumerate(fnames):
            pred_fname = fname.replace('.txt', '_pred.txt')
            pred = preds[i,:,:]
            with open(pred_fname, 'w') as f:
                for j in range(pred.size(0)):
                    for k, l in enumerate(pred[j,2:]):
                        if l == 1 and pred[j,1] > 0:
                            start_l = pred[j,0].item()
                            end_l = pred[j,1].item()
                            f.write(f'{round(start_l,2)},{round(end_l,2)},{label_map.description[label_map.core[k]]}\n')
             
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
        
    
VIDEO_STUTTER_CLASSES = ['V', 'FG', 'HM']
STUTTER_CLASSES = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM']

class VivitForStutterTrainer(Trainer):
    
    def get_model(self):
        model = VivitForStutterClassification(self.cfg)
        optim = torch.optim.SGD(model.parameters(), lr=self.cfg.solver.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=5e-6)
        
        return model, optim, scheduler, None
    
    def get_dataloaders(self):
        # TODO: Fix this function
        df = load_from_disk(f"{self.cfg.data.label_path}train").remove_columns(['input_values', 'attention_mask'])
        test_dataset = load_from_disk("outputs/fluencybank/dataset/stutter_hf/label_split/Gold_multimodal_test").remove_columns(['input_values', 'attention_mask'])
        
        label = 'labels'
        
        if self.cfg.tasks[0] == 't1':
            if self.cfg.data.annotation != 'any':
                ann_index = STUTTER_CLASSES.index(self.cfg.data.annotation)
                print(f"Label is {self.cfg.data.annotation} at index {ann_index}")
                df = df.map(lambda x: {'labels': x['labels'][ann_index]}, num_proc=8)
                df = df.filter(lambda example: len(example['pixel_values']) == 10, num_proc=8, writer_batch_size=100)
                test_dataset = test_dataset.map(lambda x: {'labels': x['labels'][ann_index]}, num_proc=8)
            else:
                df = df.map(lambda x: {'labels': max(x['labels'][5:])}, num_proc=8)
                test_dataset = test_dataset.map(lambda x: {'labels': max(x['labels'][5:])}, num_proc=8)   
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
            undersampled_dataset = undersampled_dataset.shuffle(seed=42)
            class_counts = undersampled_dataset['labels'].count(0), undersampled_dataset['labels'].count(1)
            print(f"Undersampled Class counts are {class_counts}")
            df = undersampled_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
            
        else:
            df = df.train_test_split(test_size=0.1, seed=42, shuffle=True)

        dataset = df['train']
        val_dataset = df['test']
        dataset.set_format(type='torch', columns=[ 'pixel_values',label])
        val_dataset.set_format(type='torch', columns=[ 'pixel_values',label])
        test_dataset.set_format(type='torch', columns=[ 'pixel_values', label])
             
        train_loder = DataLoader(dataset, batch_size=self.cfg.solver.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.solver.batch_size, shuffle=False)
        del df

        return train_loder, val_loader, test_loader
    
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
        for i,classes in enumerate(VIDEO_STUTTER_CLASSES):
            metrics[classes] = f1_score(y_true[:,i], y_pred[:,i])
        return {'accuracy': (y_pred==y_true.bool()).float().mean().item(), 
                'f1_weighted': f1_weighted, 'f1_macro': f1_macro, 'f1_any':f1_any, **metrics}
    
    def parse_batch_train(self, batch):
        image = batch['pixel_values'].to(self.device)
        if self.cfg.tasks[0] == 't1':
            # if self.cfg.data.annotation == 'stutter':
            #     y = torch.max(batch['labels'][:,5:], dim=-1)[0].to(self.device)
            # else:
            #     i = STUTTER_CLASSES.index(self.cfg.data.annotation)
            y = batch['labels'].to(self.device).float()
        else:
            y = batch['labels'][:,5:].to(self.device)
        return image, y
    
    def train_step(self, batch):
        image, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, labels=y)
        loss.backward()
        self.optimizer.step()
        return {
                'loss': loss.item()
        }
    
    def val_step(self, batch):
        image, y = self.parse_batch_train(batch)
        loss, logits = self.model(pixel_values=image, labels=y)
        if self.cfg.tasks[0] == 't1':
            metrics = self.compute_t1_metrics(logits, y.cpu())
            metrics['loss'] = loss.item()
        else:
            metrics = self.compute_t2_metrics(logits.cpu(), y.cpu())
            metrics['loss'] = loss.item()
        return metrics

    def test_step(self, batch):
        image, y = self.parse_batch_train(batch)
        
        loss, logits = self.model(pixel_values=image, labels=y)
        # if self.cfg.tasks[0] == 't1':
        #     metrics = self.compute_t1_metrics(logits.cpu(), y.cpu())
        #     metrics['loss'] = loss.item()
        # else:
        #     metrics = self.compute_t2_metrics(logits.cpu(), y.cpu())
        #     metrics['loss'] = loss.item()
        # print(*metrics)
        return loss, logits, y
    
    def _init_meters(self):
        self.val_meters['accuracy'] = AverageMeter('val_accuracy', writer=self.logger)
        self.val_meters['loss'] = AverageMeter('val_loss', writer=self.logger)
        self.val_meters['lr'] = AverageMeter('lr', writer=self.logger)
        
        if self.cfg.tasks[0] == 't1':
            self.val_meters[f'f1_{self.cfg.data.annotation}'] = AverageMeter(f'val_f1_{self.cfg.data.annotation}', writer=self.logger)
        else:
            self.val_meters['f1_weighted'] = AverageMeter('val_f1_weighted', writer=self.logger)
            self.val_meters['f1_macro'] = AverageMeter('val_f1_macro', writer=self.logger)
            self.val_meters['f1_any'] = AverageMeter('val_f1_any', writer=self.logger)
            
        
            for classes in VIDEO_STUTTER_CLASSES:
                self.val_meters[classes] = AverageMeter(f'val_{classes}_f1', writer=self.logger)
            
        self.train_meters['loss'] = AverageMeter('train_loss', writer=self.logger)

        self.test_meters['accuracy'] = AverageMeter('test_accuracy', writer=self.logger)
        self.test_meters['loss'] = AverageMeter('test_loss', writer=self.logger)
        if self.cfg.tasks[0] == 't1':
            self.test_meters[f'f1_{self.cfg.data.annotation}'] = AverageMeter(f'test_f1_{self.cfg.data.annotation}', writer=self.logger)
        else:
            self.test_meters['f1_weighted'] = AverageMeter('test_f1_weighted', writer=self.logger)
            self.test_meters['f1_macro'] = AverageMeter('test_f1_macro', writer=self.logger)
            self.test_meters['f1_any'] = AverageMeter('test_f1_any', writer=self.logger)
            for classes in VIDEO_STUTTER_CLASSES:
                self.test_meters[classes] = AverageMeter(f'test_{classes}_f1', writer=self.logger)

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
            # total_f1 = losses[f'f1_{self.cfg.data.annotation}']
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