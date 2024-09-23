import torch
import numpy as np

from transformers import Trainer as HuggingFaceTrainer
from transformers import TrainingArguments, AdamW, AutoModelForAudioClassification

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from stutter.trainer import Trainer
from stutter.utils.annotation import LabelMap
from stutter.config import cfg

class ClassificationTrainer(Trainer):
    
    def __init__(self, cfg, **kwargs):

        self.train_preds_t2 = []
        self.train_preds_t1 = []
        self.train_labels = []
        self.test_preds_t1 = []
        self.test_preds_t2 = []
        self.test_labels = []
        self.test_fnames = []
        self.num_classes = 5
        self.label_columns = ['SR','ISR','MUR','P','B', 'V', 'FG', 'HM','T', 'any']
        self.annotation = self.label_columns.index(cfg.data.annotation)

        super(ClassificationTrainer, self).__init__(cfg, **kwargs)
        self.weights = {
            'SR': [0.95],
            'ISR': [0.8],
            'MUR': [0.97],
            'P': [0.8],
            'B': [0.65],
            'V': [0.8],
            'FG': [0.6],
            'HM': [0.8],
            'any': [0.8],
            'T': [0.3, 0.5, 0.3, 1.0],
        }
        self.criterion['t1'].weights = self.weights[cfg.data.annotation]

    
    def parse_batch_train(self, batch):
        x = batch['audio_features']
        y = batch['label'][:,self.annotation]
        return x.to(self.device), y.to(self.device)
    
    def train_step(self, batch):

        x, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        loss_1, loss_2 = torch.tensor(0), torch.tensor(0)
        if 't1' in self.tasks:
            loss_1 = self.criterion['t1'](pred_t1.squeeze(), y)
        if 't2' in self.tasks:
            loss_2 = self.criterion['t2'](pred_t2, y)
        loss = loss_1 + loss_2
        loss.backward()
        self.optimizer.step()

        return {
            't1_loss': loss_1.item(),
            't2_loss': loss_2.item()
        }
    
    def val_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        loss_1, loss_2 = torch.tensor(0), torch.tensor(0)
        if 't1' in self.tasks:
            loss_1 = self.criterion['t1'](pred_t1.squeeze(), y) 
            metrics_1 = self.compute_metrics(pred_t1.squeeze(), y)
        if 't2' in self.tasks:
            loss_2 = self.criterion['t2'](pred_t2, y)
        return {
            't1_loss': loss_1.item(),
            't2_loss': loss_2.item(),
            'total_loss': loss_1.item() + loss_2.item(),
            **metrics_1
        }
    
    def test_step(self, batch):
        x, y= self.parse_batch_train(batch)
        pred_t1, pred_t2 = self.model(x, tasks=self.tasks)
        # metrics_1, metrics_2 = {}, {}
        if 't1' in self.tasks:
            metrics_1 = self.criterion['t1'](pred_t1.squeeze(), y)
            self.test_preds_t1.append(torch.round(pred_t1.squeeze()).detach().cpu().numpy())
        if 't2' in self.tasks:
            metrics_2 = self.compute_metrics(pred_t2, y)

        self.test_labels.append(y.detach().cpu().numpy())

        # merge the metrics
        # metrics = {**metrics_1, **metrics_2}

        return {
            'loss': metrics_1.item()
        }
    
    def after_test(self):
        self.test_preds_t1 = np.concatenate(self.test_preds_t1)
        self.test_labels = np.concatenate(self.test_labels)
        fig = self.plot_confusion_matrix(self.test_labels, self.test_preds_t1, title='Confusion matrix')
        self.logger.add_figure('Confusion matrix', fig)

    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = title + ' normalized'
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=cmap, ax=ax)
        plt.title(title)
        return fig
    

class Wave2vecTrainer(HuggingFaceTrainer):

    def __init__(self, cfg):
        
        model = AutoModelForAudioClassification.from_pretrained(cfg.model.name)
        training_args = TrainingArguments(
            output_dir=cfg.output.save_dir,
            num_train_epochs=cfg.solver.epochs,
            per_device_train_batch_size=cfg.solver.batch_size,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=10,
            learning_rate=5e-05,
            weight_decay=0.01,
            logging_dir=cfg.output.log_dir,
            logging_steps=cfg.solver.log_steps,
            seed=cfg.seed,
            evaluation_strategy="epoch",
            warmup_steps=int(0.1 * 20),
            optim="adamw_torch",
            lr_scheduler_type="linear",
            fp16=True,
            metric_for_best_model="f1",
            load_best_model_at_end=True,
            report_to='wandb',
            run_name=f"{cfg.data.split_strategy}_{cfg.data.annotator}"
        )
        
        train_dataset = None
        eval_dataset = None
        optimizer = AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
        super(Wave2vecTrainer, self).__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, None),
            compute_metrics=metric_bank['acc']
        )
        super(Wave2vecTrainer, self).__init__()

if __name__ == "__main__":
    # test the classification trainer
    cfg.data.cache_dir = 'outputs/fluencybank/fluencybank.pt'
    cfg.data.name = 'fluencybank'
    cfg.model.name = 'convlstm'
    cfg.tasks = ['t2']
    cfg.solver.losses = ['bce']
    cfg.model.output_size = 5
    cfg.loss.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.metrics = ['f1']

    trainer = ClassificationTrainer(cfg, validate_on='t2_loss')


