
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_curve, balanced_accuracy_score, multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class AverageMeter(object):
    def __init__(self, name=None, writer=None):
        self._writer = writer
        self._name = name
        self.reset()

    def reset(self):
        self.val = [0]
        self.avg = [0]
        self.sum = [0]
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        if isinstance(val, list):
            # Extend the lists if they are shorter than the incoming list
            length_difference = len(val) - len(self.sum)
            if length_difference > 0:
                self.sum.extend([0] * length_difference)
                self.avg.extend([0] * length_difference)
                self.val.extend([0] * length_difference)

            for i, v in enumerate(val):
                self.sum[i] += v
                self.avg[i] = self.sum[i] / self.count
            self.val = val
        else:
            if not isinstance(self.sum, list):
                self.sum = [self.sum]
                self.avg = [self.avg]
                self.val = [self.val]
            self.sum[0] += val
            self.avg[0] = self.sum[0] / self.count
            self.val[0] = val

    def write(self, epoch=0):
        if self.count != 0:
            if isinstance(self.avg, list):
                for i, val in enumerate(self.avg):
                    try:
                        self._writer.add_scalar(f'{self._name}_{i}', val, epoch)
                    except Exception as e:
                        print(e)
            else:
                try:
                    self._writer.add_scalar(self._name, self.avg, epoch)
                except Exception as e:
                    print(e)

class LossMeter(object):
    def __init__(self, name=None, writer=None):
        self.reset()
        self._writer = writer
        self._name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, list):
            val = sum(val) / len(val)
            # log each value in the list
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def write(self, epoch=0):
        #  if self.avg is list then log each value
        if self.count !=0:
            if isinstance(self.avg, list):
                for i, val in enumerate(self.avg):
                    try:
                        self._writer.add_scalar(f'{self._name}_{i}', val, epoch)
                    except Exception as e:
                        print(e)
            else:
                try:
                    self._writer.add_scalar(self._name, self.avg, epoch)
                except Exception as e:
                    print(e)

def EER(predictions, labels):

    if  isinstance(predictions, torch.Tensor):
        predictions = (torch.sigmoid(predictions)>0.5).float()
        predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return eer, eer_threshold

def multilabel_EER(predictions, labels):

    predictions = (torch.sigmoid(predictions) > 0.5).float()
    predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

    n_labels = predictions.shape[1]
    eers = []
    
    for i in range(n_labels):
        eer, eer_threshold = EER(labels[:, i], predictions[:, i])
        eers.append(eer)
    
    overall_eer = np.mean(eers)
    return overall_eer

def weighted_accuracy(predictions, labels):

    predictions = (torch.sigmoid(predictions) > 0.5).float()
    predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

    cm = multilabel_confusion_matrix(labels, predictions)
    
    accuracies = []
    supports = []
    
    for i in range(cm.shape[0]):
        tn, fp, fn, tp = cm[i].ravel()
        
        # Accuracy for label i
        accuracy_i = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy_i)
        
        # Support for label i
        support_i = tp + fn
        supports.append(support_i)
   
    # Compute weighted accuracy
    weighted_accuracy = np.average(accuracies, weights=supports)
    
    return weighted_accuracy

# def EER(y_pred, y_true):

#     y_true = y_true.cpu().numpy()
#     y_pred = y_pred.cpu().numpy()
#     n_classes = y_pred.shape[1]
#     y_true_binarized = label_binarize(y_true, classes=range(n_classes))
#     eers = []

#     for i in range(n_classes):
#         fpr, tpr, thresholds = roc_curve(y_true_binarized[:, i], y_pred[:, i])
#         fnr = 1 - tpr
#         eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
#         eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#         eers.append(eer)

#     average_eer = np.mean(eers)
#     return average_eer

def f1_score_(predictions, labels):
    # predictions = torch.argmax(predictions, dim=1)
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    f1 = f1_score(predictions.cpu().numpy(), labels.cpu().numpy(), average='macro')
    return f1

def f1_score_per_class(predictions, labels):
    num_classes = predictions.shape[1]
    # apply sigmoid to the predictions and set 1 if greater than 0.5
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    f1 = []
    for i in range(num_classes):
        pred = predictions[:, i]
        label = labels[:, i]
        f1.append(f1_score(pred.cpu().numpy(), label.cpu().numpy(), average='macro'))    
    # append any class
    predictions_any =  (torch.sum(predictions[:, :num_classes-1], dim=1) >0 ).int()
    labels_any = (torch.sum(labels[:, :num_classes-1], dim=1) >0).int()
    f1.append(f1_score(predictions_any.cpu().numpy(), labels_any.cpu().numpy(), average='macro'))
    return f1


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_exp(cfg):

    cfg.output.save_dir = os.path.join(cfg.output.save_dir, cfg.data.name)
    cfg.data.ckpt = os.path.join(cfg.output.save_dir, cfg.data.ckpt)

    cfg.output.save_dir = os.path.join(cfg.output.save_dir, cfg.model.name)
    cfg.output.log_dir = os.path.join(cfg.output.save_dir, cfg.output.log_dir)
    cfg.output.checkpoint_dir = os.path.join(cfg.output.save_dir, cfg.output.checkpoint_dir)
    
    # make directories
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    os.makedirs(cfg.output.log_dir, exist_ok=True)
    os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.data.ckpt, exist_ok=True)

    # set seed
    set_seed(cfg.seed)


def _load_audio_file(row):
    audio_path = row['file_path']
    try:
        waveform, sample_rate = torchaudio.load(audio_path, format='wav')
        mel_spec = torchaudio.transforms.MelSpectrogram(win_length=400, hop_length=160, n_mels=40)(waveform)
        # pad the mel_spec to have the same length
        if mel_spec.shape[-1] < 301:
            print(f'Padding {audio_path}')
            mel_spec = torch.cat([mel_spec, torch.zeros(1,40, 301 - mel_spec.shape[-1])], dim=2)
            # mel_spec = None
        return (row.name, mel_spec)  # Return the index and the mel_spec
    except Exception as e:
        print(f"Error loading file {audio_path}: {e}")
        return (row.name, None)

def load_audio_files(df):
    mel_specs = [None] * len(df)  # Preallocate list with None
    failed = []
    with ThreadPoolExecutor() as executor:
        # Submit all tasks and get future objects
        futures = [executor.submit(_load_audio_file, row) for _, row in df.iterrows()]
        for future in tqdm(as_completed(futures)):
            index, mel_spec = future.result()
            
            if mel_spec is not None:
                mel_specs[index] = mel_spec.squeeze(0)  # Place mel_spec at its original index
            else:
                failed.append(index)
    # Filter out None values in case of errors
    mel_specs = [spec for spec in mel_specs if spec is not None]
    return mel_specs, failed