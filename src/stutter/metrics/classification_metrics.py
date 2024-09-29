import numpy as np
from sklearn.metrics import f1_score, multilabel_confusion_matrix
import torcheval.metrics.functional as F
import torch

class Accuracy(object):
    def __init__(self, **kwargs):
        pass
    def __call__(self, pred, label):
        if pred.shape[1] ==1:
            pred = torch.sigmoid(pred)
            return F.multilabel_accuracy(pred, label, threshold=0.5)
        else:
            pred = torch.sigmoid(pred)
            return F.multilabel_accuracy(pred, label, threshold=0.5, criteria='hamming')


class F1Score(object):
    def __init__(self, **kwargs):
        pass
    def __call__(self, pred, label):
        if len(pred.shape)==1:
            pred = torch.sigmoid(pred)
            return F.binary_f1_score(pred, label, threshold=0.5)
        else:
            f1 = []
            for i in range(pred.shape[1]):
                pred[:, i] = torch.sigmoid(pred[:, i])
                f1.append(F.binary_f1_score(pred[:, i], label[:, i], threshold=0.5))
            return f1
        

class EER(object):
    def __init__(self, **kwargs):
        pass
    def __call__(self, y_true, y_pred):
        pass

