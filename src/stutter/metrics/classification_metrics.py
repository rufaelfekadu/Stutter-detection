import numpy as np
from sklearn.metrics import f1_score, multilabel_confusion_matrix


class Accuracy(object):
    def __init__(self, **kwargs):
        pass
    def __call__(self, y_true, y_pred):
        pass

class F1Score(object):
    def __init__(self, **kwargs):
        pass
    def __call__(self, y_true, y_pred):
        pass

class EER(object):
    def __init__(self, **kwargs):
        pass
    def __call__(self, y_true, y_pred):
        pass

