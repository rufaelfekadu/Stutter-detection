from .classification_metrics import *
from .sed_metrics import *

metric_registery = {
    'accuracy': Accuracy,
    'f1': F1Score,
    'eer': EER,
    'event_based': EventBasedMetric,
    'segment_based': SegmentBasedMetric,
}

def build_metrics(cfg):
    metrics = {}
    for metric in cfg.solver.metrics:
        metrics[metric] = metric_registery[metric](**cfg.metrics)
    return metrics