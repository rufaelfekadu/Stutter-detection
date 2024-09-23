from .loss import *
from .sed_loss import *

loss_registery = {
    'ccc': CCCLoss,
    'ce': CrossEntropyLoss,
    'focal': FocalLoss,
    'focal-m': FocalLossMultiClass,
    'bce': BCELoss,
    'bce-m': BCELossMulticlass,
    'yoho': YOHOLoss,
    'sed': SedLoss,
    'mae': WeightedMAELoss,
    'focal-ls': FocalLossWithLabelSmoothing,
}

def build_loss(cfg):
    criterion = {}
    for key, loss in zip(cfg.tasks, cfg.solver.losses):
        criterion[key] = loss_registery[loss](device=cfg.solver.device, **cfg.loss ).to(cfg.solver.device)
    return criterion