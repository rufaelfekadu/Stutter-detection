from .trainer import *
from .multimodal import MultiModalTrainer

trainer_registery = {
    'mtl': MTLTrainer,
    'yoho': SedTrainer,
    'sed': SedTrainer2,
    'vivit': VivitForStutterTrainer,
    'multimodal': MultiModalTrainer
}

def build_trainer(cfg, logger=None, metrics=['f1']):
    trainer = trainer_registery[cfg.setting](cfg, logger, metrics)
    return trainer