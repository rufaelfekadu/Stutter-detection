from .trainer import Trainer, MTLTrainer
from .multimodal import MultiModalTrainer
from .classification import ClassificationTrainer, Wave2vecTrainer
from .sed import SedTrainer, SedTrainer2
from .video import VivitForStutterTrainer

trainer_registery = {
    'classification': ClassificationTrainer,
    'wave2vec': Wave2vecTrainer,
    'mtl': MTLTrainer,
    'yoho': SedTrainer,
    'sed': SedTrainer,
    'vivit': VivitForStutterTrainer,
    'multimodal': MultiModalTrainer
}

def build_trainer(cfg, logger=None):
    trainer = trainer_registery[cfg.setting](cfg, logger=logger)
    return trainer