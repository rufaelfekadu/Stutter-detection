from .lstm import LSTMModel
from .convlstm import ConvLSTM
from .lstmext import LSTMExt
from .whisper_cnn import WhisperDetector
from .sednet import SedNet
from .wav2vec import Wav2Vec2Classifier
import torch

model_registery = {
    'lstm': LSTMModel,
    'convlstm': ConvLSTM,
    'lstmext': LSTMExt,
    'whisperyoho': WhisperDetector,
    'sednet': SedNet,
    'wav2vec': Wav2Vec2Classifier
}

class NoneScheduler:
    def __init__(self, lr):
        self.lr = lr
    def step(self, *args, **kwargs):
        pass
    def state_dict(self):
        pass
    def load_state_dict(self, state_dict):
        pass
    def get_last_lr(self):
        return [self.lr]

def build_model(cfg):
    model = model_registery[cfg.model.name](**cfg.model)

    if cfg.solver.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.solver.lr)
    elif cfg.solver.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.solver.lr, momentum=cfg.solver.momentum)

    if cfg.solver.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.solver.factor, patience=cfg.solver.patience, min_lr=cfg.solver.min_lr)
    elif cfg.solver.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.solver.milestones, gamma=cfg.solver.gamma)
    elif cfg.solver.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.solver.T_max, eta_min=cfg.solver.eta_min)
    else:
        scheduler = NoneScheduler(lr = cfg.solver.lr)
    
    return model, optimizer, scheduler