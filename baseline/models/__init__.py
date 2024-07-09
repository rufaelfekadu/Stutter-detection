from .lstm import LSTMModel
from .convlstm import ConvLSTM

available_models = {
    'lstm': LSTMModel,
    'convlstm': ConvLSTM
}

def get_model(cfg):
    return available_models[cfg.model_name]

