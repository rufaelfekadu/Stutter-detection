
import torch
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from sklearn.preprocessing import MinMaxScaler
import librosa
import torchaudio



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




def get_eaf_files(path, ext='.eaf'):

    elanfiles = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                if file not in elanfiles:
                    elanfiles[file] = [os.path.join(root, file)]
                else:
                    elanfiles[file].append(os.path.join(root, file))
    return elanfiles