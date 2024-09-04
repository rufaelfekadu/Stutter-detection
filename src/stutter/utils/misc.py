
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

    if cfg.data.encoder_name != cfg.model.encoder_name:
        print(f"Warning: Encoder name in data config is different from model config. Using encoder name from model config.")
        cfg.data.enocder_name = cfg.model.encoder_name
    
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

def plot_sample(*samples, title=None, figsize=(20, 10), save_path=None, **kwargs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(samples), 1, figsize=figsize)

    if len(samples) == 1: ax = [ax]

    for i, sample in enumerate(samples):
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().numpy()
        ax[i].imshow(sample, **kwargs)
        ax[i].set_title(title[i])
    
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    cbar = fig.colorbar(ax[0].images[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax