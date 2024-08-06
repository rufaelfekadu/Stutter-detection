
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


def _load_audio_file(row, **kwargs):

    frames = kwargs.get('n_frames', 3)
    n_mels = kwargs.get('n_mels', 40)
    hop_length = kwargs.get('hop_length', 160)
    win_length = kwargs.get('win_length', 400)
    sr = kwargs.get('sr', 16000)
    total_samples = sr*frames
    time_dim = (total_samples + win_length) // hop_length - 1

    audio_path = row['file_path']
    try:
        # waveform, sample_rate = torchaudio.load(audio_path, format='wav')
        # mel_spec = torchaudio.transforms.MelSpectrogram(win_length=win_length, hop_length=hop_length, n_mels=n_mels, sample_rate=sample_rate)(waveform)
        waveform, sample_rate = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # mfcc = librosa.feature.mfcc(S=log_s, n_mfcc=n_mels)

        f0, voiced_flag, voiced_probs = librosa.pyin(waveform, 
                                                     fmin=librosa.note_to_hz('C2'), 
                                                     fmax=librosa.note_to_hz('C7'), 
                                                     win_length=400, hop_length=160,
                                                     pad_mode='constant',sr=sample_rate)
        
        f0 = np.nan_to_num(f0)  # Replace NaNs with zeros
        f0_delta = librosa.feature.delta(f0)
        if mel_spec.shape[1] < time_dim:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, time_dim - mel_spec.shape[1])), mode='constant')
        if f0.shape[0] < time_dim:
            f0 = np.pad(f0, (0, time_dim - f0.shape[0]), mode='constant')
            f0_delta = np.pad(f0_delta, (0, time_dim - f0_delta.shape[0]), mode='constant')
            voiced_probs = np.pad(voiced_probs, (0, time_dim - voiced_probs.shape[0]), mode='constant')
        return (row.name, mel_spec, np.vstack([f0, f0_delta, voiced_probs]))  

    except Exception as e:
        print(f"Error loading file {audio_path}: {e}")
        return (row.name, None, None)

def load_audio_files(df, **kwargs):
    mel_specs = [None] * len(df)  # Preallocate list with None
    f0s = [None] * len(df)
    failed = []
    with ProcessPoolExecutor() as executor:
        # Submit all tasks and get future objects
        futures = {executor.submit(_load_audio_file, row, **kwargs):i for i, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]
            try:
                _, mel_spec, f0 = future.result()
                if mel_spec is not None:
                    mel_specs[index] = mel_spec  # Place mel_spec at its original index
                    f0s[index] = f0
                else:
                    failed.append(index)
            except Exception as e:
                print(f"Error loading file: {e}")
                failed.append(index)
    
    return mel_specs, f0s, failed