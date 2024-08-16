import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
import os
import torch

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
        # mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        # mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = logmelfilterbank(waveform, sample_rate, num_mels=n_mels, hop_size=hop_length, win_length=win_length)
        # mfcc = librosa.feature.mfcc(S=log_s, n_mfcc=n_mels)
        f0, voiced_flag, voiced_probs = librosa.pyin(waveform, 
                                                     fmin=librosa.note_to_hz('C2'), 
                                                     fmax=librosa.note_to_hz('C7'), 
                                                     win_length=400, hop_length=160,
                                                     pad_mode='constant',sr=sample_rate)
        
        f0 = np.nan_to_num(f0)  # Replace NaNs with zeros
        f0_delta = librosa.feature.delta(f0)
        if mel_spec.shape[0] < time_dim:
            mel_spec = np.pad(mel_spec, ((0, time_dim - mel_spec.shape[0]), (0, 0)), mode='constant')
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

def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=7600,
    eps=1e-10,
):
    """Compute log-Mel filterbank feature.
    (https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/bin/preprocess.py)
 
    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
 
    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).
 
    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)
 
    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
 
    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

if __name__ == "__main__":

    audio_path = "/fsx/homes/Rufael.Marew@mbzuai.ac.ae/projects/Stutter-detection/datasets/fluencybank/new_clips/FluencyBank/010/FluencyBank_010_0.wav"
    audio, sr = librosa.load(audio_path, sr=16000)
    mel_spec = logmelfilterbank(audio, sr, num_mels=40, hop_size=160, win_length=400)
    breakpoint()
