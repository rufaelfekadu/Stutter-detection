import av
import os
import torch
import random
import librosa
import numpy as np
import pandas as pd
import os.path as op
import soundfile as sf
from tqdm import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
rng = np.random.default_rng(42)

from stutter.utils.annotation import LabelMap

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

def aggregate_labels(df, item_col1, item_col2, annotator_col, label_cols, num_annotators=2, gold=False):
    if gold:
        # take the Gold annotator if it exists
        gold_ann = df.groupby([item_col1, item_col2])[annotator_col].transform(lambda x: x == 'Gold')
        merged_df = pd.merge(df, gold_ann, on=[item_col1, item_col2])
        filtered_items = merged_df[merged_df[annotator_col] == 'Gold']

        final_labels = filtered_items.groupby([item_col1, item_col2])[label_cols].any().astype(int).reset_index()
        all_items = final_labels[[item_col1, item_col2]].drop_duplicates()
        final_labels = pd.merge(all_items, df[gold_ann][[item_col1, item_col2] + label_cols].drop_duplicates(), on=[item_col1, item_col2], how='left').fillna(0)
        final_labels = final_labels.astype({col: 'int' for col in label_cols})
        # merge multiple gold labels per item if exists
        return final_labels
    
    # Group by the two item columns and annotator to get unique annotations
    annotator_counts = df.groupby([item_col1, item_col2])[annotator_col].nunique().reset_index(name='annotator_count')
    
    # Merge the annotator counts with the original DataFrame
    merged_df = pd.merge(df, annotator_counts, on=[item_col1, item_col2])
    
    # Filter items based on the number of annotators
    filtered_items = merged_df[merged_df['annotator_count'] >= num_annotators]

    # Aggregate the labels using 'any' to get the final labels
    final_labels = filtered_items.groupby([item_col1, item_col2])[label_cols].any().astype(int).reset_index()
    
    # Ensure all items are included in the final result
    all_items = df[[item_col1, item_col2]].drop_duplicates()
    final_labels = pd.merge(all_items, final_labels, on=[item_col1, item_col2], how='left').fillna(0)
    final_labels = final_labels.astype({col: 'int' for col in label_cols})
    
    return final_labels

def find_ranges(arr, smooth=0):
    ranges = []
    start = None
    zero_count = 0
    for i, val in enumerate(arr):
        if val == 1:
            if start is None:
                start = i
            zero_count = 0
        elif val == 0:
            if start is not None:
                zero_count += 1
                if zero_count > smooth:
                    ranges.append((start, i - zero_count + 1))
                    start = None
                    zero_count = 0

    if start is not None:
        ranges.append((start, len(arr)))

    return ranges

def construct_labels(label_path, num_frames, num_classes, sr=16000, clip_duration=30):
    label_map = LabelMap()
    reverse_map  = {v:k for k,v in label_map.description.items()}
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = np.zeros((num_frames, num_classes))
    for line in lines:
        start, end, class_name = line.strip().split(',')
        #  get key from dic for given value
        class_id = label_map.core.index(reverse_map[class_name])
        start = int(float(start) * num_frames / clip_duration)
        end = int(float(end) * num_frames / clip_duration)
        class_id = int(class_id)
        labels[start:end, class_id] = 1
    return labels

def deconstruct_labels(labels, sr=16000, clip_duration=30, smooth=0):
    label_map = LabelMap()
    num_frames, num_classes = labels.shape
    events = []
    for i in range(num_classes):
        class_label = labels[:, i]
        ranges = find_ranges(class_label, smooth=smooth)
        for start, end in ranges:
            start_time = start * clip_duration / num_frames
            end_time = end * clip_duration / num_frames
            if end_time - start_time < 0.7:
                continue
            class_name = label_map.description[label_map.core[i]]
            events.append((start_time, end_time, class_name))

    return events

def extract_mfcc(x, n_mfcc=40, n_fft=2048):
    y, sr = librosa.load(x)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    # mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc
# Video processing functions

STUTTER_CLASSES = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM']
PRIMARY_EVENT = ['SR', 'ISR', 'MUR', 'P', 'B']
SECONDARY_EVENT = ['V', 'FG', 'HM']

def custom_aggregation(group):
    # Calculate the number of rows for the annotator in the group
    row_count = len(group)
    aggregations ={
        'V': 'max',
        'FG': 'max',
        'HM': 'max',
        'ME': 'max',
        'SR': 'max',
        'ISR': 'max',
        'MUR': 'max',
        'P': 'max',
        'B': 'max',
        "T": "max"
    }
    group = group.agg(aggregations)
    group['events_count'] = row_count    
    return group

def most_common(x):
    return x.mode().iloc[0]

def make_video_dataframe(manifest_file, annotator, root:str = None, aggregate:bool = False, split:str = "train", agg_function = custom_aggregation, extension = ".mp4"):
    '''
    manifest_file: path to the manifest file
    annotator: path to the annotator file
    aggregate: If True, aggregate the labels
    root: Dataset root path
    agg_function: Function to aggregate the labels
    '''
    df = pd.read_csv(manifest_file)
    df = df[df['split'] == split]
    df.fillna(0, inplace=True)
    if 'clip_id' in df.columns:
        df['file_name'] = df['media_file'] + "/" + df['media_file'] + "_" +df ['clip_id'].astype(str) + extension
    else:
         df['file_name'] = df['media_file'] + "/" + df['file_name'] + extension
    full_data = df[['file_name']].drop_duplicates()
    print(f"Total {split} samples: {len(df)}")
    if annotator is not None:
        df = df[df['annotator'].isin([annotator, 0])]
    print(f"Annotator {annotator} has {len(df)} samples")
    if aggregate and agg_function is not None and annotator is not None:
        df = df.groupby('file_name').apply(agg_function).reset_index()
    else:
        df = df.groupby(['file_name', 'annotator']).apply(agg_function).reset_index()
        df = df.groupby('file_name').agg({'SR':most_common,
                                        'V': most_common,
                                        'FG': most_common,
                                        'HM': most_common,
                                        'ME': most_common,
                                        'SR': most_common,
                                        'ISR': most_common,
                                        'MUR': most_common,
                                        'P': most_common,
                                        'B': most_common,
                                        'T': "max"}).reset_index()
    df = pd.merge(full_data, df, on='file_name', how='left').fillna(0)
    df['stutter_event'] = df[STUTTER_CLASSES].apply(lambda x: (x > 0).any(), axis=1)
    df['primary_event'] = df[PRIMARY_EVENT].apply(lambda x: (x > 0).any(), axis=1)
    df['secondary_event'] = df[SECONDARY_EVENT].apply(lambda x: (x > 0).any(), axis=1)
    df['stutter_type'] = df[['primary_event', 'secondary_event']].apply(lambda x: "Both" if x['primary_event'] and x['secondary_event'] else "Primary" if x['primary_event'] else "Secondary" if x['secondary_event'] else "None", axis=1)
    if root is not None:
        df['file_name'] = df['file_name'].apply(lambda x: op.join(root, x))
    df['secondary_category'] = df[SECONDARY_EVENT].values.tolist()
    df['primary_category'] = df[PRIMARY_EVENT].values.tolist()
    df['stutter_category'] = df[STUTTER_CLASSES].values.tolist()
    print(f"Total samples after aggregation: {len(df)}")
    # print(f"{df.describe()}")
    return df

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * 1)
    if clip_len >= seg_len:
        # print(f"Segment length {seg_len} is greater than sample length")
        # converted_len = int(seg_len * 1)
        start_idx = 0
        end_idx = seg_len
    else:
        end_idx = rng.integers(converted_len, seg_len)
        start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    # assert len(indices) == 10, f"Expected 10 frames, got {len(indices)}"
    # for i in indices:
    #     reformatted_frame = container.decode(video=0)[i].reformat(width=224,height=224)
    #     frames.append(reformatted_frame)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            reformatted_frame = frame.reformat(width=224,height=224)
            frames.append(reformatted_frame)
    
    if len(frames) < 10:
        frames = random.choices(frames, k=10)
            
    new=np.stack([x.to_ndarray(format="rgb24") for x in frames])

    return new


def prepare_hf_dataset_video(example,processor,extractor, label_type:str = "secondary_event", clip_len=10):
    container = av.open(example['file_name'])
    indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=2, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container=container, indices=indices)
    inputs = processor(list(torch.tensor(video)), return_tensors='pt', do_resize=True)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values']).squeeze()
    inputs['labels'] = example[label_type]
    inputs['tension'] = example['T']
    del video,indices,container
    audio, sr = sf.read(example['file_name'].replace(".mp4", ".wav").replace("video", "audio"))
    audio_features = extractor(audio, sampling_rate=sr ,return_tensors='pt')
    inputs['input_values'] = audio_features['input_values']
    inputs['attention_mask'] = audio_features['attention_mask']
    del audio, sr, audio_features
    return inputs

if __name__ == "__main__":

    audio_path = "/fsx/homes/Rufael.Marew@mbzuai.ac.ae/projects/Stutter-detection/datasets/fluencybank/new_clips/FluencyBank/010/FluencyBank_010_0.wav"
    audio, sr = librosa.load(audio_path, sr=16000)
    mel_spec = logmelfilterbank(audio, sr, num_mels=40, hop_size=160, win_length=400)
    breakpoint()
