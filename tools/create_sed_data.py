
import os
import glob
import numpy as np
import soundfile as sf
import pandas as pd
from stutter.utils import logmelfilterbank, LabelMap
from tqdm import tqdm

# Path to the directory containing the SED data
data_dir = 'datasets/fluencybank/wavs/reading/wavs'
label_csv = 'datasets/fluencybank/ds_label/reading/A1/total_label.csv'

label_df = pd.read_csv(label_csv)

def sample_labels(df, array_size, total_duration, label_columns):

    label_arrays = {label: np.zeros(array_size, dtype=int) for label in label_columns}
    time_step = total_duration / array_size
    
    # Iterate through the DataFrame
    for _, row in df.iterrows():
        start_idx = int(row['start'] / time_step)
        end_idx = int(row['end'] / time_step)
        
        start_idx = max(0, start_idx)
        end_idx = min(array_size, end_idx)
        
        for label in label_columns:
            label_arrays[label][start_idx:end_idx] = row[label]
    
    return label_arrays

audio_files = glob.glob(os.path.join(data_dir, '*.wav'))

# max_length = 0
# for audio_file in audio_files:
#     audio, sr = sf.read(audio_file)
#     if len(audio) > max_length:
#         max_length = len(audio)

# print('Max length:', max_length)

inputs_train = []
inputs_test = []
for i in LabelMap().labels:
    globals()[f'{i}_train'] = []
    globals()[f'{i}_test'] = []

max_length = 15*16000

def get_max_segment(start, end, max_length, audio_length):
    duration = end - start
    if audio_length < max_length:
        return 0, audio_length
    else:
        clip_start = max(0, int(start - (max_length - duration)/2))
        clip_end = min(audio_length, int(end + (max_length - duration)/2))
        return clip_start, clip_end

for audio_file in tqdm(audio_files):
    audio, sr = sf.read(audio_file)
    audio_length = len(audio)
    temp_label_df = label_df[(label_df['media_file'] == os.path.basename(audio_file).replace('.wav', '')) & (label_df['split'] == 'test')]    
    print(f'Processing {os.path.basename(audio_file)}')
    for i, row in temp_label_df.iterrows():

        start = row['start']*sr/1000
        end = row['end']*sr/1000
        label = row['label']

        clip_start, clip_end = get_max_segment(start, end, max_length, audio_length)
        clip = audio[clip_start:clip_end]

        if len(clip) < max_length:
            clip = np.pad(clip, (0, max_length - len(clip)), 'constant')

        mel_spec = logmelfilterbank(clip, sr, num_mels=40, hop_size=160)

        clip_duration = clip_end - clip_start
        label = LabelMap().labelfromstr(label)

        temp = temp_label_df[(temp_label_df['start'] >= (clip_start*1000/sr)) & (temp_label_df['end'] <= (clip_end*1000/sr))]
        label_arrays = sample_labels(temp, mel_spec.shape[0], clip_duration, LabelMap().labels)
        inputs_train.append(mel_spec)
        for key, value in label_arrays.items():
            globals()[f'{key}_train'].append(value)
        

inputs = np.array(inputs_train)
for i in LabelMap().labels:
    globals()[f'{i}_train'] = np.array(globals()[f'{i}_train'])
    np.save('datasets/fluencybank/data_sed/sed_outputs_A1_validation_{}.npy'.format(i), globals()[f'{i}_train'])

outputs_total = np.stack([globals()[f'{i}_train'] for i in LabelMap().labels], axis=-1)

np.save('datasets/fluencybank/data_sed/sed_inputs_A1_validation.npy', inputs)
np.save('datasets/fluencybank/data_sed/sed_outputs_A1_validation.npy', outputs_total)





