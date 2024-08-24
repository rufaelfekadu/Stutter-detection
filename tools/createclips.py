'''

python script to create clips from the audio samples 

'''

import os
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor

np.random.seed(0)

label_columns = ['FP','SR','ISR','MUR','P','B', 'V', 'FG', 'HM', 'ME', 'T']

def check_voiced(mini_frames, vad):
    voiced = [False]*len(mini_frames)
    for i,frame in enumerate(mini_frames):
        if vad.is_speech(frame.tobytes(), 16000):
            voiced[i] = True
    return sum(voiced)/len(voiced) > 0.5

def check_voiced_librosa(audio, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                     fmin=librosa.note_to_hz('C2'), 
                                                     fmax=librosa.note_to_hz('C7'), 
                                                     win_length=400, hop_length=160,
                                                     pad_mode='constant',sr=sr)
    return sum(voiced_flag)/len(voiced_flag) > 0.3

def check_intersects(row, start_time, end_time):
    if (row['start']>= start_time or row['end']<=end_time) and (row['start']<end_time and row['end']>start_time):
        return row['start']/1000, row['end']/1000, row.iloc[4:-1].values
    return None

def expand_label(df, sr, audio_length):
    starts = df['start'].values
    ends = df['end'].values
    label = np.clip(np.sum(df[label_columns].values, axis=1), a_max=1, a_min=0)
    labels = np.zeros(audio_length)
    for i in range(len(starts)):
        start = int(starts[i]*sr/1000)
        end = int(ends[i]*sr/1000)
        labels[start:end+1] = label[i]
    return labels

def chunk_audio(audio_length, sr, chunk_duration=25, stride_duration=5):
    chunk_samples = sr * chunk_duration
    stride_samples = sr * stride_duration
    starts = np.arange(0, audio_length - chunk_samples + 1, stride_samples)
    ends = starts + chunk_samples
    return starts, ends

def check_label(start, end, label, max_duration=30, sr=16000):

    if start < 0 or end >= len(label):
        return start if start >= 0 else 0, end if end < len(label) else len(label)
    
    if label[start]==0 and label[end]==0:
        return start, end
    if label[start]==1:
        start = start - 1600
    if label[end]==1:
        end = end - 1600
    
    return check_label(start, end, label)

    
def get_frames_for_audio(audio_path, clip_dir, label_path, label_df, clip_length=20, stride_duration=5):

    if len(label_df) == 0:
        return
    # Create output directory if it doesn't exist
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    
    # vad = webrtcvad.Vad(3)
    audio, sr = librosa.load(audio_path, sr=16000)

    assert sr == 16000, 'Sample rate should be 16000'

    label = expand_label(label_df, sr, audio.shape[0])
    starts, stops = chunk_audio(audio.shape[0], sr, chunk_duration=clip_length, stride_duration=stride_duration)
    result = {}
    num_labels = []
    for i, (start, stop) in enumerate(zip(starts, stops)):

        # # check if the chunk is voiced
        # if not check_voiced_librosa(audio[start:stop], sr):
        #     continue

        # # check if the chunk has no labels
        if not np.any(label[start:stop]):
            continue

        # # check if the chunk is between start and stop of any label
        # new_start, new_stop = check_label(start, stop, label)

        # if new_stop - new_start > 480000:
        #     print(f'Skipping {i} as it is more than 30 seconds {new_stop - new_start}')
        #     continue

        new_start, new_stop = start, stop

        clip_path = os.path.join(clip_dir, f'{i}.wav')
        clip = audio[new_start:new_stop]
        sf.write(clip_path, clip, sr)
        temp_df = label_df.apply(lambda row: check_intersects(row, (new_start/sr)*1000, (new_stop/sr)*1000), axis=1)
        temp_df = temp_df.dropna()
        num_labels.append(len(temp_df))

        # write the label to json file
        # temp_df.to_json(os.path.join(label_path, f'{i}.json'))
        # write to a text file
        with open(os.path.join(label_path, f'{i}.txt'), 'w') as f:
            temp_df = temp_df.sort_values()
            for row in temp_df:
                start_l = max(0, row[0] - new_start/sr)
                end_l = min(new_stop/sr - new_start/sr, row[1] - new_start/sr)
                f.write(f'<{start_l}> <stutter> <{end_l}> ')

    result['num_labels'] = num_labels
    result['num_clips'] = len(num_labels)
    result['clip_length'] = clip_length
    result['stride_duration'] = stride_duration

    return result
        
def process_wav_file(wav_path, args):
    audio_path = os.path.join(args.ds_path, wav_path)
    clip_path = os.path.join(args.output_path, args.anotator, 'clips', wav_path.split('.')[0])
    label_path = os.path.join(args.output_path, args.anotator, 'label', wav_path.split('.')[0])
    label_df = pd.read_csv(args.label_csv)
    label_df = label_df[label_df['media_file'] == wav_path]
    label_df = label_df[label_df['anotator'] == args.anotator]
    clip_length = np.random.randint(20, 30)
    result = get_frames_for_audio(audio_path, clip_path, label_path, label_df=label_df, clip_length=clip_length)
    return wav_path, result

if __name__ == "__main__":

    output_path = 'datasets/fluencybank/ds/reading_all/train/'
    ds_path = 'datasets/fluencybank/wavs_original/reading/'
    label_csv = 'datasets/fluencybank/csv/train/reading_train.csv'
    anotator = 'A3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=output_path)
    parser.add_argument('--ds_path', type=str, default=ds_path)
    parser.add_argument('--label_csv', type=str, default=label_csv)
    parser.add_argument('--anotator', type=str, default=anotator)
    args = parser.parse_args()

    results = {}
    with ThreadPoolExecutor() as executor:

        futures = []
        for wav_path in os.listdir(args.ds_path):
            futures.append(executor.submit(process_wav_file, wav_path, args))
        
        # Optionally, wait for all futures to complete
        for future in futures:
            path, res = future.result()
            if res is not None:
                results[path] = res
    
    # dump the results to a json file with indentation
    import json
    with open(args.output_path + 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # plot the distribution of number of labels
    import matplotlib.pyplot as plt
    plt.hist([x['num_labels'] for x in results.values()], bins=20)
    plt.ylabel('Number of clips')
    plt.xlabel('Number of labels')
    plt.savefig(args.output_path + 'num_labels_dis.png')
    


        