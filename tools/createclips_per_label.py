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
import threading
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torch
import random
import json
from tqdm import tqdm
import cv2

from stutter.utils.data import aggregate_labels
from stutter.utils.annotation import LabelMap
from scipy import stats

np.random.seed(0)
random.seed(0)
label_map = LabelMap()



# Function to generate new start and stop times
def generate_non_overlapping_times(df, num_new_times, duration_dist=None):
    new_times = []
    existing_intervals = [(row['start'], row['end']) for _, row in df.iterrows()]
    
    while len(new_times) < num_new_times:
        # Sample a duration from the existing durations
        if duration_dist is not None:
            duration = duration_dist.resample(1)[0][0]
            if duration < 300:
                duration = 300
        else:
            duration = np.random.choice(df['duration'])

        # Generate a random start time
        max_start = df['end'].max() + 1  # Ensure it starts after the last stop time
        start_time = np.random.randint(0, max_start)
        stop_time = start_time + duration
        
        # Check for overlap
        overlap = any(start < stop_time and stop > start_time for start, stop in existing_intervals)
        
        if not overlap:
            new_times.append((start_time, stop_time))
            existing_intervals.append((start_time, stop_time))
        
    new_label_df = pd.DataFrame(new_times, columns=['start', 'end'])
    new_label_df['label'] = 'NoStutter'
        
    return new_label_df

def get_frames_for_audio(args, file_name):
    # split = 'test' if args.annotator == 'Gold' else 'train'
    label_df = pd.read_csv(args.label_csv)
    if args.annotator == 'Gold':
        label_df = label_df[(label_df['split'] == 'test')&(label_df['annotator'] == 'Gold')]
    else:
        label_df = label_df[(label_df['split'].isin(['train', 'val'])) | ((label_df['annotator'] == 'Gold') & (label_df['split'] == 'test'))]
    label_df['duration'] = label_df['end'] - label_df['start']
    # fit a kernel density estimate to the duration distribution
    duration_dist = stats.gaussian_kde(label_df['duration'])
    label_df = label_df[label_df['media_file'] == file_name]
    # label_df = label_df[label_df['split'] == split]


    # get the maximum count per annotator from the dataframe 
    max_count = label_df.sum()[label_map.labels[:-1]].max()
    print(f'Maximum count per annotator: {max_count} for {file_name}')
    new_label_df = generate_non_overlapping_times(label_df, max_count, duration_dist=duration_dist)
    new_label_df['annotator'] = args.annotator
    new_label_df['split'] = label_df['split'].iloc[0]
    new_label_df['media_file'] = file_name
    
    label_df = pd.concat([label_df, new_label_df], ignore_index=True)

    # label_df = label_df[label_df['annotator'] == args.annotator]
    # durations = label_df['end'] - label_df['start']
    # sample duration from the distribution of durations
    # duration = np.random.choice(durations)
    # select samples with the same duration that don't overlap with those in the label_df
    label_df = label_df.sort_values('start')
    # label_df['end'] = label_df['start'] + duration

    if len(label_df) == 0:
        print(f'No labels found for {file_name} for annotator {args.annotator}')
        return None, None
    
    # Create output directory if it doesn't exist
    audio_clip_dir = os.path.join(args.output_path, 'clips', 'audio', file_name)
    video_clip_dir = os.path.join(args.output_path, 'clips', 'video', file_name)
    # label_path = os.path.join(args.output_path, 'label', file_name)

    os.makedirs(audio_clip_dir, exist_ok=True)
    os.makedirs(video_clip_dir, exist_ok=True)

    audio_path = os.path.join(args.ds_path, 'wavs', f'{file_name}.wav')
    audio, sr = librosa.load(audio_path, sr=16000)
    assert sr == 16000, 'Sample rate should be 16000'

    # open the video
    video_path = os.path.join(args.ds_path, 'videos', f'{file_name}.mp4')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    num_labels = []
    skipped = []
    result = {}  
    label_df['file_name'] = ''    
    for i, row in label_df.iterrows():

        start = int(row['start'] * sr / 1000)
        stop = int(row['end'] * sr / 1000)

        label_df.at[i, 'file_name'] = f'{file_name}_{i}'

        duration = (stop - start)/sr
        if duration < args.min_clip_length:
            print(f'Skipping {file_name}_{i} as duration is {duration} label: {row["label"]}')
            skipped.append(f'{file_name}_{i}')
            continue

        # check if the audio clip is already created
        audio_clip_path = os.path.join(audio_clip_dir, f'{file_name}_{i}.wav')
        if not os.path.exists(audio_clip_path):
            clip = audio[start:stop]
            sf.write(audio_clip_path, clip, sr)
        
        # check if the video clip is already created
        video_clip_path = os.path.join(video_clip_dir, f'{file_name}_{i}.mp4')
        if not os.path.exists(video_clip_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start*fps/sr))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_clip_path, fourcc, fps, (width, height))
            for j in range(int(duration*fps)):
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break
            out.release()

    result['df'] = label_df[['file_name', 'start', 'end', 'label', 'annotator', 'split', 'media_file']]
    result['skpipped'] = skipped
    return file_name, result
        
def create_clips_label(args):

    results = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        print(os.listdir(os.path.join(args.ds_path,'wavs')))
        for ds_path in os.listdir(os.path.join(args.ds_path,'wavs')):
            futures.append(executor.submit(get_frames_for_audio, args, ds_path.split('.')[0]))
        
        for future in tqdm(futures):
            path, res = future.result()
            if res:
                results[path] = res
        
    total_df = pd.DataFrame()
    for key, value in results.items():
        sub_total_df = value['df']
        # drop skipped rows
        sub_total_df = sub_total_df[~sub_total_df['file_name'].isin(value['skpipped'])]
        total_df = pd.concat([total_df, sub_total_df])
        value.pop('df')

    total_df[label_map.labels] = total_df['label'].apply(lambda x: pd.Series(label_map.labelfromstr(x)))      
    total_df.to_csv(os.path.join(args.output_path, 'total_label.csv'), index=False)

    with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":

    output_path = 'datasets/fluencybank/ds_label/reading/bau_3/'
    ds_path = 'datasets/fluencybank/wavs/reading/'
    label_csv = 'datasets/fluencybank/our_annotations/reading/csv/total_dataset.csv'
    anotator = 'bau'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=output_path)
    parser.add_argument('--ds_path', type=str, default=ds_path)
    parser.add_argument('--label_csv', type=str, default=label_csv)
    parser.add_argument('--annotator', type=str, default=anotator)
    parser.add_argument('--min_clip_length', type=int, default=0.03)
    args = parser.parse_args()

    create_clips_label(args)

    
    


        