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

from stutter.utils.data import aggregate_labels, logmelfilterbank
from stutter.utils.annotation import LabelMap

np.random.seed(0)
random.seed(0)

# lock = threading.Lock()
label_columns = ['SR','ISR','MUR','P','B', 'V', 'FG', 'HM', 'ME', 'T']

def compute_iou(start1, end1, start2, end2):
    start = max(start1, start2)
    end = min(end1, end2)
    intersection = max(0, end - start)
    # union = (end1 - start1) + (end2 - start2) - intersection
    return intersection/(end1-start1+1e-6)

def chunk_audio_df(clip_df, sr, max_duration=10, min_duration=0.1):
    starts = (clip_df['start']*sr/1000).values.astype(int)
    ends = (clip_df['end']*sr/1000).values.astype(int)
    for i, (start, end) in enumerate(zip(starts, ends)):
        if end - start > max_duration:
            ends[i] = end
        if end - start < min_duration:
            ends[i] = start + min_duration
    return starts, ends

def chunk_audio(audio_length, sr, chunk_duration=10, stride_duration=5):
    chunk_samples = sr * chunk_duration
    stride_samples = sr * stride_duration
    starts = np.arange(0, audio_length - chunk_samples + 1, stride_samples)
    ends = starts + chunk_samples
    return starts, ends

def get_frames_for_audio(args, file_name):

    label_map = LabelMap()
    label_df = pd.read_csv(args.label_csv)
    label_df = label_df[label_df['media_file'] == file_name]
    label_df = label_df[label_df['annotator'] == args.annotator]

    with open(args.clip_csv, 'r') as f:
        clip_df = json.load(f)
    clip_df = clip_df[file_name+'.wav']

    with open(args.speaker_file, 'r') as f:
        speakers = json.load(f)
    speaker_id = speakers[file_name]


    if len(label_df) == 0:
        return file_name, None
    
    # Create output directory if it doesn't exist
    audio_clip_dir = os.path.join(args.output_path, 'clips', 'audio')
    feature_clip_dir = os.path.join(args.output_path, 'clips', 'feature')
    video_clip_dir = os.path.join(args.output_path, 'clips', 'video')
    label_path = os.path.join(args.output_path, 'label', 'sed')

    os.makedirs(audio_clip_dir, exist_ok=True)
    os.makedirs(video_clip_dir, exist_ok=True)
    os.makedirs(feature_clip_dir, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    audio_path = os.path.join(args.ds_path, f'{file_name}.wav')
    audio, sr = librosa.load(audio_path, sr=16000)
    assert sr == 16000, 'Sample rate should be 16000'

    # # open the video
    # video_path = os.path.join(args.ds_path, 'videos', f'{file_name}.mp4')
    # cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    result = {}
    num_labels = []
    total_df = pd.DataFrame()
    for i, row in enumerate(clip_df):
        if row['speaker_id'] != speaker_id:
            continue

        
        start = int(row['start_time']*sr)
        stop = int(row['end_time']*sr)
        # # select the labels with iou > 0.5
        temp_df = label_df.copy()
        temp_df['iou'] = temp_df.apply(lambda row: compute_iou(row['start'], row['end'], (start/sr)*1000, (stop/sr)*1000), axis=1)
        temp_df = temp_df[temp_df['iou'] > 1e-4]
        num_labels.append(len(temp_df))

        duration = (stop - start)/sr
        if duration < args.clip_length:
            continue
        # assert duration == args.clip_length, f'Clip duration is {duration}'

        # check if the audio clip is already created
        audio_clip_path = os.path.join(audio_clip_dir, f'{file_name}_{i}.wav')
        feature_clip_path = os.path.join(feature_clip_dir, f'{file_name}_{i}.npy')

        if not os.path.exists(audio_clip_path):
            clip = audio[start:stop]
            sf.write(audio_clip_path, clip, sr)
        
        if not os.path.exists(feature_clip_path):
            clip = audio[start:stop]
            feature = logmelfilterbank(clip, sr)
            np.save(feature_clip_path, feature)

        # # check if the video clip is already created
        # video_clip_path = os.path.join(video_clip_dir, f'{file_name}_{i}.mp4')
        # if not os.path.exists(video_clip_path):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, int(start*fps/sr))
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     out = cv2.VideoWriter(video_clip_path, fourcc, fps, (width, height))
        #     for j in range(int(duration*fps)):
        #         ret, frame = cap.read()
        #         if ret:
        #             out.write(frame)
        #         else:
        #             break
        #     out.release()
        
        # write to a text file for sed models
        with open(os.path.join(label_path, f'{file_name}_{i}.rttm'), 'w') as f:
            temp_df = temp_df.sort_values(by='start')
            for _, row in temp_df.iterrows():
                start_l = max(0, (row['start']/1000 - start/sr))
                end_l = min(duration, (row['end']/1000 - start/sr))
                label = label_map.get_core(row['label'])[0]
                if len(label) > 0:
                    # out = '\n'.join([f"{round(start_l,2)},{round(end_l,2)},{label_map.description[c]}" for c in label])
                    out = '\n'.join([f"SPEAKER {file_name}_{i} 1 {start_l:.3f} {(end_l-start_l):.3f} <NA> <NA> {label_map.description[c]} <NA> <NA>" for c in label])
                    f.write(f"{out}\n")
        
        with open(os.path.join(label_path, f'{file_name}_{i}.uem'), 'w') as f:
            f.write(f"{file_name}_{i} NA 0 {duration:.3f}")

        # split = label_df['split'].values[0]
        if len(temp_df) == 0:
            temp = dict(zip(label_map.labels, [0]*len(label_columns)))
            temp_df = pd.DataFrame([temp], columns=label_columns)
            temp_df['start'] = np.nan
            temp_df['end'] = np.nan
            temp_df['annotator'] = 'Gold'
            temp_df['media_file'] = file_name
            temp_df['label'] = 'NS;;'
            # temp_df['split'] = split
            # temp_df['iou'] = 0

        temp_df['clip_id'] = i
        temp_df['clip_start'] = (start/sr) * 1000
        temp_df['clip_end'] = (stop/sr) * 1000
        reord_cols = ['media_file', 'clip_id', 'clip_start', 'clip_end', 'annotator', 'start', 'end', 'label']
        temp_df = temp_df[reord_cols]
        total_df = pd.concat([total_df, temp_df], ignore_index=True, axis=0)


    result['total_df'] = total_df
    result['num_labels'] = num_labels
    result['num_clips'] = len(num_labels)
    result['clip_length'] = args.clip_length
    result['stride_duration'] = args.stride_duration

    return file_name, result


def main(args):
    
    results = {}

    # read split file
    with open(args.split_file, 'r') as f:
        split_file = json.load(f)

    with ThreadPoolExecutor() as executor:
        futures = []
        for wav_path in os.listdir(args.ds_path):
            # if wav_path.split('.')[0] in split_file['train'] or wav_path.split('.')[0] in split_file['val']:
            futures.append(executor.submit(get_frames_for_audio, args, wav_path.split('.')[0]))
        
        # Optionally, wait for all futures to complete
        for future in tqdm(futures):
            path, res = future.result()
            if res is not None:
                results[path] = res

    total_df = pd.DataFrame()
    for key, value in results.items():
        sub_total_df = value['total_df']
        total_df = pd.concat([total_df, sub_total_df])
        value.pop('total_df')
    total_df.to_csv(args.output_path + 'total_df.csv', index=False)
    
    with open(args.output_path + 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    output_path = 'datasets/fluencybank/ds_dia/interview/'
    ds_path = 'datasets/fluencybank/wavs/interview/'
    label_csv = 'datasets/fluencybank/our_annotations/interview/csv/labels_1.csv'
    clip_csv = 'datasets/fluencybank/our_annotations/interview/csv/transcripts.csv'
    split_file = 'datasets/fluencybank/our_annotations/interview_split.json'
    speaker_diarization = 'datasets/fluencybank/diarization/interview.json'
    speaker_file = 'datasets/fluencybank/diarization/speaker.json'
    annotator = 'A1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=output_path)
    parser.add_argument('--ds_path', type=str, default=ds_path)
    parser.add_argument('--split_file', type=str, default=split_file)
    parser.add_argument('--label_csv', type=str, default=label_csv)
    parser.add_argument('--clip_csv', type=str, default=speaker_diarization)
    parser.add_argument('--speaker_file', type=str, default=speaker_file)
    parser.add_argument('--annotator', type=str, default=annotator)
    parser.add_argument('--stride_duration', type=int, default=5)
    parser.add_argument('--clip_length', type=int, default=1)
    parser.add_argument('--method', type=str, default='constant_stride')
    args = parser.parse_args()

    main(args)
