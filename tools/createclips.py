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

np.random.seed(0)
random.seed(0)

# lock = threading.Lock()
label_columns = ['SR','ISR','MUR','P','B', 'V', 'FG', 'HM', 'ME', 'T']

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
    if row['start'] <= end_time and row['end']>=start_time:
        return row
    return None

def check_intersects_strict(row, start_time, end_time):
    if row['start'] >= start_time and row['end']<=end_time:
        return True
    return False

def compute_iou(start1, end1, start2, end2):
    start = max(start1, start2)
    end = min(end1, end2)
    intersection = max(0, end - start)
    # union = (end1 - start1) + (end2 - start2) - intersection
    return intersection/(end1-start1+1e-6)

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

def chunk_audio(audio_length, sr, chunk_duration=10, stride_duration=5):
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
 
def get_frames_for_audio(args, file_name):

    label_map = LabelMap()
    label_df = pd.read_csv(args.label_csv)
    label_df = label_df[label_df['media_file'] == file_name]
    # label_df = label_df[label_df['annotator'] == 'A2']
    if len(label_df) == 0:
        return
    
    # Create output directory if it doesn't exist
    audio_clip_dir = os.path.join(args.output_path, 'clips', 'audio', file_name)
    video_clip_dir = os.path.join(args.output_path, 'clips', 'video', file_name)
    label_path = os.path.join(args.output_path, 'label', file_name)

    os.makedirs(audio_clip_dir, exist_ok=True)
    os.makedirs(video_clip_dir, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    # os.makedirs(label_path, exist_ok=True)
    # os.makedirs(os.path.join(label_path,'uem'), exist_ok=True)
    # os.makedirs(os.path.join(label_path,'rttm'), exist_ok=True)
    # os.makedirs(os.path.join(label_path,'txt'), exist_ok=True)
    # os.makedirs(os.path.join(label_path,'json'), exist_ok=True)
    os.makedirs(os.path.join(label_path,'sed'), exist_ok=True)

    audio_path = os.path.join(args.ds_path, 'wavs', f'{file_name}.wav')
    audio, sr = librosa.load(audio_path, sr=16000)
    assert sr == 16000, 'Sample rate should be 16000'

    # open the video
    video_path = os.path.join(args.ds_path, 'videos', f'{file_name}.mp4')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
    # label = expand_label(label_df, sr, audio.shape[0])
    starts, stops = chunk_audio(audio.shape[0], sr, chunk_duration=args.clip_length, stride_duration=args.stride_duration)
    result = {}
    num_labels = []
    total_df = pd.DataFrame()
    for i, (start, stop) in enumerate(zip(starts, stops)):

        duration = (stop - start)/sr
        assert duration == args.clip_length, f'Clip duration is {duration}'

        # check if the audio clip is already created
        audio_clip_path = os.path.join(audio_clip_dir, f'{file_name}_{i}.wav')
        if not os.path.exists(audio_clip_path):
            clip = audio[start:stop]
            sf.write(audio_clip_path, clip, sr)
        
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
        
        split = label_df['split'].values[0]
        percentage = 0.0
        temp_df = label_df[
            (label_df['start'] <= ((stop / sr) * 1000) ) & 
            (label_df['end'] >= ((start / sr) * 1000))
            ]
        # # select the labels with iou > 0.5
        # temp_df = label_df.copy()
        # temp_df['iou'] = temp_df.apply(lambda row: compute_iou(row['start'], row['end'], (start/sr)*1000, (stop/sr)*1000), axis=1)
        # temp_df = temp_df[temp_df['iou'] > 0.1]
        num_labels.append(len(temp_df))

        if len(temp_df) == 0:
            temp = dict(zip(label_map.labels, [0]*len(label_columns)))
            temp_df = pd.DataFrame([temp], columns=label_columns)
            temp_df['start'] = np.nan
            temp_df['end'] = np.nan
            temp_df['annotator'] = 'Gold'
            temp_df['media_file'] = file_name
            temp_df['label'] = 'no_stutter'
            temp_df['split'] = split
            # temp_df['iou'] = 0

        temp_df['clip_id'] = i
        temp_df['clip_start'] = (start/sr) * 1000
        temp_df['clip_end'] = (stop/sr) * 1000
        reord_cols = ['media_file', 'clip_id', 'clip_start', 'clip_end', 'annotator', 'start', 'end', 'label', 'split'] + label_columns
        temp_df = temp_df[reord_cols]
        total_df = pd.concat([total_df, temp_df], ignore_index=True, axis=0)

        # write to a text file for sed models
        # with open(os.path.join(label_path,'sed', f'{file_name}_{i}.txt'), 'w') as f:
        #     temp_df = temp_df.sort_values(by='start')
        #     for _, row in temp_df.iterrows():
        #         start_l = max(0, (row['start']/1000 - start/sr))
        #         end_l = min(duration, (row['end']/1000 - start/sr))
        #         _, core_label = label_map.get_core(row['label'])
        #         if any(core_label):
        #             out = [f"{c} {round(start_l,2)} {round(end_l,2)}" for c in core_label]
        #             f.write(f"{' '.join(out)}\n")
        
        # if split == 'test':
        # # write refernce txt file
        # with open(os.path.join(label_path,'sed', f'{file_name}_{i}_ref.txt'), 'w') as f:
        #     temp_df = temp_df.sort_values(by='start')
        #     for _, row in temp_df.iterrows():
        #         start_l = max(0, (row['start']/1000 - start/sr))
        #         end_l = min(duration, (row['end']/1000 - start/sr))
        #         str_core, _ = label_map.get_core(row['label'])
        #         for c in str_core:
        #             f.write(f"{round(start_l,4)},{round(end_l,4)},{label_map.description[c]}\n")

    result['total_df'] = total_df
    result['num_labels'] = num_labels
    result['num_clips'] = len(num_labels)
    result['clip_length'] = args.clip_length
    result['stride_duration'] = args.stride_duration

    return file_name, result
        
# def process_wav_file(wav_path, args):
#     clip_path = os.path.join(args.output_path, 'clips')
#     label_df = pd.read_csv(args.label_csv)
#     label_df = label_df[label_df['media_file'] == wav_path.split('.')[0]]
#     result = get_frames_for_audio(ds_path, audio_clip_path, video_label_path, label_df=label_df, video_path= clip_length=args.clip_length, stride_duration=args.stride_duration)
#     return wav_path, result

def create_clips_strided(args):

    results = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for wav_path in os.listdir(os.path.join(args.ds_path,'wavs')):
            futures.append(executor.submit(get_frames_for_audio, args, wav_path.split('.')[0]))
        
        for future in futures:
            path, res = future.result()
            if res is not None:
                results[path] = res
    
    total_df = pd.DataFrame()
    for key, value in results.items():
        sub_total_df = value['total_df']
        total_df = pd.concat([total_df, sub_total_df])
        value.pop('total_df')

    
    total_df.to_csv(args.output_path + 'total_df.csv', index=False)
    # combine the dataframes
    grouped = total_df.groupby(['media_file','clip_id'])

    any_df= grouped[label_columns[:-1]].any().astype(int).reset_index()
    any_df.to_csv(args.output_path + 'any_df.csv', index=False)

    # any_2_df = aggregate_labels(total_df[total_df['split'].isin(['train'])], 'media_file', 'clip_id', 'annotator', label_columns[:-1], num_annotators=2)
    # any_2_df['split'] = 'train'
    # any_2_test_df = aggregate_labels(total_df[total_df['split'].isin(['test'])], 'media_file', 'clip_id', 'annotator', label_columns[:-1], num_annotators=1, gold=True)
    # any_2_test_df['split'] = 'test'
    # any_2_df = pd.concat([any_2_df, any_2_test_df], ignore_index=True)
    # any_2_df.to_csv(args.output_path + 'any_2_df.csv', index=False)
    # dump the results to a json file with indentation
    import json
    with open(args.output_path + 'meta_data.json', 'w') as f:
        json.dump(results, f, indent=4)

    # # plot the distribution of number of labels
    # import matplotlib.pyplot as plt
    # plt.hist([x['num_labels'] for x in results.values()], bins=20)
    # plt.ylabel('Number of clips')
    # plt.xlabel('Number of labels')
    # plt.savefig(args.output_path + 'num_labels_dis.png')

def get_frames_from_transcript(audio_path, clip_dir, label_path, label_df, clip_df, pipeline):

    '''
    Create clips from the audio sample based on the transcript
    '''

    if len(label_df) == 0:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(os.path.join(label_path,'uems'), exist_ok=True)
    os.makedirs(os.path.join(label_path,'rttms'), exist_ok=True)
    os.makedirs(os.path.join(label_path,'txts'), exist_ok=True)
    os.makedirs(os.path.join(label_path,'jsons'), exist_ok=True)
    os.makedirs(os.path.join(label_path,'yohos'), exist_ok=True)
    file_id = os.path.basename(audio_path).replace('.wav','')
    
    # vad = webrtcvad.Vad(3)
    audio, sr = librosa.load(audio_path, sr=16000)

    assert sr == 16000, 'Sample rate should be 16000'

    result = {}
    num_labels_per_clip = []
    durations = []
    total_df = pd.DataFrame()
    vad_labels = {}
    for i, row in clip_df.iterrows():

        new_start, new_stop = row['start'], row['end']
        clip_path = os.path.join(clip_dir, f'{i}.wav')

        duration = new_stop - new_start
        durations.append({i: duration})
        
        #  check length of the clip
        if new_stop - new_start < 1000: # less than 1 second
            print(f'Skipping {i} as it is less than 1 second {new_stop - new_start}')
            continue

        # check length of the clip
        if new_stop - new_start > 30000: # more than 30 seconds
            print(f'cutting {i} as it is more than 30 seconds {new_stop - new_start}')
            # cut the clip to 30 seconds
            new_stop = new_start + 30000

        # check if the clip is already created
        if not os.path.exists(clip_path):
            clip = audio[int(new_start*sr/1000):int(new_stop*sr/1000)]
            sf.write(clip_path, clip, sr)

        # get stuttering labels
        label_df['intersects'] = label_df.apply(lambda row: check_intersects(row, new_start, new_stop), axis=1)
        temp_df = label_df[label_df['intersects']]
        temp_df['clip_id'] = i
        # temp_df = temp_df.dropna()
        num_labels_per_clip.append(len(temp_df))

        
        
        # write to a text file for whisper
        with open(os.path.join(label_path,'txts', f'{i}.txt'), 'w') as f:
            temp_df = temp_df.sort_values(by='start')
            for row in temp_df.values:
                start_l = max(0, (row[2] - new_start)/1000)
                end_l = min(duration/1000, (row[3] - new_start)/1000)
                f.write(f'<{start_l}> <stutter> <{end_l}> ')
        
        
        # write to a text file for whisper
        with open(os.path.join(label_path,'yohos', f'{i}.txt'), 'w') as f:
            temp_df = temp_df.sort_values(by='start')
            for row in temp_df.values:
                start_l = max(0, (row[2] - new_start)/1000)
                end_l = min(duration/1000, (row[3] - new_start)/1000)
                l = ' '.join([str(x) for x in row[6:-2]])
                f.write(f'{start_l} {end_l} {l}\n')

        # # write the label to a json file for whispercnn
        # with open(os.path.join(label_path,'jsons', f'{i}.json'), 'w') as f:
        #     temp_df = temp_df.sort_values(by='start')
        #     if len(temp_df) == 0:
        #             continue
        #     for row in temp_df.values:
        #         start_l = max(0, row[2]/1000 - new_start/1000)
        #         end_l = min(new_stop/1000 - new_start/1000, row[3]/1000 - new_start/1000)
        #         f.write(f'{{"start": {start_l}, "end": {end_l}, "label": {row[4]}}}\n')
        

        # get new csv for classficaiton
        # temp_df = temp_df.sort_values()
        temp_df['start_rel'] = (temp_df['start'] - new_start)/1000
        temp_df['end_rel'] = (temp_df['end'] - new_start)/1000
        total_df = pd.concat([total_df, temp_df])


    # run inference on the pyannote.voice-activity-detection model
    #  aquire thread lock and run the pipeline.
    # with lock:
    #     output = pipeline(audio_path)

    # # combine the labels
    # vad_label = total_df.apply(lambda row: Segment(row[2]/1000, row[3]/1000), axis=1).tolist()
    # vad_label = vad_label+output.get_timeline().support().segments_list_
    # vad_label = merge_segments(vad_label)
    # # vad_label = clip_df.apply(lambda row: Segment(row['start']/1000, row['end']/1000), axis=1).tolist()
    # vad_labels[i] = vad_label

    # # write the vad labels to rttms file
    # with open(os.path.join(label_path, 'rttms', f'{file_id}.rttm'), 'w') as f:
    #     for i, segs in vad_labels.items():
    #         for seg in segs:
    #             f.write(f"SPEAKER {file_id} 1 {round(seg.start, 2)} {round(seg.duration, 2)} <NA> <NA> speech <NA> <NA>\n")

    # # write the vad labels to uems file
    # with open(os.path.join(label_path, 'uems', f"{file_id}.uem"), 'w') as f:
    #     f.write(f"{file_id} NA 0 {round(audio.shape[0]/sr,2)}\n")
        
    result['total_df'] = total_df
    result['num_labels'] = num_labels_per_clip
    result['duration'] = durations

    return result

def merge_segments(combined_list):
    # combined_list.sort(key=lambda seg: seg.start)
    merged_list = []
    current_segment = combined_list[0]

    for next_segment in combined_list[1:]:
        if current_segment.intersects(next_segment):
            current_segment = Segment(min(current_segment.start, next_segment.start), max(current_segment.end, next_segment.end))
        else:
            merged_list.append(current_segment)
            current_segment = next_segment

    merged_list.append(current_segment)
    return merged_list

def process_wav_files_transcript(wav_path, args, pipeline):

    audio_path = os.path.join(args.ds_path, wav_path)
    file_id = wav_path.split('.')[0]
    clip_path = os.path.join(args.output_path, args.anotator, 'clips', file_id)
    label_path = os.path.join(args.output_path, args.anotator, 'label', file_id)

    label_df = pd.read_csv(args.label_csv)
    label_df = label_df[label_df['media_file'] == file_id]
    label_df = label_df[label_df['annotator'] == args.anotator]
    
    # if split is test skip
    # if 'split' in label_df.columns and label_df['split'].values[0] == 'test':
    #     return wav_path, None

    clip_df = pd.read_csv(args.clip_csv)
    clip_df = clip_df[clip_df['media_file'] == file_id]
    clip_df = clip_df[clip_df['annotator']=='PAR']
    clip_df.reset_index(inplace=True)
    
    result = get_frames_from_transcript(audio_path, clip_path, label_path, label_df=label_df, clip_df=clip_df, pipeline=pipeline)

    return wav_path, result

def create_clips_from_transcript(args):
    
    # device = torch.device('cuda:4') if torch.cuda.is_available() else 'cpu'
    # pipeline = Pipeline.from_pretrained('pyannote/voice-activity-detection')
    # pipeline.to(device)
    pipeline = None
    
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for wav_path in tqdm(os.listdir(args.ds_path)):
            futures.append(executor.submit(process_wav_files_transcript, wav_path, args, pipeline))
        
        # Optionally, wait for all futures to complete
        for future in futures:
            path, res = future.result()
            if res is not None:
                results[path] = res

    total_df = pd.DataFrame()
    for key, value in results.items():
        sub_total_df = value['total_df']
        total_df = pd.concat([total_df, sub_total_df])
        value.pop('total_df')
    total_df.to_csv(args.output_path + 'total_df.csv', index=False)

    # plot the distribution of number of labels
    import matplotlib.pyplot as plt
    plt.hist([x['num_labels'] for x in results.values()], bins=20)
    plt.ylabel('Number of clips')
    plt.xlabel('Number of labels')
    plt.savefig(args.output_path + 'num_labels_dis.png')
    
    with open(args.output_path + 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    output_path = 'datasets/fluencybank/ds_temp/reading/'
    ds_path = 'datasets/fluencybank/wavs/reading/'
    label_csv = 'datasets/fluencybank/our_annotations/reading/csv/total_label_train.csv'
    clip_csv = 'datasets/fluencybank/our_annotations/reading/csv/transcripts.csv'
    anotator = 'A3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=output_path)
    parser.add_argument('--ds_path', type=str, default=ds_path)
    parser.add_argument('--label_csv', type=str, default=label_csv)
    parser.add_argument('--clip_csv', type=str, default=clip_csv)
    parser.add_argument('--anotator', type=str, default=anotator)
    parser.add_argument('--stride_duration', type=int, default=2)
    parser.add_argument('--clip_length', type=int, default=5)
    parser.add_argument('--method', type=str, default='constant_stride')
    args = parser.parse_args()

    if args.method == 'constant_stride':
        create_clips_strided(args)
    elif args.method == 'transcript':
        create_clips_from_transcript(args)
    else:
        print('Method not implemented')
        exit(1)

    
    


        