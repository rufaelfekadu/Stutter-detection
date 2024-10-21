
import json
import os
from glob  import glob

split_file = 'datasets/fluencybank/our_annotations/interview_split.json'
lst_path = 'datasets/fluencybank/ds_dia/interview'
wav_path = 'datasets/fluencybank/ds_dia/interview/clips/audio'

with open(split_file, 'r') as f:
    splits = json.load(f)

clips = glob(wav_path + '/*.wav')
train_lst = open(os.path.join(lst_path, 'train.lst'), 'w')
test_lst = open(os.path.join(lst_path, 'test.lst'), 'w')
val_lst = open(os.path.join(lst_path, 'val.lst'), 'w')


for clip in clips:
    clip_name = os.path.basename(clip).replace('.wav', '')
    media_name = clip_name.split('_')[0]
    if media_name in splits['train']:
        train_lst.write(clip_name + '\n')
    elif media_name in splits['test']:
        test_lst.write(clip_name + '\n')
    elif media_name in splits['val']:
        val_lst.write(clip_name + '\n')
    else:
        print(f'Clip {clip_name} not found in splits')
        break

train_lst.close()
test_lst.close()
val_lst.close()