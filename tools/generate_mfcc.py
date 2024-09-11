import os
import librosa
import numpy as np
import pandas as pd

def get_max_length(clips_folder):
    max_length = 0
    for root, dirs, files in os.walk(clips_folder):
        for file in files:
            y, sr = librosa.load(os.path.join(root, file), sr=None)
            max_length = max(max_length, (len(y)/sr)*1000)

    return max_length

def extract_mfcc(row, n_mfcc=40, n_fft=2048, max_length=4):
    y, sr = librosa.load(row['audio_file_path']+'.wav', sr=None)
    y = librosa.util.fix_length(y, size=int((max_length/1000)*sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    return mfcc

def main():


    data_dir = 'datasets/fluencybank/ds_label/reading/A1'
    gold_dir = 'datasets/fluencybank/ds_label/reading/Gold/'
    clips_test_folder = "datasets/fluencybank/ds_label/reading/Gold/clips/audio"
    clips_train_annotated_folder = "datasets/fluencybank/ds_label/reading/A1/clips/audio"

    max_length_train = get_max_length(clips_train_annotated_folder)
    max_length_test = get_max_length(clips_test_folder)
    max_length = max(max_length_train, max_length_test)

    df = pd.read_csv(os.path.join(data_dir, 'total_label.csv'))
    df_A1 = df[df['annotator'] == 'A1']
    df_A1['audio_file_path'] = df_A1.apply(lambda row: os.path.join(clips_train_annotated_folder, row['media_file'], row['file_name']), axis=1) 
    df_A1['mfcc'] = df_A1.apply(lambda row: extract_mfcc(row, max_length=max_length), axis=1)
    df_A1.to_csv(os.path.join(data_dir, 'mfcc_train_A1.csv'), index=False)

    df = pd.read_csv(os.path.join(gold_dir, 'total_label.csv'))
    df_gold = df[df['annotator'] == 'Gold']
    df_gold['audio_file_path'] = df_gold.apply(lambda row: os.path.join(clips_test_folder, row['media_file'], row['file_name']), axis=1)
    df_gold['mfcc'] = df_gold.apply(lambda row: extract_mfcc(row, max_length=max_length), axis=1)
    df_gold.to_csv(os.path.join(data_dir, 'mfcc_test_gold.csv'), index=False)

if __name__ == '__main__':
    main()