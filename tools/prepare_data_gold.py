import json
import pandas as pd
import argparse
from tqdm import tqdm
import os

from stutter.elan.elan import EafGroup
from stutter.utils.misc import get_eaf_files


def main(args):

    elanfiles = get_eaf_files(args.file_path)
    total_ann_df = pd.DataFrame()
    for key, value in tqdm(elanfiles.items()):
        eaf = EafGroup()
        eaf.initialize_from_file(value[0], 
                    #    sep28k_annotations=['datasets/fluencybank/fluencybank_labels.csv', 'datasets/fluencybank/fluencybank_episodes.csv']
                       )
        ann_df = eaf.to_dataframe(['Gold'])
        total_ann_df = pd.concat([total_ann_df, ann_df])


    total_ann_df['media_file'] = total_ann_df['media_file'].apply(lambda x: x.split('.')[0])
    total_ann_df = total_ann_df[['media_file', 'annotator', 'start', 'end', 'label']] #+list(total_ann_df.columns[5:-1])]
    total_ann_df.to_csv(args.csv_save_path+'labels_gold.csv', index=False)
    

if __name__ == '__main__':

    file_path = 'datasets/fluencybank/our_annotations/interview/Gold/'
    split_file = 'datasets/fluencybank/our_annotations/interview_split.json'
    combined_save_path = 'datasets/fluencybank/our_annotations/interview/combined/'
    csv_save_path = 'datasets/fluencybank/our_annotations/interview/csv/'

    parser = argparse.ArgumentParser(description='Merge elan files')
    parser.add_argument('--file_path', type=str, default=file_path, help='Path to elan files')
    parser.add_argument('--combine', type=bool, default=False, help='Combine the elan files')
    parser.add_argument('--combined_save_path', type=str, default=combined_save_path, help='Path to save the combined elan files')
    parser.add_argument('--csv_save_path', type=str, default=csv_save_path, help='Path to save the combined elan files')
    args = parser.parse_args()
    args.split_file = None
    main(args)