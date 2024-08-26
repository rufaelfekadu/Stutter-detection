import json
import pandas as pd
import argparse
from tqdm import tqdm

from stutter.elan.elan import EafGroup
from stutter.utils.misc import get_eaf_files


def main(args):

    elanfiles = get_eaf_files(args.file_path)

    total_ann_df = pd.DataFrame()
    total_trans_df = pd.DataFrame()

    for key, value in tqdm(elanfiles.items()):
        if len(value)==1:
            continue

        # print(f'Processing {key}')
        eaf = EafGroup()
        eaf.initialize_from_files(value, 
                    #    sep28k_annotations=['datasets/fluencybank/fluencybank_labels.csv', 'datasets/fluencybank/fluencybank_episodes.csv']
                       )
        if args.combine:
            eaf.to_file(args.combined_save_path+key)

        
        ann_df = eaf.to_dataframe()
        total_ann_df = pd.concat([total_ann_df, ann_df])

        trans_df = eaf.to_dataframe(tiers=['PAR','INV'])
        total_trans_df = pd.concat([total_trans_df,trans_df])

    if args.split_file:
        breakpoint()
        #  read split json file
        with open(args.split_file) as f:
            split = json.load(f)
        test_files = split['test']
        total_ann_df['split'] = 'train'
        total_ann_df.loc[total_ann_df['media_file'].isin(test_files), 'split'] = 'test'
        # reorder the columns 
        total_ann_df = total_ann_df[['media_file', 'annotator', 'start', 'end', 'label', 'split']+list(total_ann_df.columns[5:-1])]

    total_ann_df.to_csv(args.csv_save_path+'labels.csv', index=False)
    total_trans_df = total_trans_df[['media_file', 'annotator', 'start', 'end', 'label']]
    total_trans_df.to_csv(args.csv_save_path+'transcripts.csv', index=False)
    

if __name__ == '__main__':

    file_path = 'datasets/fluencybank/new_annotations/interview/elan/'
    split_file = 'datasets/fluencybank/new_annotations/reading_split.json'
    combined_save_path = 'datasets/fluencybank/new_annotations/interview/combined/'
    csv_save_path = 'datasets/fluencybank/new_annotations/interview/csv/'

    parser = argparse.ArgumentParser(description='Merge elan files')
    parser.add_argument('--file_path', type=str, default=file_path, help='Path to elan files')
    parser.add_argument('--combine', type=bool, default=False, help='Combine the elan files')
    parser.add_argument('--combined_save_path', type=str, default=combined_save_path, help='Path to save the combined elan files')
    parser.add_argument('--csv_save_path', type=str, default=csv_save_path, help='Path to save the combined elan files')
    parser.add_argument('--split_file', type=str, default=split_file, help='Path to split file')
    args = parser.parse_args()
    args.split_file = None
    main(args)