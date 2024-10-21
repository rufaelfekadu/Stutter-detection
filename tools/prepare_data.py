import json
import pandas as pd
import argparse
from tqdm import tqdm
import os

from stutter.elan.elan import EafGroup
from stutter.utils.misc import get_eaf_files

to_save = ['media_file', 'annotator', 'start', 'end', 'label']

def main(args):

    elanfiles = get_eaf_files(args.file_path)

    total_ann_df = pd.DataFrame()
    total_trans_df = pd.DataFrame()
    
    for key, value in elanfiles.items():
        

        is_gold = [x for x in value if 'Gold' in x]
        if len(is_gold) > 0:
            eaf_gold = EafGroup()
            eaf_gold.initialize_from_file(is_gold[0])
            ann_df = eaf_gold.to_dataframe(['INV', 'PAR']+eaf_gold.annotators)
            total_ann_df = pd.concat([total_ann_df, ann_df])
            continue
        
        if len(value)<3:
            annotators = [v.split('/')[-2] for v in value]
            print(f'Not enough annotators for {key} - {annotators}')
            continue

        print(f'Processing {key}')
        eaf = EafGroup()
        eaf.initialize_from_files(value, 
                    #    sep28k_annotations=['datasets/fluencybank/fluencybank_labels.csv', 'datasets/fluencybank/fluencybank_episodes.csv']
                       )
        if args.combine and not os.path.exists(args.save_path+'/combined/'+key):
            os.makedirs(args.save_path+'/combined/', exist_ok=True)
            eaf.to_file(args.save_path+'/combined/'+key)

        
        ann_df = eaf.to_dataframe(tiers=['PAR','INV']+eaf.annotators)
        total_ann_df = pd.concat([total_ann_df, ann_df])

        # trans_df = eaf.to_dataframe(tiers=['PAR','INV'])
        # total_trans_df = pd.concat([total_trans_df,trans_df])

    total_ann_df['media_file'] = total_ann_df['media_file'].apply(lambda x: x.split('.')[0])
    # total_trans_df['media_file'] = total_trans_df['media_file'].apply(lambda x: x.split('.')[0])

    os.makedirs(args.save_path+'/csv/', exist_ok=True)
    total_ann_df[~total_ann_df['annotator'].isin(["PAR", "INV"])][to_save].to_csv(args.save_path+'/csv/labels.csv', index=False)
    total_ann_df[total_ann_df['annotator'].isin(["PAR", "INV"])][to_save].to_csv(args.save_path+'/csv/transcripts.csv', index=False)
    

if __name__ == '__main__':

    file_path = 'datasets/fluencybank/our_annotations/interview/'
    save_path = 'datasets/fluencybank/our_annotations/interview/'

    parser = argparse.ArgumentParser(description='Merge elan files')
    parser.add_argument('--file_path', type=str, default=file_path, help='Path to elan files')
    parser.add_argument('--save_path', type=str, default=save_path, help='Path to save the combined elan files')
    parser.add_argument('--combine', action='store_true', help='Combine the elan files')
    args = parser.parse_args()
    args.split_file = None
    main(args)