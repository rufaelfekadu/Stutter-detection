
import pympi   
import os
import re
import sys
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from stutter.elan.elan import EafGroup
from stutter.utils.misc import get_eaf_files



def main(args):

    elanfiles = get_eaf_files(args.file_path)
    total_df = pd.DataFrame()

    for _, value in tqdm(elanfiles.items()):
        if len(value)==1:
            continue
        # print(f'Processing {key}')
        eaf = EafGroup(value, 
                    #    sep28k_files=['datasets/fluencybank/fluencybank_labels.csv', 'datasets/fluencybank/fluencybank_episodes.csv']
                       )
        # eaf.to_file(args.combined_save_path+key.replace('.eaf','_combined.eaf'))
        temp_df = eaf.to_dataframe()
        total_df = pd.concat([total_df, temp_df])

    total_df.to_csv(args.csv_save_path, index=False)
    

if __name__ == '__main__':

    file_path = 'datasets/fluencybank/new_annotations/reading/train/'
    combined_save_path = 'outputs/fluencybank/combined_files/reading/'
    csv_save_path = 'outputs/fluencybank/new_annotations/reading_train.csv'

    parser = argparse.ArgumentParser(description='Merge elan files')
    parser.add_argument('--file_path', type=str, default=file_path, help='Path to elan files')
    parser.add_argument('--combined_save_path', type=str, default=combined_save_path, help='Path to save the combined elan files')
    parser.add_argument('--csv_save_path', type=str, default=csv_save_path, help='Path to save the combined elan files')
    args = parser.parse_args()

    main(args)