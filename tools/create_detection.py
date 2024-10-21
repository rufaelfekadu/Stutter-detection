import numpy as np
import pandas as pd
import os
from stutter.utils.annotation import LabelMap
import argparse

def main(args):
    label_map = LabelMap()
    total_df = pd.read_csv(args.csv_path)
    total_df = total_df[~total_df['label'].isna()]
    # iterate though all annotators
    for group, df in total_df.groupby('annotator'):
        # iterate through all items
        for item, item_df in df.groupby('media_file'):
            if item_df.empty:
                continue
            out_path = f'{args.out_dir}/{group}/{item}.txt'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, 'w') as f:
                for _, row in item_df.iterrows():
                    start = row['start']/1000
                    end = row['end']/1000
                    label = label_map.get_all(row['label'])
                    for l in label:
                        f.write(f'{start},{end},{label_map.description[l]}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare data for stutter detection')
    parser.add_argument('--csv_path', type=str, default='datasets/fluencybank/our_annotations/interview/csv/labels_gold.csv', help='Path to the csv file')
    parser.add_argument('--out_dir', type=str, default='datasets/fluencybank/our_annotations/interview/txt/', help='Path to save the txt files')
    args = parser.parse_args()
    print(args)
    main(args)