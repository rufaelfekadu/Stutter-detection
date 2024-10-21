import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.patches import Rectangle
import sys
import argparse
from stutter.utils.iaa import Vector, unionize_vectorrange_sequence, fragment_by_overlaps, merge_intervals, decomp_fn



def main(args):

    total_df = pd.read_csv(args.label_csv)
    total_df['timevr'] = total_df.apply(lambda row: Vector((row['start'], row['end'])), axis=1)
    total_df['labels'] = total_df.apply(lambda row: (row['timevr'], row['label']), axis=1)
    to_save = ['media_file', 'item', 'annotator', 'start', 'end', 'annotation', 'gold_start', 'gold_end', 'gold_label']

    print(f"Total number of annotations before fragment: {len(total_df)}")
    grannodf = fragment_by_overlaps(total_df,
                                    uid_colname='annotator',
                                    item_colname='media_file',
                                    label_colname='labels',
                                    decomp_fn=decomp_fn,
                                    dist_fn=None,
                                    gold_df=total_df[total_df['annotator']=='Gold'])
    
    grannodf = grannodf.groupby('origItemID', group_keys=False).apply(lambda x: merge_intervals(x)).reset_index(drop=True)
    grannodf['newItemID'] = grannodf.apply(lambda row: F"{row['origItemID']}-{row['newItemVR']}", axis=1)
    grannodf.drop_duplicates(['newItemID','labels', 'annotator'], inplace=True)
    print(f"Total number of annotations after fragment: {len(grannodf)}")

    grannodf['start'] = grannodf.apply(lambda row: row['labels'][0].start, axis=1)
    grannodf['end'] = grannodf.apply(lambda row: row['labels'][0].end, axis=1)
    grannodf['annotation'] = grannodf.apply(lambda row: row['labels'][1], axis=1)
    grannodf['gold_start'] = grannodf.apply(lambda row: row['gold'][0].start if row['gold'] is not None else None, axis=1)
    grannodf['gold_end'] = grannodf.apply(lambda row: row['gold'][0].end if row['gold'] is not None else None, axis=1)
    grannodf['gold_label'] = grannodf.apply(lambda row: row['gold'][1] if row['gold'] is not None else None, axis=1)


    grannodf.sort_values(by=['origItemID', 'start', 'annotator', 'newItemID'], inplace=True)
    grannodf.rename(columns={'origItemID':'media_file',
                             'newItemID':'item',
                             'newItemVR':'item_time_range',
                             'goldTimeVR': 'gold_time_range'
                             }, inplace=True)
    
    grannodf[to_save].to_csv(args.save_path, index=False)

if __name__ == "__main__":

    label_csv = 'datasets/fluencybank/our_annotations/interview/csv/labels.csv'
    save_path = 'datasets/fluencybank/our_annotations/interview/csv/gran_data_total.csv'
    parser = argparse.ArgumentParser(description='Create granular data')
    parser.add_argument('--label_csv', type=str, default=label_csv, help='Path to the label csv')
    parser.add_argument('--save_path', type=str, default=save_path, help='Path to the output csv')
    args = parser.parse_args()

    main(args)
    

