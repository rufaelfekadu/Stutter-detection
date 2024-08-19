from pympi import Elan
import argparse
import os

from stutter.utils.misc import get_eaf_files
from stutter.utils.annotation import LabelMap
from stutter.elan.elan import EafGroup
import numpy as np
import pandas as pd
import sys
from collections import Counter

def calc_dist(x,y):
    return sum([np.abs(a-b) for a,b in zip(x,y)])/len(x)

def select_majority_values(lists):
    # Transpose the list of lists to group elements by their positions
    transposed = list(zip(*lists))
    
    # Find the majority element for each group
    majority_values = []
    for group in transposed:
        # Get the most common element
        most_common = Counter(group).most_common(1)[0][0]
        majority_values.append(most_common)
    
    return majority_values

def add_gold_tier(eaf, tiers, tiernew, gapt=10, sep='/', safe=True, label_map=LabelMap()):

    if tiernew is None:
        tiernew = u'{}_merged'.format('_'.join(tiers))
    eaf.add_tier(tiernew)
    aa = [(sys.maxsize, sys.maxsize, None)] + sorted((
        a for t in tiers for a in eaf.get_annotation_data_for_tier(t)),
        reverse=True)
    l = None
    while aa:
        begin, end, value = aa.pop()
        if l is None:
            l = [begin, end, [value]]
        elif begin - l[1] >= gapt:
            if not safe or l[1] > l[0]:
                labels = [label_map.labelfromstr(l[2][i])[:6] for i in range(len(l[2]))]
                dist = np.array([calc_dist(a,b) for a in labels for b in labels])>0.16
                    
                if  np.count_nonzero(dist==0)>(len(dist)//2) and len(labels)>1:
                    labels = [label_map.labelfromstr(l[2][i])for i in range(len(l[2]))]
                    labels = select_majority_values(labels)
                    eaf.add_annotation(tiernew, l[0], l[1], label_map.strfromlabel(labels))
            l = [begin, end, [value]]
        else:
            if end > l[1]:
                l[1] = end
            l[2].append(value)

    try:
        for annotation in eaf.get_annotation_data_for_tier('agreement'):
            eaf.add_annotation('Gold', annotation[0], annotation[1], annotation[2])
        for annotation in eaf.get_annotation_data_for_tier('disagreement'):
            if "/" in annotation[2]:
                continue
            eaf.add_annotation('Gold', annotation[0], annotation[1], annotation[2])
    except:
        print(f'Tier agreement not found in {eaf}')

    return tiernew

def to_df(eaf, tiers, label_map=LabelMap()):
    df = pd.DataFrame()
    for tier in tiers:
        temp_df = pd.DataFrame()
        annotations = eaf.get_annotation_data_for_tier(tier)
        temp_df['anotator'] = [tier]*len(annotations)
        temp_df['media_file'] = [None]*len(annotations)
        temp_df = pd.concat([temp_df, pd.DataFrame(annotations, columns=['start', 'end', 'label'])], axis=1)
        label = temp_df['label'].apply(lambda x: label_map.labelfromstr(x))
        for i in range(len(label_map.labels)):
            temp_df[label_map.labels[i]] = label.apply(lambda x: x[i])
        df = pd.concat([df, temp_df]).reset_index(drop=True)
    return df['media_file anotator start end'.split()+label_map.labels]


def main(args):

    eaf_files = get_eaf_files(args.eaf_dir)
    label_map = LabelMap()
    total_df = pd.DataFrame()
    for k, v in eaf_files.items():
        tier_names=['A1', 'A2', 'A3']
        eaf = Elan.Eaf(v[0])
        gold = add_gold_tier(eaf, tiers=tier_names, tiernew='Gold', label_map=label_map)
        # eaf.to_file(os.path.join(args.combined_dir, k.replace('.eaf', '_gold.eaf')))
        df = to_df(eaf, tier_names+['Gold'])
        df['media_file'] = k.replace('.eaf', '.wav')
        total_df = pd.concat([total_df, df])
    
    total_df.to_csv(os.path.join(args.csv_dir, 'reading_test.csv'), index=False)
    # total_df[total_df['anotator']=='Gold'].to_csv(os.path.join(args.csv_dir, 'reading_gold.csv'), index=False)
    # total_df[total_df['anotator']=='A1'].to_csv(os.path.join(args.csv_dir, 'reading_A1.csv'), index=False)
    # total_df[total_df['anotator']=='A2'].to_csv(os.path.join(args.csv_dir, 'reading_A2.csv'), index=False)
    # total_df[total_df['anotator']=='A3'].to_csv(os.path.join(args.csv_dir, 'reading_A3.csv'), index=False)

    
        
        

if __name__ == "__main__":

    eaf_dir = 'datasets/fluencybank/gold'
    combined_dir = 'datasets/fluencybank/combined_files'
    csv_dir = 'datasets/fluencybank/csv/test/'

    parser = argparse.ArgumentParser(description='Get gold annotations from EAF files')
    parser.add_argument('--eaf_dir', type=str, default=eaf_dir, help='Directory containing EAF files')
    parser.add_argument('--combined_dir', type=str, default=combined_dir, help='Output directory')
    parser.add_argument('--csv_dir', type=str, default=csv_dir, help='Output directory')
    args = parser.parse_args()

    main(args)