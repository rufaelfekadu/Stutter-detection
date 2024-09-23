import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import json
import argparse
sys.path.append('annotationmodeling/')
from agreement import InterAnnotatorAgreement

class Vector(object):

    def __init__(self, se, sep=','):
        if isinstance(se, str):
            self.__init_from_str__(se, sep)
        else:
            start, end = se
            self.start = start
            self.end = end
            self.centroid = (start + end) / 2

    def __init_from_str__(self, s, sep=','):
        s = s.strip('()')
        start, end = s.split(sep)
        self.start = int(start)
        self.end = int(end)
        self.centroid = (self.start + self.end) / 2

    def __sub__(self, other):
        return Vector((self.end, other.start))
    
    def normlaize(self, ref):
        new_start = self.start - ref.start
        new_end = new_start+(self.end-self.start)
        if new_end<0 or new_start<0:
            breakpoint()
            raise ValueError('Normalization failed')
        return Vector((new_start, new_end))
    
    def __gt__(self, other):
        if isinstance(other, Vector):
            return self.start > other.start
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, Vector):
            return self.start < other.start
        return NotImplemented
        
    def intersects(self, other):
        # check if two vectors intersect
        return self.start <= other.end and other.start <= self.end
        
    def __len__(self):
        return self.end - self.start
    
    def __str__(self):
        return f'({self.start}, {self.end})'

def normalize_vector(row):
    itemvr = Vector(row['newItemVR'])
    timevr = Vector(row['timevr'])
    return timevr.normlaize(itemvr)


def compute_iaa(grannodf, args):
    
    grannodf.rename(columns={args.annotator_col:'annotator',
                             args.item_col:'item',
                             args.label_col:'label',}, 
                             inplace=True)
    grannodf['item'] = grannodf['item'].astype('category').cat.codes
    grannodf['annotator'] = grannodf['annotator'].astype('category').cat.codes
    # drop items with only one annotation
    grannodf = grannodf.groupby('item').filter(lambda x: len(x) > 1)
    iaa = InterAnnotatorAgreement(annodf=grannodf, distance_fn=args.dist_fn,
                                   label_colname='label', item_colname='item', uid_colname='annotator')
    iaa.setup()
    results = {
        'alpha': iaa.get_krippendorff_alpha(),
        'ks': iaa.get_ks(),
        'sigma': iaa.get_sigma(use_kde=False),
    }

    return results

def iou(a:Vector, b: Vector):
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = len(a) + len(b) - intersection
    return intersection / union

def inv_iou(x, y):
    return 1 - iou(x, y)

def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))

def euc_dist(x,y):
    if isinstance(x, list):
        return np.mean([rmse(a,b) for a,b in zip(x,y)])
    return rmse(x,y)

def main(args):
    
    grannodf = pd.read_csv(args.grannodf)
    # grannodf['timevr'] = grannodf.apply(lambda row: normalize_vector(row), axis=1)
    grannodf['timevr'] = grannodf['timevr'].apply(lambda x: Vector(x))
    
    # for each media_file, calculate the IAA and sort the results
    results = {}
    media_file = '26f'
    df = grannodf[grannodf['origItemID'] == media_file]
    # df['timevr'] = df.apply(lambda row: normalize_vector(row), axis=1)
    result = compute_iaa(df, args)
    breakpoint()
    # for  media_file, df in tqdm(grannodf.groupby('origItemID')):
    #     result = compute_iaa(df, args)
    #     results[media_file] = result

    # write the results to a file as json with indent=4
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)
    


if __name__ == '__main__':
    grannodf_path = 'datasets/fluencybank/our_annotations/interview/csv/gran_data_temp_1.csv'
    save_path = 'datasets/fluencybank/our_annotations/interview/csv/iaa.json'
    item_col = 'newItemID'
    annotator_col = 'annotator'
    label_col = 'timevr'
    dist_fn = inv_iou

    parser = argparse.ArgumentParser()
    parser.add_argument('--grannodf', type=str, default=grannodf_path, help='Path to the granular annotation dataframe')
    parser.add_argument('--save_path', type=str, default=save_path, help='Path to save the IAA results')
    parser.add_argument('--item_col', type=str, default=item_col, help='Column name for the item')
    parser.add_argument('--annotator_col', type=str, default=annotator_col, help='Column name for the annotator')
    parser.add_argument('--label_col', type=str, default=label_col, help='Column name for the label')
    parser.add_argument('--dist_fn', type=str, default=dist_fn, help='Distance function to use for IAA')
    args = parser.parse_args()

    main(args)
