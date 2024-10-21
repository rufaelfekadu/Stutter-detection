import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import json
import argparse
sys.path.append('annotationmodeling/')
from agreement import InterAnnotatorAgreement
from stutter.utils.iaa import Vector, iou
from stutter.utils.annotation import LabelMap
from experiment_manager import DecompositionExperiment

import json
column_to_save = ['media_file', 'annotator', 'start', 'end', 'label']
# make dataframe from predictions


def vr_fromstr(x):
    x = x.split('-')
    return Vector(x[1])

def iou_dist(x, y):
    return 1 - iou(x[0], y[0])

def binary_distance(x, y):
    return 1 if x != y else 0

def binary_dist(x, y):
    label_map = LabelMap()
    x = label_map.labelfromstr(x)[:-2]
    y = label_map.labelfromstr(y)[:-2]
    return sum([1 if x[i] != y[i] else 0 for i in range(len(x))])

def iou_binary(x, y):
    alpha = 0.5
    return alpha * iou(x[0], y[0]) + (1 - alpha) * binary_dist(x[1], y[1])

class OurExperiment(DecompositionExperiment):
    def __init__(self, eval_fn, label_colname, item_colname, annotator_colname):
        super().__init__(eval_fn, label_colname, item_colname, uid_colname=annotator_colname)
        self.label_columns = ['SR','ISR','MUR','P','B', 'V', 'FG', 'HM', 'ME', 'T']
        self.label_map = LabelMap()
        # self.cluster_plotter = plot_ann
        self.dist_funs = [iou, binary_distance]
    
    def setup(self, grannodf,**kwargs):
        non_gold = grannodf[grannodf['annotator'] != 'Gold']
        gold = grannodf[grannodf['annotator'] == 'Gold']
        super().setup(annodf=grannodf, golddf=gold, c_gold_item='itemID', **kwargs)
        self.item_id_map = dict(zip(grannodf['itemID'], grannodf['item']))
        
    def save_aggregates(self, path='../datasets/'):   
        df_total = pd.DataFrame()
        column_to_save = ['media_file', 'annotator', 'start', 'end', 'label']+self.label_columns
        for aggtype in ['bau', 'mas', 'sad']:
            df = pd.DataFrame(getattr(self, f'{aggtype}_preds')).T
            df[['start', 'end']] = df[0].apply(lambda x: pd.Series([x.start, x.end]))
            df.rename(columns={1:'annotation'}, inplace=True)
            df['item'] = df.apply(lambda row: self.item_id_map[row.name], axis=1)
            df['annotator'] = aggtype
            df['media_file'] = df.apply(lambda row: row['itemID'].split('-')[0], axis=1)
            df_total = pd.concat([df_total, df], ignore_index=True, axis=0)
        breakpoint()
        df_total = pd.concat([df_total, self.grannodf])
        df_total.sort_values(by=['media_file', 'start'], inplace=True)
        
        df_total[self.label_map.labels] = df_total['annotation'].apply(lambda x: self.label_map.labelfromstr(x))

        df_total[column_to_save].to_csv(f'{path}/total_dataset_final.csv', index=False)

def normalize_vector(row):
    itemvr = Vector(row['newItemVR'])
    timevr = Vector(row['timevr'])
    return timevr.normlaize(itemvr)

def compute_iaa(grannodf, args):
    label_map = LabelMap()
    # compute iou for each class
    results = {}
    grannodf = grannodf[grannodf['annotator'] != 'Gold']
    grannodf[label_map.labels] = grannodf['labels'].apply(lambda x: pd.Series(label_map.labelfromstr(x[1])))
    for label in label_map.labels[:-3]:
        iaa = InterAnnotatorAgreement(grannodf, 
                                      item_colname=args.item_col, 
                                      uid_colname=args.annotator_col, 
                                      label_colname=label,
                                      distance_fn=binary_distance)
        iaa.setup()
        results[label] = {
            'alpha': iaa.get_krippendorff_alpha(),
            'ks': iaa.get_ks(),
            'sigma': iaa.get_sigma(use_kde=False),
        }
    iaa.setup()
    with open(f'{args.save_path}/iaa_results_binary.json', 'w') as f:
        json.dump(results, f, indent=4)

def aggregate(grannodf, args):
    our_exp = OurExperiment(eval_fn=eval(args.dist_fn), label_colname=args.label_col, 
                            item_colname=args.item_col, annotator_colname=args.annotator_col)
    
    our_exp.setup(grannodf, c_gold_label='gold')
    our_exp.train(dem_iter=500, mas_iter=100, masX_iter=0)
    our_exp.save_aggregates(path=args.save_path)
    our_exp.test(debug=False)

def inv_iou(x, y):
    return 1 - iou_dist(x, y)

def main(args):
    
    grannodf = pd.read_csv(args.gran_csv)
    grannodf.dropna(subset=['annotation'], inplace=True)
    grannodf['annotatorID'] = grannodf['annotator'].astype('category').cat.codes
    grannodf['itemID'] = grannodf['item'].astype('category').cat.codes
    grannodf['labels'] = grannodf.apply(lambda row: (Vector((row['start'], row['end'])), row['annotation']), axis=1)
    grannodf['gold'] = grannodf.apply(lambda row: (Vector((row['gold_start'], row['gold_end'])), row['gold_label']), axis=1)
    grannodf['goldTimeVR'] = grannodf['gold'].apply(lambda x: x[0])
    
    # grannodf = grannodf.groupby(args.item_col).filter(lambda x: len(x) > 1).reset_index(drop=True)
    # compute_iaa(grannodf[['annotatorID', 'itemID', 'labels', 'annotator']], args)
    aggregate(grannodf, args)

    
if __name__ == '__main__':
    grannodf_path = 'datasets/stutter-bank/gran_total.csv'
    save_path = 'datasets/stutter-bank'
    item_col = 'itemID'
    annotator_col = 'annotatorID'
    label_col = 'labels'
    dist_fn = 'iou_binary'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gran_csv', type=str, default=grannodf_path, help='Path to the granular annotation dataframe')
    parser.add_argument('--save_path', type=str, default=save_path, help='Path to save the IAA results')
    parser.add_argument('--item_col', type=str, default=item_col, help='Column name for the item')
    parser.add_argument('--annotator_col', type=str, default=annotator_col, help='Column name for the annotator')
    parser.add_argument('--label_col', type=str, default=label_col, help='Column name for the label')
    parser.add_argument('--dist_fn', type=str, default=dist_fn, help='Distance function to use for IAA')
    args = parser.parse_args()

    main(args)
