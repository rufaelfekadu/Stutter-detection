
import sys
sys.path.append('../annotationmodeling')
from agreement import InterAnnotatorAgreement

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import euclidean_distances
from collections import defaultdict
import numpy as np
import pandas as pd
import copy


class Vector(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.centroid = (start + end) / 2
        
    def __sub__(self, other):
        return Vector(self.end, other.start)
    
    def intersects(self, other):
        # check if two vectors intersect
        return self.start <= other.end and other.start <= self.end
    
    def __len__(self):
        return self.end - self.start
    
    def __str__(self):
        return f'({self.start}, {self.end})'
    
def unionize_vectorrange_sequence(vectorranges):
    min_s = min([vr.start for vr in vectorranges])
    max_e = max([vr.end for vr in vectorranges])
    return Vector(min_s, max_e)

def fragment_by_overlaps(annodf, uid_colname, item_colname, label_colname, decomp_fn, dist_fn=None, gold_df=None):
    resultdfs = []
    for item_id in annodf[item_colname].unique():
        idf = annodf[annodf[item_colname] == item_id]
        vectorranges = [vas[0] for vas in idf[label_colname]]

        regions = decomp_fn(vectorranges, dist_fn=dist_fn)
        origItemID = []
        newItemID = []
        newItemVR = []
        uid = []
        label = []
        goldtimevr = []
        goldlabel = []
        for region in regions:
            for i, row in idf.iterrows():
                if region.intersects(row[label_colname][0]):
                    origItemID.append(item_id)
                    newItemID.append(F"{item_id}-{region}")
                    newItemVR.append(region)
                    uid.append(row[uid_colname])
                    label.append(row[label_colname])
                    if gold_df is not None:
                        # get the gold label that intersects with the region
                        gold_timevr = [vr[0] for vr in gold_df[(gold_df[item_colname] == item_id)][label_colname] if region.intersects(vr[0])]
                        gold_timevr = gold_timevr[0] if len(gold_timevr) > 0 else None
                        goldtimevr.append(gold_timevr)

                        gold_label = [lbl for lbl in gold_df[(gold_df[item_colname] == item_id)][label_colname] if region.intersects(lbl[0])]
                        gold_label = gold_label[0] if len(gold_label) > 0 else None
                        goldlabel.append(gold_label)
                    else:
                        goldtimevr.append(None)
        resultdfs.append(pd.DataFrame({"origItemID":origItemID, "newItemID":newItemID, "newItemVR":newItemVR, uid_colname:uid, label_colname:label, "goldTimeVR":goldtimevr, "gold":goldlabel}))
    return pd.concat(resultdfs)

def decomp_fn(vectorranges, use_centroids=False, dist_fn=None):

    if use_centroids:
        centroids = np.array([vr.centroid for vr in vectorranges]).reshape(-1, 1)
        # dists = euclidean_distances(centroids)
        # mean_dist = np.std(dists)
        mean_dist = 1000
        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=mean_dist)
        clustering.fit(centroids)
    else:
        dists = np.array([[1 - iou(a, b) for a in vectorranges] for b in vectorranges])
        # mean_dist = np.std(dists)
        clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=1000,
                                             linkage="average")
        clustering.fit(dists)
        
    labels = clustering.labels_
    labeldict = defaultdict(list)
    for i, label in enumerate(labels):
        labeldict[label].append(i)
    result = []
    for indices in labeldict.values():
        uv = unionize_vectorrange_sequence(np.array(vectorranges)[np.array(indices)])
        result.append(uv)
    return result

#  define Distance functions
def iou(a:Vector, b: Vector):
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = len(a) + len(b) - intersection
    return intersection / union

def precision(a, b):
    true_positive = a==b
    false_positive = a!=b
    return true_positive / (true_positive + false_positive)

def recall(a, b):
    true_positive = a==b
    false_negative = a!=b
    return true_positive / (true_positive + false_negative)

def f1_score(a, b):
    p = precision(a, b)
    r = recall(a, b)
    return 2 * (p * r) / (p + r)

def iou_multiple(vas, vbs):
    return np.mean([iou(va, vb) for va in vas for vb in vbs])

def normalized_hamming_distance(label1, label2):
    differences = sum(1 for l1, l2 in zip(label1[1], label2[1]) if l1 != l2)
    hamming_distance = differences / len(label1)
    return hamming_distance

def normalized_ordinal_distance(ordinal1, ordinal2, max_value=3):
    ordinal_distance = abs(ordinal1 - ordinal2) / max_value
    return ordinal_distance

def normalized_ordinal_distance_multiple(ordinal1s, ordinal2s, max_value):
    return np.mean([normalized_ordinal_distance(o1, o2) for o1 in ordinal1s for o2 in ordinal2s])

def normalized_hamming_distance_multiple(labels1, labels2):
    return np.mean([normalized_hamming_distance(l1, l2) for l1 in labels1 for l2 in labels2])

def score(row1, row2, weights=(0.33, 0.33, 0.34)):
    # iou_score = iou(row1[0], row2[0])
    hamming_score = normalized_hamming_distance(row1[1], row2[1])
    ordinal_score = normalized_ordinal_distance(row1[2][0], row2[2][0], 3)
    return sum([hamming_score * weights[1], ordinal_score * weights[2]])

def score_multiple(row1, row2):
    return np.mean([score(r1, r2) for r1 in row1 for r2 in row2])

def simple_binary_distance(a, b):
    return a != b