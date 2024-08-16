
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import pandas as pd
import re
import json
from stutter.utils.annotation import LabelMap, clean_element
import matplotlib.pyplot as plt
import sys
import numpy as np
import pympi

def jaccard_similarity(vec1, vec2):
    intersection = np.sum(np.minimum(vec1, vec2))
    union = np.sum(np.maximum(vec1, vec2))
    return intersection / union

def hamming_distance(vec1, vec2):
    return np.sum(vec1 != vec2)/len(vec1)

class EafGroup(pympi.Elan.Eaf):

    tier_names = ['Stuttering moments', 'stuttering moments', 'stuttering_moments', 'Stuttering_Moments', 'stuttering moment', 'Stuttering moment', 'stuttering_moment', 'Stuttering_Moment', 'gold']

    def __init__(self, elan_files, sep28k_files=None, tier_names=None, gold_available=False):
        super().__init__()

        if tier_names:
            self.tier_names = tier_names
        
        self.eaf_files = []
        self.annotation_counts = {}
        self.label_map = LabelMap()
        self.annotators = []
        self.to_merge = []
        for i,elan_file in enumerate(elan_files):
            if i==0:
                # get the media file
                self.media_file = os.path.basename(elan_file).replace('.eaf','.wav')
                self.add_linked_file(self.media_file, mimetype='audio/wav', relpath=self.media_file)
                self.add_linked_file(self.media_file.replace('.wav','.mp4'), mimetype='video/mp4', relpath=self.media_file.replace('.wav','.mp4'))
            else:
                assert self.media_file == os.path.basename(elan_file).replace('.eaf','.wav'), 'Media file should be the same for all elan files'

            annotator_id = re.search(r'/(A\d+|gold)', elan_file).group(1)
            if annotator_id == 'gold':
                self.copy_gold_tier(pympi.Elan.Eaf(elan_file), 'gold')
                continue
            self.annotators.append(annotator_id)
            # print(f'Processing {annotator_id}')
            
            try:
                eaf_file = pympi.Elan.Eaf(elan_file)
                tier_name = [tier for tier in self.tier_names if tier in eaf_file.get_tier_names()][0]
                self.copy_to_new_tier(eaf_file, tier_name=tier_name, prefix=annotator_id)
                self.annotation_counts[annotator_id] = len(eaf_file.get_annotation_data_for_tier(tier_name))
                self.eaf_files.append(eaf_file)
            except Exception as e:
                print(f'No stuttering tier found in {elan_file} {e}')

        self.to_merge = [i for i in self.tiers.keys() if any(tier_name in i for tier_name in self.tier_names)]

        # _, dissagreement_count = self.merge_tiers_by_dist(self.to_merge, 'disagreement')

        if sep28k_files:
            self.add_sep28k(sep28k_files)

        self.remove_tier('default')
    # def copy_gold_tier(self, eaf, tier_name):
    #     # tier_dict = {}
    #     # tier_dict['TIER_ID'] = 'gold'
    #     # tier_dict['LINGUISTIC_TYPE_REF'] = 'default-lt'
    #     # self.add_tier('gold', tier_dict=tier_dict)
    #     self.merge_tiers_by_dist(self.to_merge, 'gold')
    #     try:
    #         for annotation in eaf.get_annotation_data_for_tier('agreement'):
    #             self.add_annotation('gold', annotation[0], annotation[1], annotation[2])
    #         for annotation in eaf.get_annotation_data_for_tier('disagreement'):
    #             if "/" in annotation[2]:
    #                 continue
    #             self.add_annotation('gold', annotation[0], annotation[1], annotation[2])
            
    #     except:
    #         print(f'Tier {'agreement'} not found in {eaf}')
        
    def copy_to_new_tier(self, eaf, tier_name, prefix):
        try:
            tier_dict = {}
            tier_dict['TIER_ID'] = prefix 
            tier_dict['LINGUISTIC_TYPE_REF'] = 'default-lt'
            self.add_tier(prefix, tier_dict=tier_dict)
            for annotation in eaf.get_annotation_data_for_tier(tier_name):
                self.add_annotation(prefix , annotation[0], annotation[1], annotation[2])
        except Exception as e:
            print(f'Tier {tier_name} not found in {eaf}: {e}')
            
    def add_disagreement_tier(self, tiers, tiernew, gapt=10, sep='/', safe=True, dist_threshold=0.16):
        if tiernew is None:
            tiernew = u'{}_merged'.format('_'.join(tiers))
        self.add_tier(tiernew)
        aa = [(sys.maxsize, sys.maxsize, None)] + sorted((
            a for t in tiers for a in self.get_annotation_data_for_tier(t)),
            reverse=True)
        l = None
        disagreement_count = 0
        dists = []
        while aa:
            begin, end, value = aa.pop()
            if l is None:
                l = [begin, end, [value]]
            elif begin - l[1] >= gapt:
                if not safe or l[1] > l[0]:
                    labels = [self.label_map.labelfromstr(l[2][i])[:6] for i in range(len(l[2]))]
                    dist = np.array([self.calc_dist(a,b) for a in labels for b in labels])>dist_threshold
                    # similarity_matrix = np.array([[hamming_distance(a, b) for b in labels] for a in labels])
                    # dists.append(similarity_matrix)
                    # if np.any(similarity_matrix < dist_threshold)and len(labels)>1:
                    #     self.add_annotation(tiernew, l[0], l[1], sep.join(l[2]))
                        # disagreement_count += 1
                        
                    if  np.count_nonzero(dist==0)<=(len(dist)//2) and len(labels)>1:
                        self.add_annotation(tiernew, l[0], l[1], sep.join(l[2]))
                        # disagreement_count += 1
                l = [begin, end, [value]]
            else:
                if end > l[1]:
                    l[1] = end
                l[2].append(value)
        return tiernew, disagreement_count
    
    def add_agreement_tier(self, tiers, tiernew, gapt=10, sep='/', safe=True):
        if tiernew is None:
            tiernew = u'{}_merged'.format('_'.join(tiers))
        self.add_tier(tiernew)
        aa = [(sys.maxsize, sys.maxsize, None)] + sorted((
            a for t in tiers for a in self.get_annotation_data_for_tier(t)),
            reverse=True)
        l = None
        agreement_count = 0
        dists = []
        while aa:
            begin, end, value = aa.pop()
            if l is None:
                l = [begin, end, [value]]
            elif begin - l[1] >= gapt:
                if not safe or l[1] > l[0]:
                    labels = [self.label_map.labelfromstr(l[2][i])[:6] for i in range(len(l[2]))]
                    dist = np.array([self.calc_dist(a,b) for a in labels for b in labels])>dist_threshold
                    # similarity_matrix = np.array([[hamming_distance(a, b) for b in labels] for a in labels])
                    # dists.append(similarity_matrix)
                    # if np.any(similarity_matrix < dist_threshold)and len(labels)>1:
                    #     self.add_annotation(tiernew, l[0], l[1], sep.join(l[2]))
                        # disagreement_count += 1
                        
                    if  np.count_nonzero(dist==0)>(len(dist)//2) and len(labels)>1:
                        self.add_annotation(tiernew, l[0], l[1], sep.join(l[2]))
                        agreement_count += 1
                l = [begin, end, [value]]
            else:
                if end > l[1]:
                    l[1] = end
                l[2].append(value)
        return tiernew, agreement_count
    
    def add_gold_tier(self):
        pass

    @staticmethod
    def calc_dist(x,y):
        return sum([np.abs(a-b) for a,b in zip(x,y)])/len(x)
    
    def add_sep28k(self, csv_files):
        if not self.media_file:
            raise ValueError('Media file not set please add elan files first')
        
        df_sep = pd.read_csv(csv_files[0])
        df_sep_episodes = pd.read_csv(csv_files[1], header=None)

        df_sep['annotation'] = df_sep.iloc[:, 5:].apply(lambda row: self.label_map.strfromsep28k(row.to_list()), axis=1)
        df_sep = df_sep[~df_sep['annotation'].isin(['', 'NoStutteredWords'])]

        df_sep_episodes[2] = df_sep_episodes[2].apply(lambda x: x.split('/')[-1].replace('.mp4', '.wav'))
        episode_map = {row[2]: row[3] for row in df_sep_episodes.itertuples()}
        df_sep['item'] = df_sep['EpId'].map(episode_map)
        df_sep = df_sep[df_sep['item']==self.media_file]

        # update the start and end times from segment to time in ms with fs=16000
        df_sep[['start_time','end_time']] = df_sep[['Start','Stop']].apply(lambda x: x*1000/16000).astype(int)

        df_sep = df_sep[['start_time', 'end_time', 'annotation']]
        df_sep['annotator_id'] = 'sep28k'
        df_sep['media_file'] = self.media_file
        # drop row if null value in row
        df_sep.dropna(inplace=True)
        tier_dict = {'TIER_ID': 'sep28k', 'LINGUISTIC_TYPE_REF': 'default-lt'}
        self.add_tier('sep28k', tier_dict=tier_dict)
        for row in df_sep.itertuples():
            self.add_annotation('sep28k', row.start_time, row.end_time, row.annotation)
       
        # _= self.merge_tiers(['sep28k'], 'sep28k_merged')
        # self.remove_tier('sep28k')

    def to_dataframe(self):
        df = pd.DataFrame()
        for tier in self.annotators:
            temp_df = pd.DataFrame()
            annotations = self.get_annotation_data_for_tier(tier)
            temp_df['anotator'] = [tier]*len(annotations)
            temp_df = pd.concat([temp_df, pd.DataFrame(annotations, columns=['start', 'end', 'label'])], axis=1)
            label = temp_df['label'].apply(lambda x: self.label_map.labelfromstr(x))
            for i in range(len(self.label_map.labels)):
                temp_df[self.label_map.labels[i]] = label.apply(lambda x: x[i])
            df = pd.concat([df, temp_df]).reset_index(drop=True)
        df['media_file'] = self.media_file
        return df
    
class ELANGroup(object):
    
    def __init__(self, elan_files, save_path=None, label_tier_name='stuttering moments', sep28k=None):

        self.media_file = None
        self.annotations = None
        self.gold_ann = None
        self.label_map = LabelMap()

        self.elan_files = elan_files
        self.elans = [self.read_elan(elan_file, label_tier_name=label_tier_name, label_map=self.label_map) for elan_file in elan_files]
        
        self.sep28k = self.add_sep28k(sep28k) if sep28k else None
        self.bau = None
        self.mas = None
        self.sad = None
        # self.compute_agreement()

        if save_path:
            self.build_combined_tree()
            self.write_to_file(save_path)

    def add_elan(self, elan_file):
        elan = self.read_elan(elan_file)
        self.elans.append(elan)
        return elan
    
    def add_sep28k(self, csv_files):

        if not self.media_file:
            raise ValueError('Media file not set please add elan files first')
        
        df_sep = pd.read_csv(csv_files[0])
        df_sep_episodes = pd.read_csv(csv_files[1], header=None)

        df_sep['annotation'] = df_sep.iloc[:, 5:].apply(lambda row: self.label_map.strfromsep28k(row.to_list()), axis=1)
        df_sep = df_sep[~df_sep['annotation'].isin(['', 'NoStutteredWords'])]

        df_sep_episodes[2] = df_sep_episodes[2].apply(lambda x: x.split('/')[-1].replace('.mp4', '.wav'))
        episode_map = {row[2]: row[3] for row in df_sep_episodes.itertuples()}
        df_sep['item'] = df_sep['EpId'].map(episode_map)
        df_sep = df_sep[df_sep['item']==self.media_file]

        # update the start and end times from segment to time in ms with fs=16000
        df_sep[['start_time','end_time']] = df_sep[['Start','Stop']].apply(lambda x: x*1000/16000).astype(int)

        df_sep = df_sep[['start_time', 'end_time', 'annotation']]
        df_sep['annotator_id'] = 'sep28k'
        df_sep['media_file'] = self.media_file
        # drop row if null value in row
        df_sep.dropna(inplace=True)

        return df_sep

    def read_elan(self, elan_file, label_tier_name="stuttering moments", label_map=None):
        elan = ELAN(elan_file, label_tier_name=label_tier_name, label_map=label_map)
        md_file = elan.media_file
        if self.media_file is None:
            self.media_file = md_file
        elif self.media_file != md_file:
            print(f'Media files do not match: {self.media_file} != {elan.media_file}')
            return None
        return elan
    
    def compute_agreement(self):
        # TODO: compute agreement between annotators and return the mas, bau and sad tiers
        self.mas = self.get_ann_from_json('datasets/fluencybank/mas_bb_ann.json')
        self.bau = self.get_ann_from_json('datasets/fluencybank/bau_bb_ann.json')
        self.sad = self.get_ann_from_json('datasets/fluencybank/sad_bb_ann.json')
        pass

    def build_combined_tree(self, mas=None, bau=None, sad=None):

        self.root = ET.Element('ANNOTATION_DOCUMENT')
        self.root.attrib['AUTHOR'] = 'FluencyBank'

        # get header from the first elan
        self.header = self.elans[0].header
        self.root.append(self.header)

        self.time_order = ET.SubElement(self.root, 'TIME_ORDER')

        # combine annotations into different tiers
        for elan in self.elans:
            
            for time_slot in elan.time_slots:
                ts = ET.SubElement(self.time_order, 'TIME_SLOT')
                old_id = time_slot.attrib.get('TIME_SLOT_ID')
                ts.attrib['TIME_SLOT_ID'] = f'{elan.annotator_id}_{old_id}'
                ts.attrib['TIME_VALUE'] = time_slot.attrib.get('TIME_VALUE')

            for tier in elan.tiers:
                if tier.attrib.get('TIER_ID').lower() == elan.label_tier_name:
                    new_tier = ET.SubElement(self.root, 'TIER')
                    new_tier.attrib['TIER_ID'] = f'sm_{elan.annotator_id}'
                    new_tier.attrib['LINGUISTIC_TYPE_REF'] = tier.attrib.get('LINGUISTIC_TYPE_REF')
                    for annotation in tier.iter('ANNOTATION'):                       
                        for alignable_annotation in annotation.iter('ALIGNABLE_ANNOTATION'):
                            ann = ET.SubElement(new_tier, 'ANNOTATION')
                            align_ann = ET.SubElement(ann, 'ALIGNABLE_ANNOTATION')
                            align_ann.attrib['ANNOTATION_ID'] = f'{elan.annotator_id}_{alignable_annotation.attrib.get("ANNOTATION_ID")}'
                            align_ann.attrib['TIME_SLOT_REF1'] = f'{elan.annotator_id}_{alignable_annotation.attrib.get("TIME_SLOT_REF1")}'
                            align_ann.attrib['TIME_SLOT_REF2'] = f'{elan.annotator_id}_{alignable_annotation.attrib.get("TIME_SLOT_REF2")}'
                            ann_value = ET.SubElement(align_ann, 'ANNOTATION_VALUE')
                            ann_value.text = alignable_annotation.find('ANNOTATION_VALUE').text

        # append mas, bau and sad tiers
        if self.mas is not None:
            self.append_ann_to_new_tier(self.mas, name='mas')
        if self.bau is not None:
            self.append_ann_to_new_tier(self.bau, name='bau')
        if self.sad is not None:
            self.append_ann_to_new_tier(self.sad, name='sad')
        
        # # append sep28k tiers
        if self.sep28k is not None:
            self.append_ann_to_new_tier(self.sep28k, name='sep28k')

        # append lingustic types and constraints
        for lingustic_type in self.elans[0].root.iter('LINGUISTIC_TYPE'):
            self.root.append(lingustic_type)
        for constraint in self.elans[0].root.iter('CONSTRAINT'):
            self.root.append(constraint)
        
    def extract_labels(self, save_path=None):
        anns = []
        total_failed = 0
        for elan in self.elans:
            ann, failed = elan.extract_labels()
            anns.append(ann)
            total_failed += len(failed)
        print(f'Total failed annotations: {total_failed}')
        self.annotations = pd.concat(anns)
        # temp_df['raw_annotations'] = temp_df[list(self.label_map.keys())+['start_time','end_time']].apply(lambda row: f'{{"bb":"({row["start_time"]},{row["end_time"]})", "ann":"{row[self.label_map.keys()].to_list()}"}}', axis=1)
        if save_path:
            self.annotations.to_csv(save_path, index=False)
        return self.annotations
    
    def append_ann_to_new_tier(self,ann, name='gold'):
        '''
        Append annotations to a new tier
        input:
            ann: pd.DataFrame containing annotations with columns start_time, end_time and annotation
            name: str
        '''
        new_tier = ET.SubElement(self.root, 'TIER')
        new_tier.attrib['TIER_ID'] = f'{name}'
        new_tier.attrib['LINGUISTIC_TYPE_REF'] = 'orthography'
        num_ann = len(ann)
        for i, (_,row) in enumerate(ann.iterrows()):

            try:
                start_time = str(row['start_time'])
                end_time = str(row['end_time'])
                annotation = row['annotation']
            except Exception as e:
                print(f'Error extracting annotation from row {i}: {e}')
                continue
            start_time_id = f'{name}_ts{i}'
            end_time_id = f'{name}_ts{i+num_ann+1}'
            

            # append time slot
            ts = ET.SubElement(self.time_order, 'TIME_SLOT')
            ts.attrib['TIME_SLOT_ID'] = start_time_id
            ts.attrib['TIME_VALUE'] = start_time
            ts = ET.SubElement(self.time_order, 'TIME_SLOT')
            ts.attrib['TIME_SLOT_ID'] = end_time_id
            ts.attrib['TIME_VALUE'] = end_time

            # append annotation
            ann = ET.SubElement(new_tier, 'ANNOTATION')
            align_ann = ET.SubElement(ann, 'ALIGNABLE_ANNOTATION')
            align_ann.attrib['ANNOTATION_ID'] = f'{name}_{i}'
            align_ann.attrib['TIME_SLOT_REF1'] = start_time_id
            align_ann.attrib['TIME_SLOT_REF2'] = end_time_id
            ann_value = ET.SubElement(align_ann, 'ANNOTATION_VALUE')
            ann_value.text = annotation
    
    def get_ann_from_json(self, json_file):
        try:
            ann = pd.read_json(json_file)
            ann = ann.T
            ann.sort_values(0, inplace=True)
            ann.rename(columns={0:'start_time', 1:'end_time'}, inplace=True)
            ann['annotation'] = ann.iloc[:, 2:].apply(lambda row: self.label_map.strfromlabel(row.to_list()), axis=1)
        except Exception as e:
            print(f'Error reading {json_file}: {e}')
            return None
        return ann
    
    def write_to_file(self, save_path='combined.eaf'):
        # write the root to .eaf file
        tree = ET.ElementTree(self.root)
        clean_element(self.root)
        xml_string = ET.tostring(self.root, encoding='utf-8', xml_declaration=True, method='xml')
        pretty_xml = xml.dom.minidom.parseString(xml_string).toprettyxml(indent="    ")

        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(pretty_xml)

class ELAN(object):

    def __init__(self, file_path, label_tier_name='stuttering moments', label_map=None):

        pattern = r'/(A\d+)_'
        self.annotator_id = re.search(pattern, file_path).group(1)
        self.file_path = file_path
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()
        self.media_file = self.get_media_file()
        self.label_tier_name = label_tier_name
        self.label_map = label_map if label_map else LabelMap()

        # xml components
        self.time_slots = self.root.findall('TIME_ORDER/TIME_SLOT')
        self.tiers = self.root.findall('TIER')
        self.header = self.root.find('HEADER')
        self.ann_tier = self.root.find(f'TIER[@TIER_ID="{self.label_tier_name}"]')
        self.failed = []

    def get_media_file(self):
        media_file = [media_file.attrib.get('MEDIA_URL') for media_file in self.root.iter('MEDIA_DESCRIPTOR') if media_file.attrib.get('MIME_TYPE') == 'audio/x-wav'][0]
        return os.path.basename(media_file)

    def extract_labels(self):

        labels = []
        annotation_ids = []
        start_times = []
        end_times = []

        for tier in self.root.iter('TIER'):
            if tier.attrib.get('TIER_ID').lower() == self.label_tier_name:
                for annotation in tier.iter('ANNOTATION'):
                    for alignable_annotation in annotation.iter('ALIGNABLE_ANNOTATION'):
                        label = self.label_map.labelfromstr(alignable_annotation.find('ANNOTATION_VALUE').text)
                        if label is None:
                            self.failed.append((self.annotator_id, self.media_file, alignable_annotation.find('ANNOTATION_VALUE').text))
                            print(f'Failed to extract label from {alignable_annotation.find("ANNOTATION_VALUE").text} by annotator {self.annotator_id} in file {self.media_file}')
                            continue
                        ann_id = alignable_annotation.attrib.get('ANNOTATION_ID')
                        s = alignable_annotation.attrib.get('TIME_SLOT_REF1')
                        start = self.root.find(f'TIME_ORDER/TIME_SLOT[@TIME_SLOT_ID="{s}"]').attrib.get('TIME_VALUE')
                        e = alignable_annotation.attrib.get('TIME_SLOT_REF2')
                        end = self.root.find(f'TIME_ORDER/TIME_SLOT[@TIME_SLOT_ID="{e}"]').attrib.get('TIME_VALUE')

                        labels.append(label)
                        annotation_ids.append(ann_id)
                        start_times.append(start)
                        end_times.append(end)

        ann_df = pd.DataFrame({'annotation_id': annotation_ids, 'start_time': start_times, 'end_time': end_times})
        ann_df[self.label_map.labels] = pd.DataFrame(labels)
        ann_df['annotator_id'] = self.annotator_id
        ann_df['media_file'] = self.media_file

        return  ann_df, self.failed

if __name__ == "__main__":

    # walk through the fluencybank directory and get all the elan files
    file_path = 'datasets/fluencybank/our_annotations/interview/'
    combined_save_path = 'outputs/fluencybank/combined_files/'
    csv_save_path = 'outputs/fluencybank/our_annotations/our_annotations.csv'

    elanfiles = {}
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.eaf'):
                if file not in elanfiles:
                    elanfiles[file] = [os.path.join(root, file)]
                else:
                    elanfiles[file].append(os.path.join(root, file))
    
    anns = pd.DataFrame()
    sep_28k_ann = pd.DataFrame()
    failed = {}
    for key, value in elanfiles.items():
        if len(value)==1:
            continue
        print(f'Processing {key}')
        elan_group = ELANGroup(value, 
                               save_path=combined_save_path+key, 
                               sep28k=('datasets/fluencybank/fluencybank_labels.csv', 'datasets/fluencybank/fluencybank_episodes.csv')
                               )
        ann = elan_group.extract_labels()
        
        # failed[key] = [(f.annotator_id, f.failed) for f in elan_group.elans]
        anns = pd.concat([anns, ann])
        if elan_group.sep28k is not None:
            sep_28k_ann = pd.concat([sep_28k_ann, elan_group.sep28k])

    groupped_ann = anns.groupby(['media_file', 'annotator_id']).size().reset_index(name='counts')
    groupped_sep28k_ann = sep_28k_ann.groupby(['media_file', 'annotator_id']).size().reset_index(name='counts')

    merged_df = pd.merge(groupped_ann, groupped_sep28k_ann, on=['media_file', 'annotator_id'], how='outer').fillna(0)

    fig, ax = plt.subplots()
    width=0.35
    ax.bar(merged_df['media_file'], merged_df['counts_x'], width, label='Our Annotations')
    ax.bar(merged_df['media_file'], merged_df['counts_y'], width, label='SEP-28k')
    ax.set_ylabel('Counts')
    ax.set_title('Annotation Counts')
    # rotate x labels
    plt.xticks(rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/fluencybank/interview.png')

    # anns.to_csv(csv_save_path, index=False)




