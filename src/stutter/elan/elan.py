
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import pandas as pd
import re
import json
from stutter.utils.annotation import LabelMap, clean_element

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

        df_sep_episodes[2] = df_sep_episodes[2].apply(lambda x: x.split('/')[-1].replace('.mp4', '.wav'))
        episode_map = {row[2]: row[3] for row in df_sep_episodes.itertuples()}
        df_sep['item'] = df_sep['EpId'].map(episode_map)
        df_sep = df_sep[df_sep['item']==self.media_file]

        # update the start and end times from segment to time in ms with fs=16000
        df_sep[['start_time','end_time']] = df_sep[['Start','Stop']].apply(lambda x: x*1000/16000).astype(int)

        df_sep = df_sep[['start_time', 'end_time', 'annotation']]
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
    file_path = 'datasets/fluencybank/our_annotations/'
    combined_save_path = 'outputs/fluencybank/combined_files/'
    csv_save_path = 'outputs/fluencybank/our_annotations.csv'

    elanfiles = {}
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.eaf'):
                if file not in elanfiles:
                    elanfiles[file] = [os.path.join(root, file)]
                else:
                    elanfiles[file].append(os.path.join(root, file))
    
    anns = pd.DataFrame()
    failed = {}
    for key, value in elanfiles.items():
        if len(value)==1:
            continue
        print(f'Processing {key}')
        elan_group = ELANGroup(value, 
                            #    save_path=combined_save_path+key, 
                            #    sep28k=('datasets/fluencybank/fluencybank_labels.csv', 'datasets/fluencybank/fluencybank_episodes.csv')
                               )
        ann = elan_group.extract_labels()
        # failed[key] = [(f.annotator_id, f.failed) for f in elan_group.elans]
        anns = pd.concat([anns, ann])
    # print the number of annotated events per item per annotator
    print(anns.groupby(['media_file', 'annotator_id']).size())
    # with open('outputs/fluencybank/failed_annotations.json', 'w') as f:
    #     json.dump(failed, f)

    # anns.to_csv(csv_save_path, index=False)




