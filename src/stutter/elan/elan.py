
import xml.etree.ElementTree as ET
import os
import pandas as pd

class LabelMap(object):
    def __init__(self):
        self.RM = 0
        self.FP = 1
        self.SR = 2
        self.ISR = 3
        self.MUR = 4
        self.P = 5
        self.B = 6
        self.V = 7
        self.NV = 8
        self.T_0 = 9
        self.T_1 = 10
        self.T_2 = 11
        self.T_3 = 12

        self.core = [self.RM, self.FP, self.SR, self.ISR, self.MUR, self.P, self.B]
        self.secondary = [self.V, self.NV]
        self.tension = [self.T_0, self.T_1, self.T_2, self.T_3]

    def __dict__(self):
        return {'RM': self.RM, 'FP': self.FP, 'SR': self.SR, 'ISR': self.ISR, 'MUR': self.MUR, 'P': self.P, 'B': self.B, 'V': self.V, 'NV': self.NV, 'T_0': self.T_0, 'T_1': self.T_1, 'T_2': self.T_2, 'T_3': self.T_3}

def strfromlabel(label_array):
    label_map = LabelMap().__dict__()
    core = '+'.join([key for key, value in label_map.items() if label_array[value] == 1 and value in label_map.core])
    secondary = ''.join([key for key, value in label_map.items() if label_array[value] == 1 and value in label_map.secondary])
    tension = ''.join([key.split('_')[1] for key, value in label_map.items() if label_array[value] == 1 and value in label_map.tension])
    return f'{core};{secondary};{tension}'

def labelfromstr(label_str):
    label_map = LabelMap().__dict__()
    core, secondary, tension = label_str.split(';')
    label_array = [0] * len(label_map)
    for c in core.split('+'):
        label_array[label_map[c]] = 1
    label_array[label_map[secondary]] = 1
    label_array[label_map[f'T_{tension}']] = 1
    return label_array

class ELANGroup(object):
    
    def __init__(self, elan_files, save_path=None, label_tier_name='stuttering moments'):
        self.label_map = LabelMap().__dict__()
        self.media_file = None
        self.annotations = None
        self.gold_ann = None

        self.elan_files = elan_files
        self.elans = [ELAN(elan_file, label_tier_name=label_tier_name) for elan_file in elan_files]

        self.compute_agreement()
        self.combine_elans()

        if save_path:
            self.write_to_file(save_path)
    
    def check_media(self, elan):
        md_file = elan.media_file
        if self.media_file is None:
            self.media_file = md_file
        elif self.media_file != md_file:
            print(f'Media files do not match: {self.media_file} != {elan.media_file}')
            return False
        return True
    
    def compute_agreement(self):
        # TODO: compute agreement between annotators and return the mas, bau and sad tiers
        self.mas = self.get_ann_from_json('datasets/fluencybank/mas_bb_ann.json')
        self.bau = self.get_ann_from_json('datasets/fluencybank/bau_bb_ann.json')
        self.sad = self.get_ann_from_json('datasets/fluencybank/sad_bb_ann.json')
        pass

    def combine_elans(self, mas=None, bau=None, sad=None):

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

        # append mas, bau and gold tiers
        if self.mas is not None:
            self.append_ann_to_new_tier(self.mas, name='mas')
        if self.bau is not None:
            self.append_ann_to_new_tier(self.bau, name='bau')
        if self.sad is not None:
            self.append_ann_to_new_tier(self.sad, name='sad')

        # append lingustic types and constraints
        for lingustic_type in self.elans[0].root.iter('LINGUISTIC_TYPE'):
            self.root.append(lingustic_type)
        for constraint in self.elans[0].root.iter('CONSTRAINT'):
            self.root.append(constraint)
        
    def extract_labels(self):
        ann_dfs = [elan.extract_labels() for elan in self.elans]
        temp_df = pd.concat(ann_dfs)
        # temp_df['raw_annotations'] = temp_df[list(self.label_map.keys())+['start_time','end_time']].apply(lambda row: f'{{"bb":"({row["start_time"]},{row["end_time"]})", "ann":"{row[self.label_map.keys()].to_list()}"}}', axis=1)
        temp_df.to_csv('datasets/fluencybank/labels_temp.csv', index=False)
        self.annotations = temp_df
        return temp_df
    
    def append_ann_to_new_tier(self,ann, name='gold'):

        new_tier = ET.SubElement(self.root, 'TIER')
        new_tier.attrib['TIER_ID'] = f'{name}'
        new_tier.attrib['LINGUISTIC_TYPE_REF'] = 'orthography'
        num_ann = len(ann)
        for i, (_,row) in enumerate(ann.iterrows()):

            try:
                start_time = str(row[0])
                end_time = str(row[1])
                annotation = strfromlabel(row.values[2:])
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
    
    @staticmethod
    def get_ann_from_json(json_file):
        try:
            ann = pd.read_json(json_file)
            ann = ann.T
            ann.sort_values(0, inplace=True)
        except Exception as e:
            print(f'Error reading {json_file}: {e}')
            return None
        return ann
    
    def write_to_file(self, save_path='combined.eaf'):
        # write the root to .eaf file
        tree = ET.ElementTree(self.root)
        tree.write(save_path, encoding='utf-8', xml_declaration=True, method='xml')

class ELAN(object):

    def __init__(self, file_path, label_tier_name='stuttering moments'):

        self.annotator_id = os.path.basename(file_path).split('.')[0].split('_')[1]
        self.file_path = file_path
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()
        self.media_file = self.get_media_file()
        self.label_tier_name = label_tier_name
        self.label_map = LabelMap().__dict__()

        # xml components
        self.time_slots = self.root.findall('TIME_ORDER/TIME_SLOT')
        self.tiers = self.root.findall('TIER')
        self.header = self.root.find('HEADER')
        self.ann_tier = self.root.find(f'TIER[@TIER_ID="{self.label_tier_name}"]')

    def get_media_file(self):
        media_file = [media_file.attrib.get('MEDIA_URL') for media_file in self.root.iter('MEDIA_DESCRIPTOR') if media_file.attrib.get('MIME_TYPE') == 'audio/x-wav'][0]
        return os.path.basename(media_file)
    
    # TODO: Update this function
    def get_label_array(self, label):
        label_map = self.label_map
        label_array = [0] * len(label_map)
        if not label:
            return label_array
        try:
            core, secondary, tension = label.split(';')
        except Exception as e:
            print(f'Error splitting label {label}')
            core, secondary, tension = None, None, None
        # get core
        try:
            if '+' in core:
                cores = core.split('+')
                for c in cores:
                    label_array[label_map[c]] = 1
            label_array[label_map[core]] = 1
        except KeyError as e:
            print(label,e)
        # get secondary
        try:
            if not secondary in ['0','_']:
                label_array[label_map[secondary]] = 1
        except KeyError as e:
            print(label,e)
        # get tension
        try:
            label_array[label_map[f'T_{tension}']] = 1
        except KeyError as e:
            print(label,e)

        return label_array

    def extract_labels(self):

        labels = []
        annotation_ids = []
        start_times = []
        end_times = []

        for tier in self.root.iter('TIER'):
            if tier.attrib.get('TIER_ID').lower() == self.label_tier_name:
                for annotation in tier.iter('ANNOTATION'):
                    for alignable_annotation in annotation.iter('ALIGNABLE_ANNOTATION'):
                        lables = self.get_label_array(alignable_annotation.find('ANNOTATION_VALUE').text)

                        ann_id = alignable_annotation.attrib.get('ANNOTATION_ID')
                        s = alignable_annotation.attrib.get('TIME_SLOT_REF1')
                        start = self.root.find(f'TIME_ORDER/TIME_SLOT[@TIME_SLOT_ID="{s}"]').attrib.get('TIME_VALUE')
                        e = alignable_annotation.attrib.get('TIME_SLOT_REF2')
                        end = self.root.find(f'TIME_ORDER/TIME_SLOT[@TIME_SLOT_ID="{e}"]').attrib.get('TIME_VALUE')

                        labels.append(lables)
                        annotation_ids.append(ann_id)
                        start_times.append(start)
                        end_times.append(end)

        ann_df = pd.DataFrame({'annotation_id': annotation_ids, 'start_time': start_times, 'end_time': end_times})
        for key, value in zip(self.label_map.keys(), zip(*labels)):
            ann_df[key] = value
        ann_df['annotator_id'] = self.annotator_id
        ann_df['media_file'] = self.media_file

        return  ann_df

if __name__ == "__main__":

    elanfiles = [i for i in os.listdir("datasets/fluencybank") if i.endswith('.eaf')]
    elan_group = ELANGroup([f'datasets/fluencybank/{i}' for i in elanfiles])
    elan_group.write_to_file()
    # elan_group.extract_labels().to_csv('datasets/fluencybank/labels_temp.csv', index=False)


