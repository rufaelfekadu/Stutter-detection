import re
import numpy as np
import pandas as pd
import sys

class LabelMap(object):
    def __init__(self):
        self.labels = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM', 'ME', 'T', 'NS']
        # self.labels = ['SR', 'ISR', 'MUR', 'P', 'B', 'NV', 'V', 'FG', 'HM', 'ME', 'T']
        self.labeltoidx = dict(zip(self.labels, range(len(self.labels))))
        self.idxtolabel = dict(zip(range(len(self.labels)), self.labels))
        self.description = {
            'SR': 'SoundRepetition',
            'ISR': 'IncompeleteSyllableRepetition',
            'MUR': 'MultisyllabicUnitRepetition',
            'P': 'Prolongation',
            'B': 'Block',
            'NV': 'NonVerbal',
            'V': 'Verbal',
            'FG': 'FacialGrimaces',
            'HM': 'HeadMovement',
            'ME': 'MovementOfExtremities',
            'T': 'Tension',
            'NS': 'NoStutteredWords'}
        self.core = ['SR', 'ISR', 'MUR', 'P', 'B']
        self.secondary = ['V', 'FG', 'HM', 'ME']
        self.tension = ['T']
        self.fluent = ['NS']
        self.sep28k_labels = ['Unsure', 'PoorAudioQuality', 'Prolongation', 'Block', 'SoundRep', 'WordRep', 'DifficultToUnderstand',
                              'Interjection', 'NoStutteredWords', 'NaturalPause', 'Music', 'NoSpeech']

    def get_label_value(self, label):
        return self.labels.get(label, None)

    def __dict__(self):
        return self.labeltoidx

    def strfromsep28k(self, sep28k_label):
        sep28k_label = (np.array(sep28k_label)>=2).astype(int)
        core = '+'.join(set([self.sep28k_labels[i] for i in range(len(sep28k_label)) if sep28k_label[i] == 1])).replace('NoStutteredWords+', '').replace('+NoStutteredWords', '')
        return core
     
    def strfromlabel(self, label_array):
        core = '+'.join([self.labels[i] for i in range(len(label_array)) if label_array[i] == 1 and self.labels[i] in self.core])
        secondary = '+'.join([self.labels[i] for i in range(len(label_array)) if label_array[i] == 1 and self.labels[i] in self.secondary])
        tension = label_array[self.labeltoidx['T']]  
        return f'{core};{secondary};{tension}'
    
    def labelfromstr(self, label_str, verbose=False):
        # strip spaces tabs and newlines from the label string
        label_original= label_str
        replacements = {
            '!':'1',
            ':': ';',
            ',': ';',
            '.': ';',
            'O': '0',
            'RS': 'SR',
            'IISR': 'ISR',
            'EM': 'ME',
            'GH': 'FG',
            'FR' : 'FG',
            'SE' : 'SR',
            'BISR': 'B+ISR',
            'FGV': 'FG+V',
            'VFG': 'V+FG',
            'PSR': 'P+SR',
            'PFG': 'P+FG',
            'FFG': 'FG',
            'SRP': 'SR+P',
            'ISRP': 'ISR+P',
            'FV+G': 'FG+V',
            'V_0': 'V;0',
            '_': '',
            ' ': '',
            '\t': '',
            '\n': '',
        }

        for key, value in replacements.items():
            label_str = label_str.replace(key, value)

        # label_str = label_str.upper().replace(' ', '').replace('_', '0').replace(':', ';').replace(',', ';').replace('++', '+').replace('SE', 'SR').replace('\t', '').replace('\n', '').replace('.',';').replace('O','0').replace('RS','SR').replace('IISR','ISR').replace('EM','ME').replace('GH','FG')
        # pattern = r'((?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;+\d]+)(?:[+;](?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;+\d]+))*)(?:;(\d))$'
        # pattern = r'((?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;+\d]+)(?:[+;](?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;+\d]+))*)(?:;[^;\d]*)*;(\d)$'
        # pattern = r'((?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;\d]+)(?:[+;]SR|ISR|MUR|P|B|V|FG|HM|ME|[^;]+)*)(?:;(\d))?$'        # pattern = r'((?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;+\d]+)(?:;[^;\d]*)*);(\d)$'
        # if label_str starts with ; append 0
        label_str = label_str.strip().upper()
        if label_str.endswith(';'): label_str = label_str + '0'
        pattern = r"^(?:;)*(.*?)(?:;(\d))?$"
        matches = re.findall(pattern, label_str)
        label_array = [0] * len(self.labels)
        if not matches:
            if verbose:
                print(f'No matches found for {label_str}', file=open('label_errors.log', 'a'))
            return label_array
        for match in matches:
            behavior, tension = match
            behavior = re.split(r'[+;]', behavior)
            behavior = [b for b in behavior if b and not b.isdigit()]
            for b in behavior:
                # b = b.lower()
                try:
                    b = b.strip(' ')
                    label_array[self.labeltoidx[b]] = 1
                except KeyError:
                    if verbose:
                        print(f'Unknown abbreviation found: {b} from {label_original}', file=open('label_errors.log', 'a'))
            if tension:
                try:
                    tension = int(tension)
                    label_array[self.labeltoidx['T']] = tension
                except KeyError:
                    if verbose:
                        print(f'Unknown tension value found: {tension} from {label_original}', file=open('label_errors.log', 'a'))
            if verbose:
                print(f'label: {label_original} Behavior: {behavior} tension: {tension} label: {label_array}', file=open('labels_found.log', 'a'))
        return label_array

    def check_core(self, label_array):
        if isinstance(label_array, str):
            label_array = self.labelfromstr(label_array)
        return any([label_array[self.labeltoidx[core]] for core in self.core])

    def check_secondary(self, label_array):
        if isinstance(label_array, str):
            label_array = self.labelfromstr(label_array)
        return any([label_array[self.labeltoidx[secondary]] for secondary in self.secondary])
    
    def check_tension(self, label_array):
        if isinstance(label_array, str):
            label_array = self.labelfromstr(label_array)
        return label_array[self.labeltoidx['T']]
    
    def get_core(self, label_array):
        if isinstance(label_array, str):
            label_array = self.labelfromstr(label_array)
        return [self.idxtolabel[i] for i in range(len(label_array)) if label_array[i] == 1 and self.idxtolabel[i] in self.core], [label_array[self.labeltoidx[core]] for core in self.core]
    
    def get_secondary(self, label_array):
        if isinstance(label_array, str):
            label_array = self.labelfromstr(label_array)
        return [self.idxtolabel[i] for i in range(len(label_array)) if label_array[i] == 1 and self.idxtolabel[i] in self.secondary], [label_array[self.labeltoidx[secondary]] for secondary in self.secondary]
    
    def get_tension(self, label_array):
        if isinstance(label_array, str):
            label_array = self.labelfromstr(label_array)
        return label_array[self.labeltoidx['T']]
    
    def get_all(self, label_array):
        # get everything except tension
        if isinstance(label_array, str):
            label_array = self.labelfromstr(label_array)
     
        return [self.labels[i] for i in range(len(label_array[:-2])) if label_array[i] == 1]

def clean_element(element):
    # Replace None text with an empty string
    if element.text is None:
        element.text = ''
    # Replace None attributes with an empty string
    for key, value in element.attrib.items():
        if value is None:
            element.attrib[key] = ''
    # Recursively clean child elements
    for child in element:
        clean_element(child) 
# class LabelMap(object):
#     def __init__(self):
#         self.FP = 1
#         self.SR = 2
#         self.ISR = 3
#         self.MUR = 4
#         self.PWS = 5
#         self.P = 6
#         self.B = 7
#         self.V = 8
#         self.FG = 9
#         self.HM = 10
#         self.ME = 11
#         self.T_0 = 12
#         self.T_1 = 13
#         self.T_2 = 14
#         self.T_3 = 15

#         self.core = [ self.FP, self.SR, self.ISR, self.MUR, self.P, self.B, self.nan]
#         self.secondary = [self.V, self.FG, self.HM, self.ME, self.nan]
#         self.tension = [self.T_0, self.T_1, self.T_2, self.T_3]

#     def __dict__(self):
#         return {'fp': self.FP, 'sr': self.SR, 'isr': self.ISR, 'mur': self.MUR, 'pws': self.PWS, 'p': self.P, 'b': self.B, 
#             'v': self.V, 'fg': self.FG, 'hm': self.HM, 'me': self.ME,
#             't_0': self.T_0, 't_1': self.T_1, 't_2': self.T_2, 't_3': self.T_3,
#             '0': self.nan
#         }

def strfromlabel(label_array):
    label_map = LabelMap().__dict__()
    core = '+'.join([key for key, value in label_map.items() if label_array[value] == 1 and value in label_map.core])
    secondary = '+'.join([key for key, value in label_map.items() if label_array[value] == 1 and value in label_map.secondary])
    tension = ''.join([key.split('_')[1] for key, value in label_map.items() if label_array[value] == 1 and value in label_map.tension])
    return f'{core};{secondary};{tension}'



def labelfromstr(label_str):
    label_map = LabelMap().__dict__()
    # strip _ from the label string
    original_label_str = label_str
    label_str = label_str.lower()
    label_str = label_str.replace('_','0').replace(':',';').replace(',',';').replace('++','+').replace(' ','').replace('se','sr')

    label_array = [0] * len(label_map)
    label = label_str.split(';')
    tention = label[-1]
    for l in label[:-1]:
        for x in l.split('+'):
            try:
                label_array[label_map[x]] = 1
            except KeyError as e:
                print(f'Error extracting label from {original_label_str}: {e}')
                return None
            
    try:
        # split digit and non digit characters from the tension label
        pattern = r'\d+|\D+'
        matches = re.findall(pattern, tention)
        digit = [match for match in matches if match.isdigit()][0]
        non_digit = [match for match in matches if not match.isdigit()][0]
    
        label_array[label_map[f't_{digit}']] = 1
        label_array[label_map[non_digit]] = 1
    except KeyError as e:
        print(f'Error extracting label from {original_label_str}: {e}')
        return None
    # try:
    #     core, secondary, tension = label_str.split(';')
    # except Exception as e:
    #     print(f'Error splitting label {original_label_str}: {e}')
    #     label = label_str.split(';')
    #     tension = label[-1]
    #     for l in label[:-1]:
    #         if l in label_map:
    #             label_array[label_map[l]] = 1
    #         else:
    #             print(f'Error extracting label from {original_label_str}')
    #             return None
    # try:
    #     # get core brhaviour
    #     if core:
    #         for c in core.split('+'):
    #             label_array[label_map[c]] = 1
        
    #     # get secondary behaviour
    #     if secondary:
    #         for s in secondary.split('+'):
    #             label_array[label_map[s]] = 1
        
    #     # get tension
    #     if tension:
    #         label_array[label_map[f't_{tension}']] = 1
    # except Exception as e:
    #     print(f'Error extracting label from {original_label_str}: {e}')
    #     return None

    return label_array



def labelfromstr_re(label_str):
    label_str = label_str.replace(' ', '').replace('_', '0').replace(':', ';').replace(',', ';').replace('++', '+').replace('SE', 'SR')
    pattern = r'((?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;+\d]+)(?:[+;](?:SR|ISR|MUR|P|B|V|FG|HM|ME|[^;+\d]+))*)(?:;(\d))?'
    matches = re.findall(pattern, label_str)
    label_map = LabelMap().__dict__()
    label_array = [0] * len(label_map)
    
    for match in matches:
        behavior, tension = match
        behavior = re.split(r'[+;]', behavior)
        for b in behavior:
            b = b.lower()
            try:
                label_array[label_map[b]] = 1
            except KeyError:
                print(f'Unknown abbreviation found: {b}, skipping...')
        if tension:
            try:
                label_array[label_map[f't_{tension}']] = 1
            except KeyError:
                print(f'Unknown tension value found: {tension}, skipping...')

    return label_array

# def labelfromstr_re(label_str):
#     label_str = label_str.replace(' ','').replace('_','0').replace(':',';').replace(',',';').replace('++','+').replace('SE','SR')
#     pattern = r'((?:SR|ISR|MUR|P|B|V|FG|HM|ME)(?:[+;](?:SR|ISR|MUR|P|B|V|FG|HM|ME))*)(?:;(\d))?'
#     # pattern = r'(SR|ISR|MUR|P|B|V|FG|HM|ME|0)(?:[+;](SR|ISR|MUR|P|B|V|FG|HM|ME|0))*;(\d)'
#     matches = re.findall(pattern, label_str)
#     label_map = LabelMap().__dict__()
#     label_array = [0] * len(label_map)
#     for match in matches:
#         behavior, tension = match
#         behavior = re.split(r'[+;]', behavior)
#         for b in behavior:
#             b = b.lower()
#             try:
#                 label_array[label_map[b]] = 1
#             except KeyError as e:
#                 print(f'Error extracting label from {label_str}: {e}')
#                 return None
#         try:
#             label_array[label_map[f't_{tension}']] = 1
#         except KeyError as e:
#             print(f'Error extracting label from {label_str}: {e}')
#             return None
#     print(f' {label_str}: {matches}')
#     return label_array