import pandas as pd
import glob
import os
import torch
from stutter.utils.annotation import LabelMap
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

def write_rttm(df, output_path):
    for group, group_df in df.groupby('media_file'):
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{group}.rttm'), 'w') as f:
            for _, row in df.iterrows():
                rttm_line = f"SPEAKER {group} 1 {row['start']:.3f} {row['duration']:.3f} <NA> <NA> {row['label']} <NA> <NA>\n"
                f.write(rttm_line)

def write_uem(df, output_path):
    start = 0
    end = df['end'].max()
    
    for group, group_df in df.groupby('media_file'):
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{group}.uem'), 'w') as f:
            f.write(f"{group} NA {start:.3f} {end:.3f}\n")

def prep_df(df_path):
    label_map = LabelMap()
    df = pd.read_csv(df_path)
    df.dropna(subset=['label'], inplace=True)
    df = df[df['annotator'] == args.annotator]

    df['start'] = df['start']/1000
    df['end'] = df['end']/1000
    df['duration'] = df['end'] - df['start']

    new_rows = []
    for _, row in df.iterrows():
        labels = label_map.get_all(row['label'])

        for label in labels:
            new_row = row.copy()
            new_row['label'] = label_map.description[label]
            new_rows.append(new_row)
    df = pd.DataFrame(new_rows)
    df.reset_index(drop=True, inplace=True)
    return df


def main(args):
    # read the audio file 
    df = prep_df(args.label_path)
    write_rttm(df, os.path.join(args.output_file, args.annotator, 'rttms'))
    write_uem(df, os.path.join(args.output_file, args.annotator, 'uems'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare RTTM files")

    parser.add_argument("--label_path", type=str, help="Path to the label file")
    parser.add_argument("--output_file", type=str, help="Path to the output directory")
    parser.add_argument("--annotator", type=str, help="Annotator name")
    args = parser.parse_args()
    main(args)
