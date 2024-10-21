# instantiate the pipeline


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
from glob import glob
import os
import json

def merge_speaker_entries(entries, threshold):
    merged_entries = []
    prev_entry = entries[0]

    for current_entry in entries[1:]:
        # Check if the speakers are the same and the time gap is below the threshold
        if (prev_entry['speaker_id'] == current_entry['speaker_id'] and
            current_entry['start_time'] - prev_entry['end_time'] <= threshold):
            # Merge the current entry with the previous one by extending the duration
            prev_entry['end_time'] = current_entry['end_time']
            prev_entry['duration'] = prev_entry['end_time'] - prev_entry['start_time']
        else:
            # Append the previous entry to the result and move to the next
            merged_entries.append(prev_entry)
            prev_entry = current_entry

    # Don't forget to append the last entry
    merged_entries.append(prev_entry)
    return merged_entries


def main(data_path, output_path):
    wav_files = glob(data_path + '/*.wav')
    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1").to(torch.device("cuda"))

    dia_results = {}
    for wav_file in wav_files:
        dia_result = []
        with ProgressHook()as hook:
            diarization = pipeline(wav_file, 
                                hook=hook, 
                                num_speakers=2)
            for s,t,l in diarization.itertracks(yield_label=True):
                dia_result.append({
                    'start_time':s.start,
                    'duration':s.duration,
                    'end_time':s.start + s.duration,
                    'speaker_id':l    
                })
            merged_result = merge_speaker_entries(dia_result, 10)
            dia_results[os.path.basename(wav_file)] = merged_result
            visualize_speaker_intervals(merged_result, title=wav_file, save_path='datasets/fluencybank/diarization/imgs/' + os.path.basename(wav_file).replace('.wav', '.png'))    

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dia_results, f, indent=4)

def visualize_speaker_intervals(entries, title='Speaker Intervals', save_path=None):
    # Define colors for speakers
    speaker_colors = {}
    y = {}
    colors = plt.get_cmap('tab10')  # Colormap with 10 different colors
    current_color = 0
    cols = [1,3,5]
    fig, ax = plt.subplots(figsize=(20, 4))
    for entry in entries:
        speaker_id = entry['speaker_id']
        
        # Assign a color to each speaker
        if speaker_id not in speaker_colors:
            speaker_colors[speaker_id] = colors(current_color)
            current_color += 1
            y[speaker_id] = cols.pop(0)
        # Create a rectangle for the speaking interval
        start_time = entry['start_time']
        duration = entry['duration']
        
        ax.broken_barh([(start_time, duration)], (y[speaker_id], 1), facecolors=speaker_colors[speaker_id])
        
        # Label each bar with the speaker ID
        # ax.text(start_time + duration / 2, 14, speaker_id, ha='center', va='center', fontsize=10)

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speaker Intervals')
    ax.set_title(title)

    # Create a legend to identify speakers by color
    legend_handles = [mpatches.Patch(color=color, label=speaker) for speaker, color in speaker_colors.items()]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display grid and adjust layout
    ax.grid(False)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':

    data_path = 'datasets/fluencybank/wavs/interview'
    output_path = 'datasets/fluencybank/diarization/interview.json'
    # main(data_path, output_path)
    with open(output_path, 'r') as f:
        dia_results = json.load(f)
    
    for k,v in dia_results.items():
        visualize_speaker_intervals(v, title=k, save_path='datasets/fluencybank/diarization/imgs/' + k.replace('.wav', '.png'))
