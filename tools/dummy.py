import pandas as pd

to_save = ['media_file', 'annotator', 'start', 'end', 'label']
reading = pd.read_csv('datasets/stutter-bank/reading/csv/total_dataset.csv')[to_save]
reading['media_file'] = reading['media_file'].apply(lambda x: f'reading_{x}')
interview = pd.read_csv('datasets/stutter-bank/interview/csv/total_dataset.csv')
interview['media_file'] = interview['media_file'].apply(lambda x: f'interview_{x}')
total = pd.concat([reading, interview])
total.to_csv('datasets/stutter-bank/total_dataset.csv', index=False)