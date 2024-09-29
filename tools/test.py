import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

# Your JSON data
with open('datasets/fluencybank/our_annotations/interview/csv/iaa_temp_2.json') as f:
    data = json.load(f)

# Convert the JSON data to a DataFrame
df = pd.DataFrame(data).T.reset_index()
df = pd.melt(df, id_vars='index', value_vars=['alpha', 'ks', 'sigma'], 
             var_name='Metric', value_name='Value')

# Plotting
plt.figure(figsize=(15, 8))
sns.barplot(data=df, x='index', y='Value', hue='Metric', palette='Set2')
plt.xticks(rotation=90)
plt.title('Comparison of Alpha, KS, and Sigma across Different media files')
plt.xlabel('Category')
plt.ylabel('Value')
plt.tight_layout()

plt.savefig('comparison.png')

df = pd.read_csv('datasets/fluencybank/our_annotations/interview/csv/labels_2.csv')
df.sort_values('media_file', inplace=True)
fig, ax = plt.subplots(figsize=(15, 8))
for i, (media_file, group) in enumerate(df.groupby('annotator')):
    sns.histplot(group['media_file'], ax=ax, label=media_file)
plt.legend(title='Annotator')
plt.xticks(rotation=90)
plt.title('Distribution of counts of annotations')
plt.savefig('distribution.png')

# solution to leetcode proble 123
