import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
from stutter.utils.annotation import LabelMap
import seaborn as sns

sns.set(style="whitegrid")

def plot_spider(*values, categories, legend = ['train']):
    N = len(categories)

    # Create angles for radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Circular completion for plotting

    # Initialize radar chart
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Define color palette for multiple values (using seaborn colors)
    colors = sns.color_palette("husl", len(values))

    for i, val in enumerate(values):
        val += val[:1]  # Make the data circular
        ax.fill(angles, val, color=colors[i], alpha=0.25)
        ax.plot(angles, val, color=colors[i], linewidth=1, label=f'{legend[i]}')

    # Add labels
    plt.xticks(angles[:-1], categories, color='black', size=10)

    # Set fewer y-ticks
    max_ytick = ax.get_yticks()[-1]
    yticks = [max_ytick * i / 4 for i in range(5)]  # Adjust the number of y-ticks as needed
    ytick_labels = [str(round(tick,2)) for tick in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)  # Reduce font size for y-ticks


    # Set title and adjust layout
    # plt.title("Spider Plot for Different Classes", size=15, color='black', pad=20)

    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the plot
    plt.tight_layout()
    return fig
    
def main():
    label_map = LabelMap()
    df = pd.read_csv('datasets/fluencybank/our_annotations/reading/csv/total_label.csv')
    df_train = df[df['split'] == 'train']
    df_test = df[df['annotator'] == 'Gold']
    # categories = [label_map.description[i] for i in label_map.labels]
    # categories = ['P', 'B', 'ISR', 'SR', 'FG', 'HM', 'V']
    
    # train_count = df_train[categories].sum().to_list()
    # test_count = df_test[categories].sum().to_list()

    categories = ['SR', 'ISR', 'MUR', 'B', 'P', 'V', 'FG', 'HM']

    
    A1 = [0.45, 0.59, 0.34, 0.47, 0.29, 0.43, 0.62, 0.40]
    A2 = [0.65, 0.31, 0.66, 0.54, 0.60, 0.49, 0.62, 0.45]
    A3= [0.37, 0.50, 0.20, 0.46, 0.46, 0.31, 0.32, 0.53]
    SAD = [0.40, 0.37, 0.37, 0.47, 0.34, 0.32, 0.31, 0.13]
    BAU= [0.43, 0.51, 0.43, 0.44, 0.33, 0.29, 0.16, 0.00]
    MAS = [0.41, 0.51, 0.29, 0.48, 0.32, 0.30, 0.16, 0.00]
    


    # A1 = df_train[df_train['annotator']=='A1'][categories].sum().to_list()
    # A2 = df_train[df_train['annotator']=='A2'][categories].sum().to_list()
    # A3 = df_train[df_train['annotator']=='A3'][categories].sum().to_list()

    to_plot = [A1, A2, A3, BAU, MAS, SAD]
    legend = ['A1', 'A2', 'A3', 'BAU', 'MAS', 'SAD']

    fig = plot_spider(*to_plot, categories=categories, legend=legend)
    # remove background
    # fig.patch.set_visible(False)
    # plt.axis('off')
    plt.savefig('outputs/spider_plot.png')


if __name__ == '__main__':
    main()