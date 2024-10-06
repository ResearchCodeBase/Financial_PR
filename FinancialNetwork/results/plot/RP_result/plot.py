# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Times New Roman'
font = FontProperties(size=16)

# Data for Asia
countries_asia = ['Japan', 'China']
FCNN_asia = [0.6454, 0.36678]
KNN_asia = [0.6583, 0.46466]
LR_asia = [0.6349, 0.43048]
RF_asia = [0.6170, 0.75494]
ours_asia = [0.7511, 0.8441]


def plot_and_save_asia():
    bar_width = 0.1  # Make bars narrower
    spacing = 0.12  # Adjust spacing for better visual separation
    x = np.arange(len(countries_asia)) + 1  # Shift positions to the right

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    colors = ['#2E7BA6', '#C88FBD', '#2ca02c', '#6A5ACD', '#C99436']
    labels = ['FCNN', 'KNN', 'LR', 'RF', 'Ours']
    data = [FCNN_asia, KNN_asia, LR_asia, RF_asia, ours_asia]

    for i, (label, color) in enumerate(zip(labels, colors)):
        if color == '#C99436':
            ax.bar(x - i * (bar_width / 2 + spacing / 2) - 2.75 * bar_width, data[i], width=bar_width, color=color,
                   label=label)
        else:
            ax.bar(x - i * (bar_width / 2 + spacing / 2) - 2.75 * bar_width, data[i], width=bar_width, color=color,
                   alpha=0.5, label=label)

    ax.set_title('Asia', fontsize=26, fontweight='bold')
    ax.set_ylabel('MCC', fontsize=20)
    #ax.set_xticks(x + 2 * (bar_width + spacing) / 2)
    #ax.set_xticklabels(countries_asia, fontsize=17)
    ax.set_xticks([0.5,1.5])
    ax.set_xticklabels(['Japan', 'China'],fontsize=22)
    ax.set_xlim(-0.2,2.2)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5,color='gray',alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [4, 0, 1, 2, 3]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 11}, loc='upper right')

    ax.tick_params(axis='both', which='both', direction='in', labelsize=17)

    plt.tight_layout()
    plt.savefig(r'C:\Users\DELL-USER02\Desktop\写论文\论文中的图\柱状图\柱状图_Asia.png', format='png')
    plt.show()

plot_and_save_asia()




# Data for North America
countries_na = ['US', 'Canada']
FCNN_na = [0.7799, 0.7204]
KNN_na = [0.7774, 0.7455]
LR_na = [0.4786, 0.5822]
RF_na = [0.6368, 0.6031]
ours_na = [0.8864, 0.8702]

def plot_and_save_na():
    bar_width = 0.1  # Make bars narrower
    spacing = 0.12  # Adjust spacing for better visual separation
    x = np.arange(len(countries_na)) + 1  # Shift positions to the right

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    colors = ['#2E7BA6', '#C88FBD', '#2ca02c', '#6A5ACD', '#C99436']
    labels = ['FCNN', 'KNN', 'LR', 'RF', 'Ours']
    data = [FCNN_na, KNN_na, LR_na, RF_na, ours_na]

    for i, (label, color) in enumerate(zip(labels, colors)):
        if color == '#C99436':
            ax.bar(x - i * (bar_width / 2 + spacing / 2) - 2.75 * bar_width, data[i], width=bar_width, color=color,
                   label=label)
        else:
            ax.bar(x - i * (bar_width / 2 + spacing / 2) - 2.75 * bar_width, data[i], width=bar_width, color=color,
                   alpha=0.5, label=label)

    ax.set_title('North America', fontsize=26, fontweight='bold')
    ax.set_ylabel('MCC', fontsize=20)
    #ax.set_xticks(x + 2 * (bar_width + spacing) / 2)
    #ax.set_xticklabels(countries_asia, fontsize=17)
    ax.set_xticks([0.5,1.5])
    ax.set_xticklabels(['US', 'Canada'],fontsize=22)
    ax.set_xlim(-0.2,2.2)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5,alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [4, 0, 1, 2, 3]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 11}, loc='lower right')

    ax.tick_params(axis='both', which='both', direction='in', labelsize=18)

    plt.tight_layout()
    plt.savefig(r'C:\Users\DELL-USER02\Desktop\写论文\论文中的图\柱状图\柱状图_northamerica.png', format='png')
    plt.show()

plot_and_save_na()


# Data for Europe
countries_europe = ['Austria', 'France', 'Germany', 'UK', 'Italy']
FCNN_europe = [0.1753, 0.33718, 0.37413, 0.43606, 0.0035]
KNN_europe = [0.6548, 0.4972, 0.78233, 0.7787, 0.6571]
LR_europe = [0.7038, 0.73767, 0.28947, 0.47037, 0.4989]
RF_europe = [0.6385, 0.75486, 0.68252, 0.56049, 0.7062]
ours_europe = [0.8028, 0.8281, 0.8617, 0.8706, 0.9334]

def plot_and_save_europe():
    bar_width = 0.1
    spacing = 0.015
    x = np.arange(len(countries_europe))

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    colors = ['#C99436', '#2E7BA6', '#C88FBD', '#2ca02c', '#6A5ACD']
    labels = ['Ours', 'FCNN', 'KNN', 'LR', 'RF']
    data = [ours_europe, FCNN_europe, KNN_europe, LR_europe, RF_europe]

    for i, (label, color) in enumerate(zip(labels, colors)):
        if color == '#C99436':
            ax.bar(x + i * (bar_width + spacing), data[i], width=bar_width, color=color,  label=label)
        else:
            ax.bar(x + i * (bar_width + spacing), data[i], width=bar_width, color=color, alpha=0.5, label=label)

    ax.set_title('Europe', fontsize=26, fontweight='bold')
    ax.set_ylabel('MCC', fontsize=20)
    ax.set_xticks(x + 2 * (bar_width + spacing) / 2)
    ax.set_xticklabels(countries_europe, fontsize=19)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5,alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3, 4]  # Updated order to match new labels order
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 10}, loc='lower right')

    ax.tick_params(axis='both', which='both', direction='in', labelsize=18)

    plt.tight_layout()

    plt.show()


plot_and_save_europe()
