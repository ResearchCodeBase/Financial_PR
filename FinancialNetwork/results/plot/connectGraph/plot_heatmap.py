import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import os

data = np.array([
    [0	,0.00168017	,0.003191325	,0.013209998,	0.005111806,	0.008794658	,0.009401605	,0.00356357	,0.000570807,	0.004994248
],
    [0.000381855,	0	,0.000254241	,0.001052391	,0.000407238	,0.000700637,	0.00074899,	0.000283896	,4.55E-05	,0.000397873
],
    [0.000138816	,4.87E-05	,0	,0.000382578	,0.000148044,	0.000254704,	0.000272282	,0.000103205	,1.65E-05	,0.00014464
],
    [0.005284519,	0.001852399	,0.003518459	,0	,0.005635804,	0.009696175	,0.010365339,	0.003928862	,0.000629319	,0.005506196
],
    [0.005485035	,0.001922687	,0.003651964	,0.015116742	,0	,0.010064088	,0.010758643	,0.004077939	,0.000653198	,0.005715123
],
    [0.006210795,	0.00217709	,0.004135179	,0.017116933	,0.006623653,	0	,0.012182186	,0.004617517	,0.000739627,	0.006471327
],
    [0.005484711	,0.001922573,	0.003651749	,0.01511585,	0.005849304	,0.010063494,	0,	0.004077698	,0.000653159,	0.005714786
],
    [0.003498443	,0.00122632	,0.002329281	,0.009641699,	0.003731	,0.006419036,	0.006862034,	0	,0.00041662	,0.003645197
],
    [0.000777461	,0.000272526	,0.000517638,	0.002142681	,0.000829142,	0.001426506	,0.001524954,	0.000578016,	0,	0.000810074
],
    [0.00280454	,0.000983085,	0.001867277	,0.007729306	,0.002990971	,0.005145845,	0.005500976	,0.002085081	,0.000333985	,0
],

])

bank_names = [
    "CDB", "CEXIM", "ADBC", "BOC", "CCB", "ICBC", "ABC", "IB", "CGB", "BCM"
]



plt.rcParams['axes.unicode_minus'] = False  # This is needed to display the minus sign correctly
plt.rcParams['font.family'] = 'Times New Roman'

# Define section size
section_size = (10, 10)  # Since the data is 10x10, this will display the entire matrix

# Create a heatmap using seaborn with the specified color palette
plt.figure(figsize=(10, 8))
mask = np.ones_like(data, dtype=bool)  # Start with a mask that hides everything

# Define the area to display (top-left corner)
display_area = (slice(section_size[0]), slice(section_size[1]))
mask[display_area] = False  # Reveal only the section specified

# Apply the mask to the data as well to match the user's requirements
data_to_display = data[display_area]

# Plot the section of the data that is not masked
with sns.axes_style("white"):
    ax = sns.heatmap(data_to_display, annot=False, cmap="RdBu_r",
                     xticklabels=bank_names[:section_size[0]], yticklabels=bank_names[:section_size[1]],
                     cbar_kws={'shrink': 1})  # Adjust colorbar size if needed

    plt.xticks(rotation=45, ha='right', fontsize=22)
    plt.yticks(rotation=0, ha='right', fontsize=22)
    plt.title("Transaction Probability Heatmap", fontsize=28, fontweight='bold')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)  # Set the font size of colorbar ticks

# Save the heatmap
output_path_asia = r'C:\Users\DELL-USER02\Desktop\写论文\论文中的图\热力图\热力图.png'
plt.tight_layout()
plt.savefig(output_path_asia, format='png')

plt.show()