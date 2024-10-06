# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'


selected_data = {
    "Bank": ["ABC", "CCB", "BOXIA","BOCD", "BOCQ"],
    "Initial Risk": [ 0.372360468,0.447564811, 0.403804094, 0.420745873, 0.483227313],
    "Now Risk": [ 0.23732466, 0.319730078,0.280723457, 0.34144253,0.362198805],
    "Rescue Ratio": [0,0.003133, 0.01207371, 0.018124059, 0.04999836534]
}

df_selected = pd.DataFrame(selected_data)

# Convert the values to percentages
df_selected['Initial Risk'] = df_selected['Initial Risk'] * 100
df_selected['Now Risk'] = df_selected['Now Risk'] * 100
df_selected['Rescue Ratio'] = df_selected['Rescue Ratio'] * 100

# Create the enhanced plot with percentage values
fig, ax1 = plt.subplots(figsize=(6, 5))  # Canvas size 6x5 inches

# Plot Initial Risk and Now Risk as percentages with narrower bars
bar_width = 0.2  # Narrower bars
index = np.arange(len(df_selected))

bars1 = ax1.bar(index - bar_width/2, df_selected['Initial Risk'], bar_width, label='Pre-Rescue Risk', color='#BD3646')
bars2 = ax1.bar(index + bar_width/2, df_selected['Now Risk'], bar_width, label='Post-Rescue Risk', color='#D69C7F')

# Set titles and labels

ax1.set_ylabel('Risk (%)',fontsize=20)
ax1.set_title('Financial Risk (China)',fontsize=22)
ax1.set_ylim(0,60)
ax1.set_yticks([0,20,40,60])
ax1.set_xticks(index)
ax1.set_xticklabels(df_selected['Bank'])
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot Rescue Ratio as percentage
ax2 = ax1.twinx()
ax2.set_ylabel('Rescue Ratio (%)',fontsize=20)
ax2.set_ylim(0,9)
ax2.set_yticks([0,3,6,9])
#ax2.set_yticklabels([0,3,6,9,12])
line = ax2.plot(index, df_selected['Rescue Ratio'], color='green', marker='o', linewidth=2, label='Rescue Ratio')


# Add data labels for Rescue Ratio
for i, txt in enumerate(df_selected['Rescue Ratio']):
    ax2.text(i, txt+0.2, f'{round(txt, 2)}%', ha='center', va='bottom', fontsize=11)

ax1.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax2.tick_params(axis='both', which='both', direction='in', labelsize=15)
# Show the legends outside the plot area
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.92),fontsize=12)
# Save and display the plot
plt.tight_layout()  # Adjust layout to make space for the legend

plt.show()
