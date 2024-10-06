import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

selected_data = {
    "Bank": ["BAC", "WFC", "MS", "CADE", "KEY"],
    "Initial Risk": [0.201923581, 0.389680177,0.380893916, 0.427784967, 0.470501029],
    "Now Risk": [0.028518915, 0.1832, 0.253024936,0.274374127,  0.224595115],
    "Rescue Ratio": [0, 0.028518915, 0.024362894, 0.017697077, 0.052861322]
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
'#3583B4'  '#ED831E'
bars1 = ax1.bar(index - bar_width/2, df_selected['Initial Risk'], bar_width, label='Pre-Rescue Risk', color='#E05C34')
bars2 = ax1.bar(index + bar_width/2, df_selected['Now Risk'], bar_width, label='Post-Rescue Risk', color='#F4B183')

# Set titles and labels
ax1.set_xlabel('Financial Institutions',fontsize=20)
ax1.set_ylabel('Risk (%)',fontsize=20)
ax1.set_title('Financial Risk (US)',fontsize=22)
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

# Annotate the differences in percentage
# for i in range(len(df_selected)):
#     diff = df_selected['Initial Risk'][i] - df_selected['Now Risk'][i]
#     ax1.annotate(f'{diff:.2f}%', xy=(i, df_selected['Initial Risk'][i]), xytext=(i, df_selected['Initial Risk'][i] + 0.3),
#                  textcoords='data', ha='center', va='bottom', fontsize=11, color='black')
#     ax1.arrow(index[i] + bar_width/2, df_selected['Now Risk'][i], 0, diff, head_width=0.1, head_length=0.15, alpha=0.3)

# Add data labels for Rescue Ratio
for i, txt in enumerate(df_selected['Rescue Ratio']):
    ax2.text(i, txt+0.2, f'{round(txt, 2)}%', ha='center', va='bottom', fontsize=11)

ax1.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax2.tick_params(axis='both', which='both', direction='in', labelsize=15)
# Show the legends outside the plot area
fig.legend(loc="upper left", bbox_to_anchor=(0.13, 0.92),fontsize=12)
# Save and display the plot
plt.tight_layout()  # Adjust layout to make space for the legend


# plt.savefig("/mnt/data/narrow_bars_selected_bank_risk_rescue_ratio_percentage_no_overlap.png")
plt.show()
