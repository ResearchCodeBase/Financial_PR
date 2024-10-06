import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file1 = 'liquidity_data_SDE.csv'
file2 = 'liquidity_data_noSDE.csv'


data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)


SDE_data = data1['Total Liquidity']
baseline_data = data2['Total Liquidity']


plt.figure(figsize=(6, 5))
plt.rcParams['font.family'] = 'Times New Roman'

plt.plot(range(len(SDE_data)), SDE_data, linestyle='-', color='forestgreen', label='SDE', markersize=4)
plt.plot(range(len(baseline_data)), baseline_data, linestyle='-', color='firebrick', label='Baseline', markersize=4)


plt.xlabel('Iterations', fontsize=22)
plt.ylabel('Liquidity', fontsize=22)


plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


plt.grid(True, linestyle='--', linewidth=0.3, color='gray')

plt.legend(loc='upper right', fontsize=22)


plt.tick_params(axis='both', which='both', direction='in')


plt.subplots_adjust(bottom=0.2)


ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.tight_layout()

plt.show()
