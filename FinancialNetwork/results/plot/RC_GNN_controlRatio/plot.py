# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
# us canada china japan france germany uk austria italy
df = np.array([[0.3432, 0.3052, 0.2688, 0.2362, 0.2156],
               [0.4009, 0.3821, 0.3387, 0.3114, 0.3009],
               [0.4485, 0.4154, 0.4018, 0.3445, 0.3120],
               [0.2914, 0.2698, 0.24829, 0.2034, 0.1362],
               [0.3793, 0.2723, 0.2224, 0.1768, 0.1297],#Franc 0.2496
               [0.4569, 0.4357, 0.3867, 0.3692, 0.3075],#Germany 0.1494
               [0.1821, 0.1519, 0.1157, 0.0803, 0.0721], #UK 0.11
               [0.3920, 0.3661, 0.3207, 0.3091, 0.2664], #austria 0.1256
               [0.4951, 0.4642, 0.4357, 0.4229, 0.3870]]) #Italy 0.11
# boc bac WFC hwc hsbc
df_1 = np.array([[0.2676, 0.2376, 0.2276, 0.1724, 0.1574],
                 [0.2019, 0.1437, 0.0622, 0.0312, 0.0243],
                 [0.3896, 0.3071, 0.2729, 0.2265, 0.1832],
                 [0.2120, 0.1812, 0.1420, 0.1322, 0.1102],
                 [0.2678, 0.2489, 0.1876, 0.1671, 0.1478]
                 ])

x = np.linspace(0, 1, 5)
def custom_percent_formatter(x, pos):
    if x == 0:
        return '0'
    return f'{x:.1f}'



def percent_formatter(value, tick_position):
    return '{:.0f}'.format(value * 100)


# 绘制图表1
config = {
    "font.family": 'serif',
    "font.size": 15,
    "mathtext.fontset": 'stix',
    'font.family': 'Times New Roman, Simhei'
}
plt.rcParams.update(config)

fig, ax = plt.subplots(figsize=(5, 4), dpi=500)



ax.plot(x, df[0], 'o-', linestyle='--', color='#4995C6', label='US', markersize=6)
ax.plot(x, df[1], 'o-', linestyle='--', color='#1663A9', label='Canada', markersize=6)
ax.plot(x, df[2], 's-', linestyle='-', color='#BD3646', label='China', markersize=6)
ax.plot(x, df[3], 's-', linestyle='-', color='#369f2d', label='Japan', markersize=6)


ax.set_xticks([0, 0.5, 1])



ax.set_xticks([0, 0.5, 1])
ax.xaxis.set_major_formatter(FuncFormatter(custom_percent_formatter))
ax.set_ylim(0, 0.55)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.set_xlabel('Overall Rescue Ratio')
ax.set_ylabel('Overall Risk(%)')

ax.tick_params(axis='both', which='both', direction='in')
ax.grid(axis='both', linestyle='--', alpha=0.5, color='gray')
plt.legend(fontsize=10,loc='upper right')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))

plt.savefig('tu-1.png', bbox_inches='tight')
plt.tight_layout()
plt.show()


# 绘制图表3
config = {
    "font.family": 'serif',
    "font.size": 15,
    "mathtext.fontset": 'stix',
    'font.family': 'Times New Roman, Simhei'
}
plt.rcParams.update(config)

fig, ax = plt.subplots(figsize=(5, 4), dpi=500)




ax.plot(x, df_1[2], '^-',  linestyle='--',color='#92C2DD', label='WFC(US)', markersize=6)
ax.plot(x, df_1[1], 's-', linestyle='--', color='#4995C6', label='BAC(US)', markersize=6)
ax.plot(x, df_1[3], 'v-',  linestyle='--',color='#1663A9', label='HWC(US)', markersize=6)


ax.plot(x, df_1[4], 'D-', linestyle='-.',color='#FABB6E', label='HSBC(UK)', markersize=6)

ax.plot(x, df_1[0], 'o-', linestyle='-', color='#BD3646', label='BOC(CN)', markersize=6)





ax.set_xticks([0, 0.5, 1])
ax.xaxis.set_major_formatter(FuncFormatter(custom_percent_formatter))

ax.set_ylim(0, 0.55)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.set_xlabel('Overall Rescue Ratio')
ax.set_ylabel('Risk(%)')

ax.tick_params(axis='both', which='both', direction='in')
ax.grid(axis='both', linestyle='--', alpha=0.5, color='gray')
plt.legend(fontsize=10, loc='upper right')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))

plt.savefig('tu-3.png', bbox_inches='tight')
plt.tight_layout()
plt.show()


def custom_percent_formatter(x, pos):
    if x == 0:
        return '0'
    return f'{x:.1f}'




# 绘制图表4
config = {
    "font.family": 'serif',
    "font.size": 15,
    "mathtext.fontset": 'stix',
    'font.family': 'Times New Roman, Simhei'
}
plt.rcParams.update(config)

fig, ax = plt.subplots(figsize=(5, 4), dpi=500)


ax.plot(x, df[4], '^-', color='#3e6691', label='France', markersize=6)
ax.plot(x, df[5], 's-', color='#9b8cac', label='Germany', markersize=6)
ax.plot(x, df[6], 'v-', color='#498e49', label='UK', markersize=6)
ax.plot(x, df[7], 'D-', color='#d47b3f', label='Austria', markersize=6)
ax.plot(x, df[8], 'o-', color='#be3848', label='Italy', markersize=6)






#
# ax.set_xticks(x)

# Adjust x-axis ticks
ax.set_xticks([0, 0.5, 1])
ax.xaxis.set_major_formatter(FuncFormatter(custom_percent_formatter))

ax.set_ylim(0, 0.55)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax.set_xlabel('Overall Rescue Ratio')
ax.set_ylabel('Overall Risk(%)')

ax.tick_params(axis='both', which='both', direction='in')

ax.grid(axis='both', linestyle='--', alpha=0.55, color='gray')
plt.legend(fontsize=8, loc='lower left')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))

plt.savefig('tu-EUrpoe.png', bbox_inches='tight')
plt.tight_layout()
plt.show()


