# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# 数据
ratio = [10, 40, 75]
mcc_france = [0.5301, 0.8281, 0.9222]
mcc_germany = [0.7452, 0.971, 0.9334]
mcc_italy = [0.6205, 0.7439, 0.8617]
mcc_austria = [0.1, 0.42, 0.8029]
mcc_UK = [0.4019, 0.8245, 0.8705]
mcc_china = [0.7185, 0.8098, 0.9565]
mcc_japan = [0.5121, 0.5789, 0.7511]
mcc_america = [0.4713, 0.6562, 0.8864]
mcc_canada = [0.3561, 0.4983, 0.7782]

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

ax1.scatter(ratio, mcc_america, c='red', marker='s', s=30, label='US')
ax1.scatter(ratio, mcc_canada, c='blue', marker='o', s=30, label='Canada')
ax1.set_title('MCC for North America', fontsize=16,fontweight='bold')
ax1.set_xlabel('Percentage of pre-labeled nodes', fontsize=14)
ax1.set_ylabel('MCC', fontsize=14)
ax1.set_xticks(ratio)
ax1.set_xticklabels(ratio, fontsize=14)
ax1.set_ylim(0, 1)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax1.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax1.spines['top'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.legend(prop={'size': 12}, loc='lower right')
ax1.tick_params(axis='both', which='both', direction='in')  # 设置坐标轴刻度为内刻度

# 绘制欧洲数据的散点图
ax2.scatter(ratio, mcc_germany, c='red', marker='s', s=20, label='Germany')
ax2.scatter(ratio, mcc_france, c='blue', marker='o', s=20, label='France')
ax2.scatter(ratio, mcc_italy, c='green', marker='^', s=20, label='Italy')
ax2.scatter(ratio, mcc_austria, c='orange', marker='D', s=20, label='Austria')
ax2.scatter(ratio, mcc_UK, c='purple', marker='P', s=20, label='UK')
ax2.set_title('MCC for Europe', fontsize=16,fontweight='bold')
ax2.set_xlabel('Percentage of pre-labeled nodes', fontsize=14)
ax2.set_ylabel('MCC', fontsize=14)
ax2.set_xticks(ratio)
ax2.set_xticklabels(ratio, fontsize=14)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax2.spines['top'].set_linewidth(1)
ax2.spines['right'].set_linewidth(1)
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1)
ax2.legend(prop={'size': 9}, loc='lower right')
ax2.tick_params(axis='both', which='both', direction='in')  # 设置坐标轴刻度为内刻度

# 绘制亚洲数据的散点图
ax3.scatter(ratio, mcc_china, c='red', marker='^', s=30, label='China')
ax3.scatter(ratio, mcc_japan, c='blue', marker='D', s=30, label='Japan')
ax3.set_title('MCC for Asia', fontsize=16,fontweight='bold')
ax3.set_xlabel('Percentage of pre-labeled nodes', fontsize=14)
ax3.set_ylabel('MCC', fontsize=14)
ax3.set_xticks(ratio)
ax3.set_xticklabels(ratio, fontsize=14)
ax3.set_ylim(0, 1)
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax3.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax3.spines['top'].set_linewidth(1)
ax3.spines['right'].set_linewidth(1)
ax3.spines['bottom'].set_linewidth(1)
ax3.spines['left'].set_linewidth(1)
ax3.legend(prop={'size': 12}, loc='lower right')
ax3.tick_params(axis='both', which='both', direction='in')

plt.tight_layout()

plt.show()
