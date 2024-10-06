# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
file_path = 'data.csv'  # 替换为你的CSV文件路径

# 读取CSV文件
data = pd.read_csv(file_path)

# 提取数据列
# 假设列名为 'Threshold', 'Max_Nodes' 和 'Second_Max_Nodes'
threshold = data['阈值']
max_nodes = data['最大连通子图']
second_max_nodes = data['第二大连通子图']
plt.rcParams['font.family'] = 'Times New Roman'
# 创建画布
plt.figure(figsize=(6, 5))
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制数据
plt.plot(threshold, max_nodes, linestyle='-', label='Max Connected Subgraph', markersize=4)
plt.plot(threshold, second_max_nodes, linestyle='-', label='Second-largest Connected Subgraph', markersize=5)

# 标出特定阈值的点
highlight_thresholds = [0.00001, 0.00005]

# 查找最接近的阈值
def find_nearest(thresholds, value):
    array = np.asarray(thresholds)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

highlight_max_nodes = []
highlight_second_max_nodes = []
highlight_nearest_thresholds = []

for t in highlight_thresholds:
    nearest_t = find_nearest(threshold, t)
    highlight_nearest_thresholds.append(nearest_t)
    idx = threshold[threshold == nearest_t].index[0]  # 获取索引
    highlight_max_nodes.append(max_nodes[idx])
    highlight_second_max_nodes.append(second_max_nodes[idx])

# 绘制标记点
plt.scatter(highlight_nearest_thresholds, highlight_max_nodes, color='r', s=100, edgecolor='black', zorder=5)
plt.scatter(highlight_nearest_thresholds, highlight_second_max_nodes, color='m', s=100, edgecolor='black', zorder=4)

# 添加文本标签
for t, max_n, sec_max_n in zip(highlight_nearest_thresholds, highlight_max_nodes, highlight_second_max_nodes):
    plt.text(t + 0.000001, max_n, f'({t:.2e}, {max_n})', fontsize=15, color='r', ha='left', va='center')


# 设置标题和标签

plt.xlabel('Threshold', fontsize=22)
plt.ylabel('Nodes', fontsize=24)
# 设置x轴刻度


# 设置网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)  # alpha 控制网格的透明度

# 设置图例
plt.legend(loc='upper right', fontsize=13.2)

# 设置坐标轴刻度为内刻度
plt.tick_params(axis='both', which='both', direction='in',labelsize=16)

# 调整子图边距
plt.subplots_adjust(bottom=0.2)

# 加粗坐标轴线条
ax = plt.gca()  # 获取当前轴
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.tight_layout()
# 保存图形
output_path = r'Nodes_Threshold_plot.png'
plt.savefig(output_path, format='png')

# 展示图表
plt.show()
