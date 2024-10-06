# coding=gbk
import argparse
import os
from datetime import datetime

import networkx as nx
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import seaborn as sns



def iteration1(x0, L, A, eps):

    n, m = x0.shape
    delta = float('inf')
    iter = 0
    converge = []

    while delta > eps and iter < 500:

        row_sum = x0.sum(axis=1)
        rho = np.where(row_sum > 0, L / row_sum, 0)
        x1 = x0 * rho[:, np.newaxis]


        col_sum = x1.sum(axis=0)
        xu = np.where(col_sum > 0, A / col_sum, 0)
        x2 = x1 * xu


        delta = np.linalg.norm(x2 - x0, 'fro') / np.linalg.norm(x0, 'fro')
        converge.append(delta)

        x0 = x2
        iter += 1

    return x0,  converge

def calculate_debt_ratio_fullmatrix(x0, proportion_L, proportion_A, L,A,eps):
    x_result, convergence = iteration1(x0, proportion_L, proportion_A, eps)

    row_result = np.sum(x_result, axis=1)
    col_result = np.sum(x_result, axis=0)

    xA_result = x_result * total_A
    sumA_result = np.sum(xA_result)


    print('*****************ծ����ȫ����Ľ��*********************')

    print("ծ�����L���к��ܺͣ��Ǳ���:", np.sum(row_result) )
    print("ծ�����L���к��ܺͣ��Ǳ���:", np.sum(col_result))
    print("ծ������ܺͣ���λ�ǰ���:", sumA_result)
    print("ծ�����L���к��ܺͣ��ǰ���:", np.sum(np.sum(xA_result, axis=1)))
    print("ծ�����L���к��ܺͣ��ǰ���:",  np.sum(np.sum(xA_result, axis=0)))

    # ������������
    plt.plot(convergence)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence of Iteration')
    plt.grid(True)
    plt.show()


    row_error = np.maximum(L - np.sum(xA_result , axis=1), 0)
    col_error = np.maximum(A - np.sum(xA_result , axis=0), 0)
    print(np.sum(row_error))
    print( np.sum(col_error))
    result1 = np.vstack([xA_result , col_error])
    row_error_column = row_error[:, np.newaxis]
    result2 = np.hstack([result1, np.vstack([row_error_column, [0]])])

    result_round = np.round(result2).astype(int)

    print('����������к��к���ֵ',np.sum(np.sum( result_round, axis=1)))
    print('�������к���ֵ',np.sum(L))
    print('����������к��к���ֵ', np.sum(np.sum(result_round, axis=0)))
    print('�������к���ֵ',total_A)


    return result_round, xA_result

def calculate_debt_ratio_preferencematrix(x0, proportion_L, proportion_A, A_indices,B_indices,C_indices,eps):

    Q1 = Q2 = 1
    Q3, Q4, Q5 = 0.8, 0.6, 0.4  # ��Щֵ��Ҫ�������ľ����������

    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            if i == j :  # Zero out values for self-relationships or C-category banks
                x0[i, j] = 0
            elif (i in A_indices and j in B_indices) or (i in B_indices and j in A_indices):
                x0[i, j] *= Q3
            elif (i in A_indices and j in C_indices)or (i in C_indices and j in A_indices):
                x0[i, j] *= Q4
            elif (i in B_indices and j in C_indices) or (i in C_indices and j in B_indices):
                x0[i, j] *= Q5
    # RAS algorithm
    x_result, convergence = iteration1(x0, proportion_L, proportion_A, eps)

    row_result = np.sum(x_result, axis=1)
    col_result = np.sum(x_result, axis=0)

    xA_result = x_result * total_A
    sumA_result = np.sum(xA_result)


    print("ծ�����L���к��ܺͣ��Ǳ���:", np.sum(row_result) )
    print("ծ�����L���к��ܺͣ��Ǳ���:", np.sum(col_result))
    print("ծ����󣨵�λ�ǰ���:", sumA_result)
    print("ծ�����L���к��ܺͣ��ǰ���:", np.sum(sumA_result) )
    print("ծ�����L���к��ܺͣ��ǰ���:", np.sum(sumA_result))

    # ����������
    plt.plot(convergence)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence of Iteration')
    plt.grid(True)
    plt.show()
    return xA_result

def data_process(L_path,A_path,name_path):

    L = np.loadtxt(L_path)


    A = np.loadtxt(A_path)

    banks_df = pd.read_csv(name_path, encoding='utf-8')


    total_L = np.sum(L)
    total_A = np.sum(A)


    proportion_L = L / total_L
    proportion_A = A / total_A

    # ��׼�� ������ȷ
    x = np.zeros((L.shape[0], A.shape[0]))

    # ѭ���������x��ÿ��Ԫ��
    for i in range(proportion_L.shape[0]):
        for j in range(proportion_A.shape[0]):
            x[i, j] = proportion_L[i] * proportion_A[j]


    print('proportion_L',np.sum(proportion_L))
    print('proportion_A', np.sum(proportion_A))

    # ������ x �ĶԽ�������Ϊ 0
    x0 = x - np.diag(np.diag(x))
    print('x[0][0],x[5][5]',x0[0][0],x0[5][5])
    return x0, proportion_L,proportion_A,total_A,banks_df,L,A

def classify_banks(banks_df):
    # Define the bank categories
    A_banks = ['�й�����', '�й���������', '�й���������', '�й�ũҵ����','��ͨ����','�й�������������']
    B_banks = [
        '��������', '����������', 'ũҵ��չ����',
        '��������', '�ַ�����', '��������', '�������',
        '��������', '�й���������', '�㷢����', '��ҵ����',
        'ƽ������', '��������', '�������', '��������'
    ]
    # Initialize the indices lists
    A_indices = []
    B_indices = []
    C_indices = []
    for index, row in banks_df.iterrows():
        bank_name = row['��������']  # Assuming the column name is 'BankName'
        if bank_name in A_banks:
            A_indices.append(index)
        elif bank_name in B_banks:
            B_indices.append(index)
        else:
            C_indices.append(index)

    for idx in A_indices:
        print(banks_df.loc[idx, '��������'])

    for idx in B_indices:
        print(banks_df.loc[idx, '��������'])



    return A_indices, B_indices, C_indices



def apply_threshold_and_calculate_edges(xA_result, L, A, name,e_path,threshold):


    xA_normalized = xA_result / np.sum(xA_result)
    XA_rowsum = np.sum(xA_result, axis=1)
    xA_thresholded = np.where(xA_normalized >= threshold,xA_result, 0)

    mac,second=calculate_connected_components(xA_thresholded)
    density = calculate_edge_density(xA_thresholded)
    num_edges = np.count_nonzero(xA_thresholded)
    total_edges = xA_thresholded.shape[0] ** 2 - xA_thresholded.shape[0]
    isolated_nodes = np.where(np.all(xA_thresholded == 0, axis=1) & np.all(xA_thresholded == 0, axis=0))[0]
    xA_after_isolation_removal = np.delete(np.delete(xA_thresholded, isolated_nodes, axis=0), isolated_nodes, axis=1)

    row_error = np.maximum(L - np.sum(xA_thresholded, axis=1), 0)
    col_error = np.maximum(A - np.sum(xA_thresholded, axis=0), 0)
    result1 = np.vstack([xA_thresholded, col_error])
    row_error_column = row_error[:, np.newaxis]
    result2 = np.hstack([result1, np.vstack([row_error_column, [0]])])

    # ȡ��
    result_round = np.round(result2).astype(int)
    result_roundremove = np.delete(np.delete(result_round, isolated_nodes, axis=0), isolated_nodes, axis=1)

    name_after_isolation_removal = np.delete(name, isolated_nodes)
    e=np.loadtxt(e_path)

    e_data_after_isolation_removal = np.delete(e,isolated_nodes)

    return xA_after_isolation_removal,result_roundremove,density,name_after_isolation_removal, e_data_after_isolation_removal
def find_isolated_nodes(result_round):
    adjacency_matrix = (result_round != 0).astype(int)
    isolated_nodes = []
    for i in range(adjacency_matrix.shape[0]):
        if np.all(adjacency_matrix[i, :] == 0):
            isolated_nodes.append(i)

    # ������Ƿ�Ϊ�����ڵ�
    for j in range(adjacency_matrix.shape[1]):
        if np.all(adjacency_matrix[:, j] == 0):
            isolated_nodes.append(-j)
    print(isolated_nodes)
    return isolated_nodes

def calculate_connected_components(matrix):
    # Create a graph from the numpy matrix
    G = nx.Graph(matrix)
    # Find the connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    # Get the sizes of the largest and second largest connected components
    max_cc_size = len(connected_components[0]) if connected_components else 0
    second_cc_size = len(connected_components[1]) if len(connected_components) > 1 else 0
    return max_cc_size, second_cc_size

def determine_threshold(xA_result, threshold_range= np.linspace(0, 0.01, 1000)):
    print('**************��ʼѰ���������ֵ*****************')
    densities = []
    max_cc_sizes = []
    second_cc_sizes = []

    xA_normalized = xA_result / np.sum(xA_result)

    XA_rowsum = np.sum(xA_result, axis=1)

    for threshold in threshold_range:
        xA_thresholded = np.where(xA_normalized >= threshold, xA_result, 0)
        density = calculate_edge_density(xA_thresholded)
        max_cc_size, second_cc_size = calculate_connected_components( xA_thresholded )
        densities.append(density)
        max_cc_sizes.append(max_cc_size)
        second_cc_sizes.append(second_cc_size)


    return densities,max_cc_sizes, second_cc_sizes
def calculate_edge_density(xA_thresholded):

    num_edges = np.count_nonzero(xA_thresholded)
    total_edges = xA_thresholded.shape[0] ** 2 - xA_thresholded.shape[0]
    density = num_edges / total_edges
    return density


def plot_convergence(convergence, font):

    plt.figure(figsize=(10, 6))
    plt.plot(convergence)
    plt.xlabel('��������', fontproperties=font)
    plt.ylabel('���', fontproperties=font)
    plt.title('�����Ż����̵��������', fontproperties=font)
    plt.grid(True)
    plt.show()


def plot_edge_density(densities, specific_thresholds, font, save_path, threshold_range=np.linspace(0, 0.01, 1000)):

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_range, densities, label='Edge Density')

    # ͻ����ʾ�����ض���
    for specific_threshold in specific_thresholds:
        specific_index = np.argmin(np.abs(threshold_range - specific_threshold))
        specific_density = densities[specific_index]
        plt.scatter([specific_threshold], [specific_density], color='red')  # ͻ����ʾ�ض���
        plt.text(specific_threshold, specific_density, f'({specific_threshold:.5f}, {specific_density:.2f})',
                 color='red', fontproperties=font)

    plt.xlabel('Threshold', fontproperties=font)
    plt.ylabel('Edge Density', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)
    if save_path:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path_with_time = os.path.join(save_path, f"���ܶ�_{current_time}.png")
        plt.savefig(save_path_with_time)  # Save the heatmap to the specified path with time

    plt.show()


def plot_connected_components(max_cc_sizes, second_cc_sizes, x_values, font, save_path,
                              threshold_range=np.linspace(0, 0.01, 1000)):

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_range, max_cc_sizes, label='Maximal Connected Subgraph')
    plt.plot(threshold_range, second_cc_sizes, label='Second Largest Connected Subgraph')

    # ͻ����ʾ�����ض��ĵ�
    for x_value in x_values:
        index = np.where(threshold_range >= x_value)[0][0]
        y_value_max_cc = max_cc_sizes[index]
        y_value_second_cc = second_cc_sizes[index]

        plt.scatter([x_value], [y_value_max_cc], color='red')
        plt.text(x_value, y_value_max_cc, f'({x_value:.5f}, {y_value_max_cc:.2f})', color='red', fontproperties=font)


    plt.xlabel('Threshold', fontproperties=font)
    plt.ylabel('Number of Nodes', fontproperties=font)

    plt.legend(prop=font)
    plt.grid(True)
    if save_path:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        save_path_with_time = os.path.join(save_path, f"��ͨ��ͼ_{current_time}.png")
        plt.savefig(save_path_with_time)  # Save the heatmap to the specified path with time
        print("��ͨ��ͼ����·��:", save_path_with_time)

    plt.show()

def plot_matrix_comparison(full_matrix, preferencematrix):

    full_matrix = full_matrix/ np.sum(full_matrix)
    preferencematrix = preferencematrix / np.sum(preferencematrix)
    plt.figure(figsize=(12, 6))

    # Plotting the full_matrix
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.title('Full Matrix')
    plt.imshow(full_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()

    # Plotting the preferencematrix
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.title('Preference Matrix')
    plt.imshow(preferencematrix, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def plot_heatmap(data,banks_df, save_path,title, section_size=(10, 10)):

    # Create a heatmap using seaborn with the specified color palette
    data = data / np.sum(data)
    plt.figure(figsize=(10, 6))
    mask = np.ones_like(data, dtype=bool)  # Start with a mask that hides everything

    # Define the area to display (top-left corner)
    display_area = (slice(section_size[0]), slice(section_size[1]))
    mask[display_area] = False  # Reveal only the 20x20 section in the top-left corner

    # We apply the mask to the data as well to match the user's requirements.
    # This will only display the top-left 20x20 section of the data.
    data_to_display = data[display_area]
    bank_names = banks_df['��������'].iloc[:section_size[0]].tolist()
    # Set the font to support Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' is a commonly used Chinese font
    plt.rcParams['axes.unicode_minus'] = False  # This is needed to display the minus sign correctly

    with sns.axes_style("white"):
        # Only plot the section of the data that is not masked
        sns.heatmap(data_to_display, annot=True, cmap="RdBu_r",
                    xticklabels=bank_names, yticklabels=bank_names)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    plt.title(title)

    if save_path:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        save_path_with_time = os.path.join(save_path, f"���Ͻ�_{current_time}.png")
        plt.savefig(save_path_with_time)  # Save the heatmap to the specified path with time
        print("���Ͻ�����ͼ����·��:", save_path_with_time)

    plt.show()


def plot_heatmap1(data, banks_df, save_path,title, section_size=(10, 10)):

    # Create a heatmap using seaborn with the specified color palette
    data = data / np.sum(data)
    plt.figure(figsize=(10, 6))
    mask = np.ones_like(data, dtype=bool)  # Start with a mask that hides everything

    data_to_display = data[-10:, :10]
    # Get the bank names for the y-axis (last 10) and x-axis (first 10)
    bank_names_y = banks_df['��������'].iloc[-section_size[0]:].tolist()
    bank_names_x = banks_df['��������'].iloc[:section_size[1]].tolist()
    # Set the font to support Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' is a commonly used Chinese font
    plt.rcParams['axes.unicode_minus'] = False  # This is needed to display the minus sign correctly

    with sns.axes_style("white"):
        # Only plot the section of the data that is not masked
        sns.heatmap(data_to_display, annot=True, cmap="RdBu_r",
                    xticklabels=bank_names_x, yticklabels= bank_names_y)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    plt.title(title)
    if save_path:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path_with_time = os.path.join(save_path, f"���½�_{current_time}.png")
        plt.savefig(save_path_with_time)  # Save the heatmap to the specified path with time
    plt.show()
if __name__ == '__main__':
    # ��ʼ��������
    # ����һ���������ڴ洢��� �ָ�ծ����󣬲���Ҫ��������
    year = '2022'
    country = 'America'

    parser = argparse.ArgumentParser(description='Process Banking Network Features')
    parser.add_argument('--L_path', type=str, default=f'data/{country}_{year}/ԭʼ����/�����ʽ�_{year}.txt',
                        help='Path to the liability matrix file')
    parser.add_argument('--A_path', type=str, default=f'data/{country}_{year}/ԭʼ����/����ʽ�_{year}.txt',
                        help='Path to the external assets file')
    parser.add_argument('--e_path', type=str, default=f'data/{country}_{year}/ԭʼ����/�ⲿ�ʲ�����������_{year}.txt',
                        help='Path to the external assets file')
    parser.add_argument('--name_path', type=str, default=f'data/{country}_{year}/ԭʼ����/��������_{year}.csv',
                        help='Path to the bank names file')
    parser.add_argument('--specific_threshold', type=float, default=0.00001, help='Specific threshold for the analysis')

    # ���渺ծ�����·��
    parser.add_argument('--data_path', type=str, default=f'data/{country}_{year}/ԭʼ����',
                        help='Path to the data directory')
    parser.add_argument('--data_path1', type=str, default=f'data/{country}_{year}/ɾ�������ڵ�/��ֵ',
                        help='Path to the data directory')

    parser.add_argument('--year', type=int, default=year,
                        help='Year for the analysis')
    # ��������
    args = parser.parse_args()
    # ��ֵͼӢ��չʾ
    font = FontProperties(fname='C:\\Windows\\Fonts\\Arial.ttf', size=12)
    # ��ֵͼ����չʾ
    # font = FontProperties(fname='C:\\Windows\\Fonts\\msyh.ttc', size=12)
        # ���args.data_path·���Ƿ���ڣ�����������򴴽�
    if not os.path.exists(args.data_path1):
        os.makedirs(args.data_path1)

    # ʾ����ͼ

    x0, proportion_L,proportion_A,total_A,banks_df,L,A = data_process(args.L_path,args.A_path,args.name_path)


    # ������ȫծ�����  full_savematrix����������  full_matrix����������
    full_savematrix,full_matrix = calculate_debt_ratio_fullmatrix(x0, proportion_L, proportion_A,L,A,eps = 1e-6)
    np.savetxt(os.path.join(args.data_path, f'{args.year+1}��ȫ��������������.txt'),full_matrix, fmt='%d')
    np.savetxt(os.path.join(args.data_path, f'{args.year+1}��ȫ��������������.txt'), full_savematrix, fmt='%d')

    print("��ȫ��ծ�����shape",full_matrix.shape)
    xA_after_isolation_removal1, threshold_qitamatrix1, density1, name1, e_data_after_isolation_removal1 = apply_threshold_and_calculate_edges(
        full_matrix, L, A, banks_df, args.e_path, args.specific_threshold)

    print('���ո�ծ����������',threshold_qitamatrix1.shape)
    np.savetxt(os.path.join(args.data_path1, f'��ծ��������������_{args.year}.txt'), xA_after_isolation_removal1, fmt='%d')
    np.savetxt(os.path.join(args.data_path1, f'��ծ��������������_{args.year}.txt'), threshold_qitamatrix1, fmt='%d')
    np.savetxt(os.path.join(args.data_path1, f'�ⲿ�ʲ�����������_{args.year}.txt'),
               e_data_after_isolation_removal1.reshape(1, -1), fmt='%d')
    e_data_with_zero = np.append(e_data_after_isolation_removal1, 0)
    print('�ⲿ�ʲ�����������',len( e_data_with_zero))
    np.savetxt(os.path.join(args.data_path1, f'�ⲿ�ʲ�����������_{args.year}.txt'), e_data_with_zero.reshape(1, -1),
               fmt='%d',
               )

    name_count = len(name1)

    length_file_path = f'{args.data_path1}/��������.txt'

    # Save the integer data to a text file
    with open(length_file_path, 'w') as file:
        file.write(str(name_count))
    name_file_path = os.path.join(args.data_path1, f'��������_{args.year}.txt')
    with open(name_file_path, 'w', encoding='utf-8') as file:
        for item in name1:
            file.write(str(item) + '\t')

    name_df = pd.DataFrame({'Bank Name': name1})
    name_file_path = os.path.join(args.data_path1, f'��������csv_{args.year}.csv')
    name_df.to_csv(name_file_path, index=False, encoding='utf-8')
    densities,max_cc_sizes, second_cc_sizes =  determine_threshold(full_matrix)
    plot_edge_density( densities, [0.00001,0.00005], font,args.data_path1)
    plot_connected_components( max_cc_sizes, second_cc_sizes, [0.00001,0.00005], font,args.data_path1)
