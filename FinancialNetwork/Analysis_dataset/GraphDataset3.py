# 样例
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from torch_geometric.data import InMemoryDataset



def load_dataset(l_path, e_path, default_path):
    l = np.loadtxt(l_path)
    e = np.loadtxt(e_path)
    default_data = pd.read_csv(default_path)
    return l, e, default_data


def calculate_default_impact(total_liabilities,total_assets, solvency_status, impact_factor=0.5):
    solvency_status = solvency_status.numpy()
    print(solvency_status.shape)
    print(solvency_status)
    default_impact = total_liabilities * (1 + impact_factor * solvency_status) / (1 + total_assets)
    return default_impact

def weighted_adjacency_matrix_to_edge_index(debt_matrix, external_assets, solvency_status):

    total_liabilities = np.sum(debt_matrix, axis=1)
    total_assets = np.sum(debt_matrix, axis=0) + external_assets


    default_impact = calculate_default_impact(total_liabilities, total_assets, solvency_status)

    edge_default_impact = np.zeros_like(debt_matrix, dtype=float)
    for i in range(debt_matrix.shape[0]):
        for j in range(debt_matrix.shape[1]):
            debt_ratio = debt_matrix[i, j] / total_liabilities[i] if total_liabilities[i] > 0 else 0
            edge_default_impact[i, j] = (default_impact[i] if solvency_status[i] == 1 else 1) * debt_ratio


    num_nodes = debt_matrix.shape[0]
    edge_list = []
    edge_weight = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if debt_matrix[i][j] != 0:
                edge_list.append((i, j))
                edge_weight.append(edge_default_impact[i][j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float).view(-1, 1)

    print('Edge index shape:', edge_index.shape)
    print('Edge weight shape:', edge_weight.shape)
    print('Edge index:', edge_index)
    print('Edge weight:', edge_weight)

    return edge_index, edge_weight

# 用于处理数据，提取特征数据x与标签数据y
def data_process(path):
    df = pd.read_csv(path)
    print(df)

    columns1 = [
        "External_Assets",
        "Lent_Funds",
        "Borrowed_Funds",
        "LMI",
        "Basic_Default",
        "First_Order_Neighbor_Default_Rate",
        "Loan_Count_Number_of_Creditors",
        "In_Degree_Centrality",
        "Out_Degree_Centrality",
        "Closeness_Centrality",
        "Betweenness_Centrality",
        "PageRank",
        "Average_Indegree_of_In_Neighbors"
    ]

    print(df[columns1])

    x = torch.tensor(df[columns1].values)
    print('x shape', x.shape)
    colunms2 = ['y']

    y = torch.tensor(df[colunms2].values)
    print('y shape', y.shape)

    return x,y
def train_test_split_with_masks(train_size,test_size,val_size,y):

    index = np.arange(len(y))

    indices_train_val, indices_test, y_train_val, y_test = train_test_split(index,y,  stratify=y, test_size=test_size,
                                                                     random_state=48, shuffle=True)

    indices_train, indices_val, y_train, y_val = train_test_split(indices_train_val, y_train_val, stratify=y_train_val,
                                                                  train_size=train_size / (train_size + val_size),
                                                                   random_state=48, shuffle=True)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[indices_train] = True
    val_mask[indices_val] = True
    test_mask[indices_test] = True




    return train_mask,val_mask,test_mask


def feature_process(l,e,feature_path,train_size,test_size,val_size):

    x, y = data_process(feature_path)
    y = y.squeeze()
    print('***y', y.shape)
    edge_index, edge_weight = weighted_adjacency_matrix_to_edge_index(l, e, y)
    train_mask, val_mask, test_mask = train_test_split_with_masks(train_size,test_size,val_size, y)
    edge_weight = edge_weight.squeeze()
    print('*******edgeweitg', edge_weight.shape)
    return x, y, edge_index, edge_weight, train_mask, val_mask, test_mask

class FeatureBankingNetworkDataset(InMemoryDataset):

    def __init__(self, root, L_path, e_path, rawfeature_path,normalized_features_path, cdf_scaled_features_path, train_size, val_size, test_size,
                 transform=None, pre_transform=None):
        self.L_path = L_path
        self.e_path = e_path
        self.rawfeature_path = rawfeature_path
        self.normalized_features_path= normalized_features_path
        self.cdf_scaled_features_path =  cdf_scaled_features_path
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        super(FeatureBankingNetworkDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):

        return []

    @property
    def processed_file_names(self):
        return ['BankingNetwork.dataset']

    def download(self):
        pass


    def process(self):
        data_list = []
        l = np.loadtxt(self.L_path)
        e = np.loadtxt(self.e_path)
        x, y, edge_index, edge_weight, train_mask, val_mask, test_mask = feature_process(l,e,self.rawfeature_path,self.train_size,self.test_size,self.val_size)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask)
        data_list.append(data)
        x1, y1, edge_index1, edge_weight1, train_mask1, val_mask1, test_mask1 = feature_process(l, e, self.normalized_features_path,
                                                                                         self.train_size,
                                                                                         self.test_size, self.val_size)
        data1 = Data(x=x1, edge_index=edge_index1, edge_weight=edge_weight1, y=y1, train_mask=train_mask1, val_mask=val_mask1,
                    test_mask=test_mask1)
        data_list.append(data1)
        x2, y2, edge_index2, edge_weight2, train_mask2, val_mask2, test_mask2 = feature_process(l, e, self.cdf_scaled_features_path,
                                                                                         self.train_size,
                                                                                         self.test_size, self.val_size)
        data2 = Data(x=x2, edge_index=edge_index2, edge_weight=edge_weight2, y=y, train_mask=train_mask2, val_mask=val_mask2,
                    test_mask=test_mask2)
        data_list.append(data2)
        torch.save(data_list, self.processed_paths[0])









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Banking Network Features')
    parser.add_argument('--L_path', type=str, default='../data/2022-182/2022全-负债矩阵(无其他银行).txt',
                        help='Path to the liability matrix file')
    parser.add_argument('--e_path', type=str, default='../data/2022-182/2022全-外部资产(无其他银行).txt',
                        help='Path to the external assets file')
    parser.add_argument('--rawfeature_path', type=str, default='data_feaature/2022全-finalresult .csv',
                        help='Path to the default data file')
    parser.add_argument('--processfeature_path', type=str, default='2022_1211_200331/cdf_scaled_features.csv',
                        help='Path to the default data file')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--root', type=str, default='../GCN/1211182/', help='Root directory for saving the dataset')

    args = parser.parse_args()

    dataset = FeatureBankingNetworkDataset(
        root=args.root,
        L_path=args.L_path,
        e_path=args.e_path,
        rawfeature_path=args.rawfeature_path,
        processfeature_path=args.processfeature_path,
        train_size=args.train_ratio,
        val_size=args.val_ratio,
        test_size=args.test_ratio
    )

