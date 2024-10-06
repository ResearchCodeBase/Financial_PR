# coding=gbk
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
# coding=gbk

def calculate_network_features(L, e, default):


    row_sums = np.sum(L, axis=1)
    column_sums = np.sum(L, axis=0)
    LMI = (column_sums - row_sums) / (e + column_sums)

    neighbor_ratios = []
    for node in range(L.shape[0]):
        in_neighbors = np.where(L[:, node] > 0)[0]
        out_neighbors = np.where(L[node, :] > 0)[0]
        neighbors = np.union1d(in_neighbors, out_neighbors)
        neighbors = np.setdiff1d(neighbors, node)
        neighbor_count = len(neighbors)
        neighbor_default_count = np.sum(default[neighbors] == 1) if neighbor_count > 0 else 0
        neighbor_ratio = neighbor_default_count / neighbor_count if neighbor_count > 0 else 0
        neighbor_ratios.append(neighbor_ratio)

    L[L > 0] = 1
    print(L)
    G = nx.DiGraph(L)

    in_degrees = np.sum(L, axis=0)
    print('**********',in_degrees)

    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    closeness_centrality = nx.closeness_centrality(G.reverse())
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    pagerank_centrality = nx.pagerank(G)
    average_indegree = []
    for node in range(L.shape[0]):
        in_neighbors = np.where(L[:, node] == 1)[0]
        sum_indegree = np.sum(L[:, in_neighbors], axis=0)
        if len(sum_indegree) > 0:
            average_indegree.append(np.mean(sum_indegree))
        else:
            average_indegree.append(0)


    features = {
        "External_Assets": pd.Series(e),
        "Lent_Funds": pd.Series(column_sums),
        "Borrowed_Funds": pd.Series(row_sums),
        "LMI": pd.Series(LMI),
        "First_Order_Neighbor_Default_Rate": pd.Series(neighbor_ratios),
        "Basic_Default": pd.Series(default),
        'Loan_Count_Number_of_Creditors': pd.Series(in_degrees),
        'In_Degree_Centrality': pd.Series(in_degree_centrality),
        'Out_Degree_Centrality': pd.Series(out_degree_centrality),
        'Closeness_Centrality': pd.Series(closeness_centrality),
        'Betweenness_Centrality': pd.Series(betweenness_centrality),
        'PageRank': pd.Series(pagerank_centrality),
        'Average_Indegree_of_In_Neighbors': pd.Series(average_indegree)
    }
    return features


def save_features_to_csv(features, directory_name):


    df = pd.DataFrame(features)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{directory_name}/原始特征.csv"
    df.to_csv(filename, index=False)


    return filename

def save_features_to_csv1(features, directory_name):

    df = pd.DataFrame(features)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{directory_name}/max_min.csv"
    df.to_csv(filename, index=False)
    return filename