# coding=gbk

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from Network_Feature_Analysis import *
from Feature_Transformation_and_KDE_Analysis import yeojohnson_transform, plot_feature_transformation, cdf_scale, \
    kde_difference, save_yeojohnsonfeatures_to_csv


def signed_min_max_normalize_columns(data_frame, columns_to_normalize):


    normalized_data_frame = data_frame.copy()

    for column in columns_to_normalize:
        max_abs_val = normalized_data_frame[column].abs().max()
        if max_abs_val == 0:
            max_abs_val = 1


        normalized_data_frame[column] = normalized_data_frame[column] / max_abs_val

    return normalized_data_frame


def create_timestamped_directory(prefix):
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    new_dir_name = f"{prefix}_{timestamp}"
    os.makedirs(new_dir_name, exist_ok=True)
    return new_dir_name


def save_combined_features(original_features_file, transformed_features, directory_name, filename, selected_columns):
    original_df = pd.read_csv(original_features_file)
    unselected_columns = [col for col in original_df.columns if col not in selected_columns]
    transformed_df = pd.DataFrame(transformed_features, columns=selected_columns)
    combined_df = pd.concat([original_df[unselected_columns], transformed_df], axis=1)
    combined_file_path = os.path.join(directory_name, filename)
    combined_df.to_csv(combined_file_path, index=False)
    return combined_file_path


def process_features(L_path, e_path, default_path,country):
    L = np.loadtxt(L_path)
    e = np.loadtxt(e_path)

    default_data = pd.read_csv(default_path)
    default = default_data.iloc[:, 0].to_numpy().squeeze()
    labels = default_data.iloc[:, 1].to_numpy().squeeze()

    data_feature_directory = "data_feature"
    data_feature_directory = os.path.join(data_feature_directory, country)
    if not os.path.exists(data_feature_directory):
        os.makedirs(data_feature_directory)

    file_name = os.path.basename(L_path)[-8:-4]

    directory_name = os.path.join(data_feature_directory, file_name)
    if not os.path.exists( directory_name ):
        os.makedirs(directory_name)


    # Create the directory
    os.makedirs(directory_name, exist_ok=True)
    # directory_name = os.path.join(data_feature_directory, timestamped_directory_name)

    features = calculate_network_features(L, e, default)
    # 将labels作为新列追加到features字典中
    features['y'] = labels
    # 保存未处理的特征，文件名带原始特征
    raw_features_path = save_features_to_csv(features, directory_name)

    selected_columns = [
        "External_Assets", "Lent_Funds", "Borrowed_Funds", "LMI",
        "First_Order_Neighbor_Default_Rate", "Loan_Count_Number_of_Creditors",
        "In_Degree_Centrality", "Out_Degree_Centrality", "Closeness_Centrality",
        "Betweenness_Centrality", "PageRank", "Average_Indegree_of_In_Neighbors"
    ]

    columns_to_normalize = [
        "External_Assets", "Lent_Funds", "Borrowed_Funds", "LMI", "Average_Indegree_of_In_Neighbors"
    ]
    normalized_features = signed_min_max_normalize_columns(features, columns_to_normalize)
    normalized_features_path = save_features_to_csv1(normalized_features, directory_name)

    df = pd.read_csv(raw_features_path, usecols=selected_columns)
    features = df.to_numpy()

    yeojohnson_transformed_features = yeojohnson_transform(features)

    yeojohnson_transformed_path = save_yeojohnsonfeatures_to_csv(raw_features_path, yeojohnson_transformed_features,
                                                                 directory_name, "yeojohnson_features",
                                                                 selected_columns)

    img_directory = os.path.join(directory_name, "img")
    os.makedirs(img_directory, exist_ok=True)
    cdf_scaled_features = np.apply_along_axis(cdf_scale, 0, yeojohnson_transformed_features)
    cdf_scaled_features_path = save_yeojohnsonfeatures_to_csv(raw_features_path, cdf_scaled_features, directory_name,
                                                              "cdf_scaled_features", selected_columns)

    kde_differences = [kde_difference(cdf_scaled_features[:, i], labels, selected_columns[i], img_directory) for i in
                       range(cdf_scaled_features.shape[1])]
    kde_diff_df = pd.DataFrame(kde_differences, columns=["KDE Difference"])
    kde_diff_df.to_csv(f"{directory_name}/kde_differences.csv", index=False)

    return raw_features_path, normalized_features_path, yeojohnson_transformed_path, cdf_scaled_features_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process Banking Network Features')
    parser.add_argument('--L_path', type=str, default = '../data/2022-182/2022全-负债矩阵(无其他银行).txt', help='Path to the liability matrix file' )
    parser.add_argument('--e_path', type=str, default='../data/2022-182/2022全-外部资产(无其他银行).txt', help='Path to the external assets file')
    parser.add_argument('--default_path', type=str, default='../data/2022-182/违约标签.csv',help='Path to the default data file')


    args = parser.parse_args()

    saved_features_path,normalized_features_path,yeojohnson_transformed_path,cdf_scaled_features_path = process_features(args.L_path, args.e_path, args.default_path)

