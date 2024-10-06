# coding=gbk
import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson
from matplotlib.font_manager import FontProperties
from scipy.stats import yeojohnson
import numpy as np

def yeojohnson_transform(features):

    transformed_features = features.copy()

    for i in range(features.shape[1]):
        try:
           transformed_features[:, i], _ = yeojohnson(features[:, i])
        except Exception as e:
            print(f"An error occurred while transforming feature column {i}: {e}. Original data is kept.")
    return transformed_features

def save_yeojohnsonfeatures_to_csv(saved_features_file,features, directory_name,filename_prefix="features",column_names=None):

    original_df = pd.read_csv(saved_features_file)


    basic_default_col = original_df[["Basic_Default"]]
    y = original_df[["y"]]

    if column_names is None:
        transformed_df = pd.DataFrame(features)
    else:
        transformed_df = pd.DataFrame(features, columns=column_names)

    combined_df = pd.concat([transformed_df, basic_default_col, y], axis=1)


    filename = f"{directory_name}/{filename_prefix}.csv"
    combined_df.to_csv(filename, index=False)
    return filename

def cdf_scale(feature):

    sorted_feature = np.sort(feature)
    cdf = np.arange(1, len(sorted_feature) + 1) / len(sorted_feature)
    cdf_mapping = dict(zip(sorted_feature, cdf))
    scaled_feature = np.array([cdf_mapping[val] for val in feature])
    return scaled_feature

def kde_difference(feature, labels,feature_name,save_path):

    feature_class_0 = feature[labels == 0]
    feature_class_1 = feature[labels == 1]
    # KDE
    kde_class_0 = gaussian_kde(feature_class_0)
    kde_class_1 = gaussian_kde(feature_class_1)

    values = np.linspace(min(feature), max(feature), 100)

    kde_diff = np.sum(np.abs(kde_class_0(values) - kde_class_1(values)))
    plt.figure(figsize=(12, 6))
    # KDE

    plt.plot(values, kde_class_0(values), color='blue', label='Class 0')
    plt.plot(values, kde_class_1(values), color='orange', label='Class 1')
    plt.title(f'{feature_name} - KDE Difference: {kde_diff:.2f}')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    kde_dir = os.path.join(save_path, 'kde')
    os.makedirs(kde_dir, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'kde', f'{feature_name}.png'))
    plt.tight_layout()
    plt.show()






    return kde_diff

def plot_feature_transformation(original_feature, transformed_feature, feature_name, save_path):

    plt.figure(figsize=(8, 4))

    plt.hist(transformed_feature, bins=30, alpha=0.5, color='green', label='Transformed')
    plt.title(f'{feature_name} - Original vs Transformed')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{feature_name}.png'))
    plt.close()



