#coding=utf-8


import argparse
import os
from datetime import datetime

from  Bankruptcy_Label_Generation1 import label_main
from data_process2 import process_features

from GraphDataset3 import FeatureBankingNetworkDataset

# 主函数
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='从银行网络数据分析和处理特征。')

    year = '2023'

    country = 'United Kingdom'
    parser = argparse.ArgumentParser(description='Process Banking Network Features')

    type='阈值'
    path=f'data/{country}_{year}/删除孤立节点/{type}'
    parser.add_argument('--L1_path', type=str, default=f'{path}/负债矩阵有其他银行_{year}.txt',help='负债矩阵文件路径（包括其他银行）')
    parser.add_argument('--e1_path', type=str, default=f'{path}/外部资产有其他银行_{year}.txt', help='外部资产文件路径')
    parser.add_argument('--name_path', type=str, default=f'{path}/银行名称_{year}.txt', help='银行名称文件路径')
    parser.add_argument('--alpha', type=float, default=0.5, help='模型权重参数alpha')
    parser.add_argument('--beta', type=float, default=0.8, help='模型权重参数beta')

    parser.add_argument('--L_path', type=str, default=f'{path}/负债矩阵无其他银行_{year}.txt',help='负债矩阵文件路径（不包括其他银行）')
    parser.add_argument('--e_path', type=str, default=f'{path}/外部资产无其他银行_{year}.txt',help='外部资产文件路径（不包括其他银行）')

    parser.add_argument('--train_ratio', type=float, default=0.32, help='用于训练的数据集比例')
    parser.add_argument('--val_ratio', type=float, default=0.08, help='用于验证的数据集比例')
    parser.add_argument('--test_ratio', type=float, default=0.6, help='用于测试的数据集比例')


    parser.add_argument('--root', type=str, default=f'../GCN/foreign_dataset/{country}/{year}', help='保存处理后数据集的根目录。')


    args = parser.parse_args()

    labelcsv_path = label_main(args.L1_path, args.e1_path, args.name_path, args.alpha, args.beta)

    raw_features_path, normalized_features_path,yeojohnson_transformed_path, cdf_scaled_features_path = process_features(args.L_path,
                                                                                                args.e_path,
                                                                                                labelcsv_path,country)


    if not os.path.exists(args.root):

        os.makedirs(args.root, exist_ok=True)

    train_val_dir = f"train{args.train_ratio}_val{args.val_ratio}_test{args.test_ratio}"

    args.root = os.path.join(args.root, train_val_dir)


    os.makedirs(args.root, exist_ok=True)

    dataset = FeatureBankingNetworkDataset(
        root=args.root,
        L_path=args.L_path,
        e_path=args.e_path,
        rawfeature_path= raw_features_path,
        normalized_features_path =  normalized_features_path,
        cdf_scaled_features_path=cdf_scaled_features_path,
        train_size=args.train_ratio,
        val_size=args.val_ratio,
        test_size=args.test_ratio
    )

