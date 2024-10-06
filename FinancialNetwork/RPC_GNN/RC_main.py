# -*- coding: utf-8 -*-
import datetime
import os
import random
import time

import numpy as np
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from model.RC_model import *
from torch.distributions import Categorical, Normal
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv

import csv


def load_bank_importance(file_path):
    df = pd.read_csv(file_path)
    importance_values = df['外部资产'].values
    min_value = importance_values.min()
    max_value = importance_values.max()
    normalized_importance_values = (importance_values - min_value) / (max_value - min_value)

    return normalized_importance_values, max_value

# 计算返回函数
def compute_returns(next_value, rewards, gamma):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns


# 注入设计
def adjust_standardized_value_nonlinear(current_value, rescue_ratio, max_value=1, nonlinearity='log'):
    rescue_ratio = torch.tensor(rescue_ratio)
    if nonlinearity == 'exp':
        new_value = current_value + (max_value - current_value) * (1 - torch.exp(-100 * rescue_ratio))
    elif nonlinearity == 'log':
        new_value = current_value + (max_value - current_value) * torch.log(1 + 100 * rescue_ratio)
    elif nonlinearity == 'sigmoid':
        new_value = current_value + (max_value - current_value) * torch.sigmoid(10 * rescue_ratio)
    elif nonlinearity == 'tanh':
        new_value = current_value + (max_value - current_value) * torch.tanh(5 * rescue_ratio)
    else:
        raise ValueError("Unsupported nonlinearity type")
    #
    # print('策略网络输出', rescue_ratio)
    # print('注入之后实际的差值', new_value - current_value)
    return new_value
#奖励函数设计
def compute_reward(total_risk_before, total_risk_after, node_risk_before, node_risk_after, rescue_ratio,
                   selected_node_assetvalue, normalized_importance_values,  system_weight=0.6, node_weight=0.4, scale_factor=10,
                   min_reward=1,
                   all_nodes_risk_probs=None):
    system_risk_reduction = total_risk_before - total_risk_after
    normalized_system_risk_reduction = system_risk_reduction / total_risk_before if total_risk_before != 0 else 0

    node_risk_reduction = node_risk_before - node_risk_after
    normalized_node_risk_reduction = node_risk_reduction / node_risk_before if node_risk_before != 0 else 0
    system_efficiency = normalized_system_risk_reduction / (rescue_ratio + 1e-8)

    node_efficiency = normalized_node_risk_reduction / (rescue_ratio + 1e-8)
    quartiles = np.percentile(normalized_importance_values, [25, 50, 75])
    if selected_node_assetvalue > quartiles[2]:
        if node_risk_before > 0.75:
            selection_weight = 10
        elif node_risk_before > 0.5:
            selection_weight = 5
        elif node_risk_before > 0.25:
            selection_weight = 2
        else:
            selection_weight = 1
    elif selected_node_assetvalue > quartiles[1]:
        if node_risk_before > 0.75:
            selection_weight = 5
        elif node_risk_before > 0.5:
            selection_weight = 2
        elif node_risk_before > 0.25:
            selection_weight = 1
        else:
            selection_weight = 0.5
    elif selected_node_assetvalue > quartiles[0]:
        if node_risk_before > 0.75:
            selection_weight = 5
        elif node_risk_before > 0.5:
            selection_weight = 2
        elif node_risk_before > 0.25:
            selection_weight = 0.5
        else:
            selection_weight = -1
    else:  # 低于25%，低重要性
        if node_risk_before > 0.75:
            selection_weight = 2
        elif node_risk_before > 0.5:
            selection_weight = 1
        elif node_risk_before > 0.25:
            selection_weight = 0.5
        else:
            selection_weight = -2

    reward = (system_weight * system_efficiency + node_weight * node_efficiency) * selection_weight

    if system_risk_reduction < 0 or node_risk_reduction < 0:
        reward = -1
    nodes_below_threshold = sum(all_nodes_risk_probs < 0.4).item() if all_nodes_risk_probs is not None else 0

    total_nodes = len(all_nodes_risk_probs) if all_nodes_risk_probs is not None else 1
    proportion_below_threshold = nodes_below_threshold / total_nodes

    return reward

def environment_step(graph_model, epoch_data_x, selected_node, rescue_ratio, adjust_func, edge_index, edge_weight,
                     normalized_importance_values):

    total_risk_before = graph_model.compute_total_risk(epoch_data_x.float(), edge_index, edge_weight)
    node_risk_before = graph_model.compute_node_risks(epoch_data_x .float(), edge_index, edge_weight)[1][selected_node]

    # 采取动作，注入救援资金
    updated_epoch_data_x = epoch_data_x.clone()

    updated_epoch_data_x[selected_node, 0] = adjust_func(epoch_data_x[selected_node, 0], rescue_ratio)

    total_risk_after = graph_model.compute_total_risk(updated_epoch_data_x.float(), edge_index, edge_weight)
    node_risk_after = graph_model.compute_node_risks(updated_epoch_data_x.float(), edge_index, edge_weight)[1][selected_node]

    all_nodes_risk_probs = graph_model.compute_node_risks(updated_epoch_data_x.float(), edge_index, edge_weight)[1]

    selected_node_assetvalue = normalized_importance_values[selected_node]

    reward = compute_reward(total_risk_before, total_risk_after, node_risk_before, node_risk_after, rescue_ratio, selected_node_assetvalue,
                            normalized_importance_values,  all_nodes_risk_probs=all_nodes_risk_probs)

    node_embeddings, risk_probs = graph_model(updated_epoch_data_x.float(), edge_index, edge_weight)
    risk_probs = F.softmax(risk_probs, dim=1)[:, 1].unsqueeze(1)  # 扩展维度
    next_state = torch.cat((node_embeddings, risk_probs), dim=1)

    return next_state, reward, updated_epoch_data_x

def train_a2c(a2c, graph_model, data, edge_index, edge_weight, adjust_func, epochs=500, normalized_importance_values=None):
    rewards_history = []
    total_risk_history = []
    test_rewards_history = []
    test_risk_history = []
    best_reward = float('-inf')
    best_risk_probs = None
    best_rescue_ratios = None
    initial_total_risk = graph_model.compute_total_risk(data.x.float(), edge_index, edge_weight)
    initial_node_risk_probs = graph_model.compute_node_risks(data.x.float(), edge_index, edge_weight)[1]
    test_count = 1
    import os
    import datetime
    import shutil


    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = f'America0.06_training_results_{current_time}'
    os.makedirs(dir_name, exist_ok=True)
    filename = os.path.join(dir_name, 'test_rewards_and_risks.csv')
    risk_filename = os.path.join(dir_name, 'initial_and_best_risk_probs.csv')
    new_dir_name = '0.05RL-result'
    os.makedirs(new_dir_name, exist_ok=True)
    shutil.move(dir_name, os.path.join(new_dir_name, dir_name))

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Average Test Reward', 'Average Test Risk'])

    with open(risk_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Node', 'Initial Risk Probability', 'Best Risk Probability', 'Rescue Ratio'])

    for epoch in range(epochs):

        epoch_start_time = time.time()
        total_reward = 0
        epoch_data_x = data.x.clone()
        remaining_budget = 1.0
        states, actions, rewards, values, log_probs = [], [], [], [], []
        count=0
        node_rescue_ratios = torch.zeros(data.num_nodes)

        node_embeddings, risk_probs = graph_model(epoch_data_x.float(), edge_index, edge_weight)
        risk_probs = F.softmax(risk_probs, dim=1)[:, 1].unsqueeze(1)  # 扩展维度
        state = torch.cat((node_embeddings, risk_probs), dim=1)

        for step in range(data.num_nodes):
            if remaining_budget <= 0:
                break
            count=count+1
            selected_node, rescue_ratio, log_prob = a2c.select_action(state, remaining_budget, edge_index, edge_weight)
            selected_node = selected_node.item() if isinstance(selected_node, torch.Tensor) else selected_node
            rescue_ratio = rescue_ratio.item() if isinstance(rescue_ratio, torch.Tensor) else rescue_ratio

            next_state, reward, update_epoch_data_x = environment_step(
                graph_model, epoch_data_x, selected_node, rescue_ratio, adjust_func, edge_index, edge_weight, normalized_importance_values)
            rewards.append(torch.tensor([reward]))  # 确保是张量
            values.append(a2c.value_net(state, edge_index, edge_weight).unsqueeze(0))
            states.append(state)

            log_probs.append(log_prob)

            node_rescue_ratios[selected_node] += rescue_ratio
            remaining_budget -= rescue_ratio
            epoch_data_x = update_epoch_data_x
            state = next_state
            total_reward += reward

        next_value = a2c.value_net(next_state, edge_index, edge_weight).unsqueeze(0)

        a2c.update(states, actions, rewards, values, log_probs, next_value, edge_index, edge_weight, epoch)

        rewards_history.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
            best_risk_probs = graph_model.compute_node_risks(epoch_data_x.float(), edge_index, edge_weight)[1]
            best_rescue_ratios = node_rescue_ratios.clone()

            with open(risk_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for i in range(len(initial_node_risk_probs)):
                    best_risk = best_risk_probs[i].item() if best_risk_probs is not None else 'N/A'
                    rescue_ratio = best_rescue_ratios[i].item() if best_rescue_ratios is not None else 'N/A'
                    csvwriter.writerow([i, initial_node_risk_probs[i].item(), best_risk, rescue_ratio])

        if (epoch + 1) % 50 == 0:
            average_test_reward, average_test_risk = test_policy(a2c, graph_model, data, edge_index, edge_weight, adjust_func,
                                              normalized_importance_values,dir_name,test_count)
            test_rewards_history.append(average_test_reward)
            test_risk_history.append(average_test_risk)

            # 写入 CSV 文件
            with open(filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([(epoch + 1), average_test_reward, average_test_risk])

            print(f"Test after Epoch {epoch + 1}: Average Test Reward: {average_test_reward}")
            test_count = test_count + 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(
            f"Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward}, Step: {count}, Duration: {epoch_duration:.2f} seconds")


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, label='Total Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(os.path.join(dir_name, 'total_reward_history.png'))  # 保存图像

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(test_rewards_history, label='Average Test Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Average Test Reward')
    plt.legend()
    plt.savefig(os.path.join(dir_name, 'test_reward_history.png'))  # 保存图像

    plt.subplot(1, 2, 2)
    plt.plot(test_risk_history, label='Average Test Risk')
    plt.xlabel('Epoch')
    plt.ylabel('Average Test Risk')
    plt.legend()
    plt.savefig(os.path.join(dir_name, 'test_risk_history.png'))  # 保存图像


    plt.show()
    for i, ratio in enumerate(node_rescue_ratios):
        print(f"Node {i}: Rescue Ratio: {ratio:.4f}")

def test_policy(a2c, graph_model, data, edge_index, edge_weight, adjust_func, normalized_importance_values, dir_name,test_count,num_episodes=10):
    total_test_reward = 0
    total_test_risk = 0
    risk_changes = []
    node_rescue_sums = torch.zeros(data.num_nodes)
    node_risk_sums = torch.zeros(data.num_nodes)

    for _ in range(num_episodes):
        epoch_data_x = data.x.clone()
        remaining_budget = 1.0
        node_rescue_ratios = torch.zeros(data.num_nodes)

        node_embeddings, risk_probs = graph_model(epoch_data_x.float(), edge_index, edge_weight)
        risk_probs = F.softmax(risk_probs, dim=1)[:, 1].unsqueeze(1)
        state = torch.cat((node_embeddings, risk_probs), dim=1)

        test_reward = 0

        for step in range(data.num_nodes):
            if remaining_budget <= 0:
                break
            selected_node, rescue_ratio, _ = a2c.test_select_action(state, remaining_budget, edge_index, edge_weight)
            selected_node = selected_node.item() if isinstance(selected_node, torch.Tensor) else selected_node
            rescue_ratio = rescue_ratio.item() if isinstance(rescue_ratio, torch.Tensor) else rescue_ratio

            next_state, reward, update_epoch_data_x = environment_step(
                graph_model, epoch_data_x, selected_node, rescue_ratio, adjust_func, edge_index, edge_weight,normalized_importance_values)
            test_reward += reward

            node_rescue_ratios[selected_node] += rescue_ratio
            remaining_budget -= rescue_ratio
            epoch_data_x = update_epoch_data_x
            state = next_state


        test_risk = graph_model.compute_total_risk(epoch_data_x.float(), edge_index, edge_weight)
        total_test_risk+= test_risk
        total_test_reward += test_reward
        node_rescue_sums += node_rescue_ratios
        node_risk_sums += graph_model.compute_node_risks(epoch_data_x.float(), edge_index, edge_weight)[1]

    average_test_reward = total_test_reward / num_episodes
    average_test_risk = total_test_risk/ num_episodes
    average_node_rescues = node_rescue_sums / num_episodes
    average_node_risks = node_risk_sums / num_episodes

    # Log the average node rescues and risks
    average_node_rescues = average_node_rescues.tolist()
    average_node_risks = average_node_risks.tolist()

    # 创建目录

    file_path = os.path.join(dir_name, f'node_averages_{test_count}.csv')

    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Node', 'Average Rescue Ratio', 'Average Risk Probability'])
        for i in range(len(average_node_rescues)):
            csvwriter.writerow([i, average_node_rescues[i], average_node_risks[i]])

    return average_test_reward,average_test_risk


if __name__ == '__main__':
    mode = "train"

    year = "2022"
    country = "America"
    type = 2
    ratio = "train0.6_val0.15_test0.25"
    dataset = torch.load(f'foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')
    country_name = "America_2022"
    normalized_importance_values, max_importance_value = load_bank_importance(f'../Foreigh/data/{country_name}/{country}_bank_assets.csv')

    if mode == "train":
        print("开始训练")
        epochs = 7500
        data = dataset[type]
        graph_model = GraphGCN(in_channels=data.num_features, data=data)
        model_path = f'save_models/{country}/{year}/{ratio}/best_model.pth'
        graph_model.load_state_dict(torch.load(model_path))
        graph_model.eval()
        print(data.x.size())
        policy_net = PolicyNetwork(in_channels=17, hidden_dim=16, num_nodes=data.num_nodes)
        value_net = ValueNetwork(in_channels=17, hidden_dim=16)
        a2c = A2C(policy_net, value_net)
        train_a2c(a2c, graph_model, data, data.edge_index, data.edge_weight, adjust_standardized_value_nonlinear,
              epochs=epochs,   normalized_importance_values= normalized_importance_values)
