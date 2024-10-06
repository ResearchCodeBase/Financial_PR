#coding=utf-8

import argparse
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from pulp import *
import matplotlib.pyplot as plt

import pandas as pd


def load_data(L_path, e_path, name_path):
    l = np.loadtxt(L_path)
    e = np.loadtxt(e_path)
    with open(name_path, encoding='utf-8') as file:
        names = np.array(file.read().split('\t'))

    return l, e, names


def set_debt_matrix(l, n):
    L = np.sum(l, axis=1)
    π = np.zeros((n, n))
    for i in range(n):
      for j in range(n):
        if L[i] > 0:
          if l[i, j] > 0:
            π[i, j] = l[i, j] / L[i]
          else:
            π[i, j] = 0
        else:
          π[i, j] = 0
    return π

def update_solvent_banks(P, S, l):
  # 提取数据
  P_new = P.copy()
  P_new[S, :] = l[S, :]
  return P_new


# 已经更新e<0的情况
def solve_insolvent_banks(l, π, S, I, e, P,alpha,beta):
  n = len(I)
  P_new = P.copy()
  e_new = e.copy()

  A = np.zeros((I.size, I.size))
  b = np.zeros(I.size)

  for index1, value1 in enumerate(I):
      for index2, value2 in enumerate(I):
          if (index1 == index2):
              A[index1][index2] = 1 - beta * π[value1][value1]
          else:
              A[index1][index2] = -beta * π[value2][value1]
  for index, value in enumerate(I):
      # 处理当外部资产小于0
      if(e[value] < 0):
        b[index] =  e[value]+beta * l[S, value].sum()
      else:
        b[index] = e[value] * alpha + beta * l[S, value].sum()


  x = np.linalg.solve(A, b)

  for index, value in enumerate(I):
      P_new[value, :] = x[index] * π[value, :]
      if(e[value]<0):
        e_new[value] = e[value]
      else:
        e_new[value] = e[value] * alpha
  return P_new, e_new

def simulate_sde(alpha, beta, mu_alpha, sigma_alpha, mu_beta, sigma_beta, dt):
    dW_alpha = np.random.normal(0, np.sqrt(dt))
    dW_beta = np.random.normal(0, np.sqrt(dt))

    # 确保下降趋势
    mu_alpha = -np.abs(mu_alpha)
    mu_beta = -np.abs(mu_beta)

    alpha += mu_alpha * alpha * dt + sigma_alpha * alpha * dW_alpha
    beta += mu_beta * beta * dt + sigma_beta * beta * dW_beta

    return alpha, beta
# 算法主体
def compute_GVA(l, π, e,n,path,alpha,beta,names,mu_alpha, sigma_alpha, mu_beta, sigma_beta, dt):


  v = np.zeros((n,1))
  P = l.copy()
  epoch = 0
  liquidities = []
  updated = True
  L = np.sum(l, axis=1)
  print( 'alpha ={},beta={}',alpha,beta)

  first_epoch_bank_status = [0] * n
  last_epoch_bank_status = [0] * n
  while updated == True:
    # 银行间资产按列求和
    v = np.sum(P, axis=0)

    v_result = v + e - L
    print('*************************************')
    print('epoch:', epoch+1)


    I = np.where(v_result < 0)[0]
    S = np.where(v_result >= 0)[0]

    P = update_solvent_banks(P, S, l)
    P, e = solve_insolvent_banks(l, π, S, I, e, P,alpha,beta)

    if (epoch != 0):
      if (I.size == prev_I.size):
        if ((prev_I == I).all() and (abs(P.sum() - prev_P.sum()) < 0.05)):
          print('P.sum() - prev_P.sum()',abs(P.sum() - prev_P.sum()))
          updated = False


    prev_I = I
    prev_P = P

    if epoch == 0:
      # 将破产银行的位置标记为
      first_epoch_bankrupt_banks = I.copy()
      for bank in first_epoch_bankrupt_banks:
        first_epoch_bank_status[bank] = 1

    L_result = np.sum(P, axis=1)
    liquidities.append(P.sum())
    # Append the epoch number and transposed bank list values to the CSV file
    epoch += 1
    alpha, beta = simulate_sde(alpha, beta, mu_alpha, sigma_alpha, mu_beta, sigma_beta, dt)
  last_epoch_bankrupt_banks = I.copy()
  for bank in last_epoch_bankrupt_banks:
    last_epoch_bank_status[bank] = 1

  l_file_directory = os.path.dirname(path)
  # Generate a timestamp to append to the filename
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  output_filename = os.path.join(l_file_directory, f'违约标签_{timestamp}.csv')
  with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['First Epoch Bankrupt Banks', 'y'])
    for i in range(n-1):
      writer.writerow([first_epoch_bank_status[i], last_epoch_bank_status[i]])
  np.savetxt('P_values.txt', P)



  return P,liquidities,output_filename

def plot_liquidity_over_iterations(liquidities, alpha, beta,path):

    plt.plot(liquidities)
    plt.title("Total Liquidity over Iterations")
    plt.text(1, 1, f"alpha = {alpha}, beta = {beta}", transform=plt.gca().transAxes)
    plt.xlabel('Iterations')
    plt.ylabel('Total Liquidity')
    file_directory = os.path.dirname(path)
    output_filename = os.path.join(file_directory,  'liquidity_plot.png')
    plt.savefig(output_filename)
    plt.show()

def label_main(L_path, e_path, name_path,alpha, beta):
    l, e, names = load_data(L_path,e_path, name_path)
    n = l.shape[0]  # 银行总数
    π = set_debt_matrix(l, n)
    output_dir = os.path.dirname(L_path)
    os.makedirs(output_dir, exist_ok=True)

    sigma_alpha = 0.1
    sigma_beta = 0.1
    mu_alpha = -0.5
    mu_beta = -0.5

    dt = 1 / 7
    P, liquidities, output_filename = compute_GVA(l, π, e,n,L_path,alpha,beta,names,mu_alpha, sigma_alpha, mu_beta, sigma_beta, dt)


    plot_filename = os.path.join(output_dir, f'liquidity_plot_alpha_{alpha}_beta_{beta}.png')
    plot_liquidity_over_iterations(liquidities, alpha, beta,L_path)

    return output_filename


# 主函数
if __name__ == '__main__':
    # 这里是需要其他银行的
    year='2022'
    parser = argparse.ArgumentParser(description='Process Banking Network Features')
    parser.add_argument('--L_path', type=str, default=f'data/{year}/删除孤立节点/负债矩阵有其他银行_{year}.txt', help='Path to the liability matrix file')
    parser.add_argument('--e_path', type=str, default=f'data/{year}/删除孤立节点/外部资产有其他银行_{year}.txt', help='Path to the external assets file')
    parser.add_argument('--name_path', type=str, default=f'data/{year}/删除孤立节点/银行名称_{year}.txt', help='Path to the bank names file')
    parser.add_argument('--alpha', type=float, default='0.5',
                        help='Path to the bank names file')
    parser.add_argument('--beta', type=float, default='0.8')

    args = parser.parse_args()

    labelcsv_path = label_main(args.L_path, args.e_path, args.name_path,args.alpha, args.beta)