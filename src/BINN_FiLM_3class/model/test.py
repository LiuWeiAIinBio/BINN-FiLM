import torch
import pandas as pd
import torch.nn as nn
import os
from typing import List, Tuple
from torch.utils.data import DataLoader
from util import load_reactome_db, train_validation_split
from pathway_network import dataframes_to_pathway_network
from binn import BINN
from dataset import BINNDataset
import config
import torchsummary


# 逐层展示邻接矩阵，并计算连接数，连接比率
n_layers = 4
for i in range(n_layers+1):
    print(f"第{i+1}层和第{i+2}层的邻接矩阵: ", "\n", config.pathway_network.get_connectivity_matrices(n_layers)[i], "\n",
          f"第{i+1}层和第{i+2}层邻接矩阵的连接数量: ", config.pathway_network.get_connectivity_matrices(n_layers)[i].sum().sum(), "\n",
          f"第{i+1}层和第{i+2}层的连接比率: ", config.pathway_network.get_connectivity_matrices(n_layers)[i].sum().sum() /
          (config.pathway_network.get_connectivity_matrices(n_layers)[i].shape[0] *
           config.pathway_network.get_connectivity_matrices(n_layers)[i].shape[1]), "\n")


# 打印 train_data 和 train_loader
train_data = BINNDataset(config.X_train, config.y_train)
print(train_data.data)


# 打印模型
binn_model = BINN(data_matrix=config.data_matrix, mapping=config.mapping, pathways=config.pathways)
print(binn_model)


# 假设输入 size 为 (233,)，查看模型输入输出情况
summary = torchsummary.summary(model=binn_model, input_size=(233,))
print(summary)


# 遍历模型的每一线性层（输出层除外），计算有效权重数量
print("模型的线性层包括：", [layer_name for layer_name, layer in binn_model.layers.named_children() if isinstance(layer, nn.Linear)])

for layer_name, layer in binn_model.layers.named_children():
    if isinstance(layer, nn.Linear) and hasattr(layer, 'weight_mask'):  # 只处理有剪枝的线性层（Linear层），过滤激活函数、BN层等
        effective_weight = layer.weight * layer.weight_mask
        print(f"{layer_name}层的有效权重数量：", (effective_weight != 0).sum().item())
