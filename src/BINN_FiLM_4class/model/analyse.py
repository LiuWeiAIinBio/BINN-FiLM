import os
import torch
from binn import BINN
from binn_original import BINN_original
from trainer import BINNTrainer
import config
from util import set_seed
from analysis.explainer import BINNExplainer
import pandas as pd
from torch.utils.data import DataLoader
from dataset import BINNDataset
from plot.network import binn_to_edges_dataframe


set_seed(6)
train = True
evaluate = False

X_train = config.X_train[0]
T_train = config.T_train[0]
y_train = config.y_train[0]
X_val = config.X_val[0]
T_val = config.T_val[0]
y_val = config.y_val[0]

# initialize BINN and BINNTrainer
binn_model = BINN(data_matrix=config.data_matrix,
                  mapping=config.mapping,
                  pathways=config.pathways,
                  activation=config.activation,
                  n_layers=config.n_layers,
                  n_outputs=config.n_outputs,
                  dropout=config.dropout)

# binn_original_model = BINN_original(data_matrix=config.data_matrix,
#                                     mapping=config.mapping,
#                                     pathways=config.pathways,
#                                     activation=config.activation,
#                                     n_layers=config.n_layers,
#                                     n_outputs=config.n_outputs,
#                                     dropout=config.dropout)


checkpoint_path = "D:\\Desktop\\桑基图_012\\data\\example_data\\PXD013629_1_checkpoint\\checkpoint\\fold0_checkpoint_epoch_194.pt"  # not fixed
# checkpoint_path = "D:\\Desktop\\桑基图_012\\data\\example_data\\PDAC150_1_checkpoint\checkpoint\\fold0_checkpoint_epoch_284.pt"
checkpoint = torch.load(checkpoint_path, weights_only=True)
state_dict = checkpoint["model_state_dict"]

# 过滤掉包含"modulation_heads"的参数（FiLM层）
# filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("modulation_heads")}  # 排除FiLM层的权重

# 加载过滤后的权重（此时可以用strict=False，因为已主动排除不匹配的键）
# binn_original_model.load_state_dict(filtered_state_dict, strict=True)
# binn_original_model.eval()
# print(binn_original_model)
binn_model.load_state_dict(state_dict, strict=True)
binn_model.eval()

train_data = BINNDataset(X_train, T_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
val_data = BINNDataset(X_val, T_val, y_val)
val_loader = DataLoader(dataset=val_data, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)


explainer = BINNExplainer(model=binn_model)

single_explanations = explainer.explain_single(back_dataloaders=list(train_loader),
                                               test_dataloaders=list(val_loader), normalization_method="subgraph")
single_explanations.to_csv("single_explanations.csv")

layer_specific_top_n = {"0": 10, "1": 10, "2": 10, "3": 10, "4": 10}
# 从原始 dataframe 生成整理后的边表
edges_df = binn_to_edges_dataframe(single_explanations, top_n=10)
edges_df.to_csv("edges_df.csv")
