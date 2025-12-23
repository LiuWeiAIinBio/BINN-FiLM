import os
import numpy as np
import pandas as pd
from typing import Tuple, Any, Dict
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn.functional as F


def set_seed(seed):
    np.random.seed(seed)                  # NumPy 随机数
    torch.manual_seed(seed)               # CPU 上的 PyTorch 随机数
    torch.cuda.manual_seed(seed)          # 当前 GPU
    torch.cuda.manual_seed_all(seed)      # 所有 GPU（多卡时）
    torch.backends.cudnn.deterministic = True  # 确保卷积等确定性
    torch.backends.cudnn.benchmark = False     # 禁止自动调优以保证复现


# load data from reactome database
def load_reactome_db(mapping_path: str, pathways_path: str) -> dict:
    """
    mapping_path: directory of uniprot-reactome mapping file from Reactome database
    pathways_path: directory of pathways relation file from Reactome database
    """
    mapping = pd.read_csv(mapping_path, sep="\t", header=None, names=["input", "translation", "url", "name", "x", "species"],)
    pathways = pd.read_csv(pathways_path, sep="\t", header=None, names=["target", "source"])

    return {"mapping": mapping, "pathways": pathways}


# Split into train and validation sets
def train_validation_split(feature_data_dir: str, label_data_dir: str, valid_features_index) -> tuple[
    dict[int, Any], dict[int, Any], dict[int, Any], dict[int, Any], dict[int, Any], dict[int, Any]]:
    """
    feature_data_dir: directory of sample data
    label_data_dir: directory of label
    valid_features_index: index of valid features
    """
    sample_data = pd.read_csv(feature_data_dir)
    design_data = pd.read_csv(label_data_dir)

    # transpose sample_data
    sample_data_T = sample_data.set_index("Protein").T.reset_index().rename(columns={'index': 'sample'})

    # merge sample_data_T and label_data
    merged_data = pd.merge(sample_data_T.astype(float), design_data.astype(float), on='sample', how='inner')
    print(design_data, sample_data, merged_data)

    # extract feature data and lable data according to valid_features_index,
    # shape of merged_data is (sample size * features size), so use valid_features_index to index columns
    X = merged_data.drop(columns=['sample', 'group']).loc[:, valid_features_index]
    T = merged_data.loc[:, ['task_ID']]
    y = merged_data.loc[:, ['group']]

    # 联合任务与标签，构造分层依据
    y_joint = np.array([f"{t}_{yy}" for t, yy in zip(T.values, y.values)])

    # split
    X_train, X_val, T_train, T_val, y_train, y_val = {}, {}, {}, {}, {}, {}
    s5f = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(s5f.split(X, y_joint)):
        X_train[fold] = X.iloc[train_idx]
        T_train[fold] = T.iloc[train_idx]
        y_train[fold] = y.iloc[train_idx]
        X_val[fold] = X.iloc[val_idx]
        T_val[fold] = T.iloc[val_idx]
        y_val[fold] = y.iloc[val_idx]

    return X_train, X_val, T_train, T_val, y_train, y_val


# compute loss
def compute_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    return loss


# compute loss
def compute_loss1(outputs, labels):
    criterion = nn.BCELoss()
    if labels.ndim == 1:
        labels = labels.reshape((-1, 1))  # reshape labels to 2D

    encoder = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False)  # manually specify one hot code object
    labels_onehot = encoder.fit_transform(labels)
    labels_onehot = torch.tensor(labels_onehot, dtype=torch.float32)
    loss = criterion(outputs, labels_onehot)

    return loss


def focal_loss(outputs, labels, index_0, alpha=torch.tensor([1, 1, 1]), gamma=2.0):
    ce_loss = F.cross_entropy(outputs, labels, reduction='none')
    pt = torch.exp(-ce_loss)  # 等价于 softmax(logits)[target]
    alpha_t = alpha[index_0]
    ce_loss = alpha_t * ce_loss
    loss = ((1 - pt) ** gamma) * ce_loss

    return loss.mean()


# plot final logits(3D)
def plot_logits(logits, labels, description, save_dir):
    # 处理logits
    if logits.requires_grad:
        logits = logits.detach().numpy()  # 带有梯度的张量不能直接用.numpy()转换，因为这会破坏计算图的完整性，需要先detach剥离梯度信息，再转换为numpy数组

    # 处理标签（确保为numpy数组）
    if labels.requires_grad:
        labels = labels.detach().numpy()

    x = logits[:, 0]
    y = logits[:, 1]
    z = logits[:, 2]

    colors = ['red', 'green', 'blue']  # 三类标签对应的颜色
    labels_text = ['class 0', 'class 1', 'class 2']  # 图例文本

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for label in [0, 1, 2]:
        mask = (labels == label)

        ax.scatter(x[mask], y[mask], z[mask],
                   s=50,
                   c=colors[label],
                   alpha=0.7,
                   label=labels_text[label])

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.legend()
    ax.set_title('logits in 3D', fontsize=14, pad=20)

    plt.tight_layout()  # 自动调整布局，避免标签重叠
    plt.show()
    # plt.savefig(os.path.join(save_dir, f'{description}.png'))
