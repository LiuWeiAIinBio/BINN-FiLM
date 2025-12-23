import pandas as pd
import os
import time
import torch
from util import load_reactome_db, train_validation_split
from pathway_network import dataframes_to_pathway_network

# description = "肝胆癌血浆蛋白组，56个控制【标签为0】，57个胆管癌CCA【标签为1】，148个肝细胞癌HCC【标签为2】；" \
#               "将标签为0和标签为1的样本的task_ID标记为0，编码为[1, 0]，将标签为0和标签为2的样本的task_ID标记为1，编码为[0, 1]，标签为0的样本分成了两份。"

# super parameters
activation = "tanh"
n_layers = 4
n_outputs = 2
dropout = 0.1
weight_decay = 0.01
device = torch.device("cpu")
BATCH_SIZE = 7
num_epochs = 1000
learning_rate = 1e-4


# file dir
base_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(base_dir, "..", "data", "download_data", "uniprot_2_reactome_2025_01_14.txt")
pathways_path = os.path.join(base_dir, "..", "data", "download_data", "reactome_pathways_relation_2025_01_14.txt")

dataset_folder_name = "PDAC150_1"
feature_data_filename = "sample_data_matrix_PDAC150_zscore.csv"
label_data_filename = "sample_design_matrix_PDAC150.csv"
feature_data_dir = os.path.join(base_dir, "..", "data", "example_data", dataset_folder_name, feature_data_filename)
label_data_dir = os.path.join(base_dir, "..", "data", "example_data", dataset_folder_name, label_data_filename)
evaluate_feature_dir = ""
evaluate_label_dir = ""

timestamp = time.strftime("%Y%m%d_%H%M%S")
checkpoint_path = os.path.join(base_dir, "..", "output", timestamp, "checkpoint")
save_dir = os.path.join(base_dir, "..", "output", timestamp, "logs")


# load file info
mapping = load_reactome_db(mapping_path, pathways_path)["mapping"]
pathways = load_reactome_db(mapping_path, pathways_path)["pathways"]
data_matrix = pd.read_csv(feature_data_dir)
design_matrix = pd.read_csv(label_data_dir)


"""
create pathway_network instance by initialize PathwayNetwork class,
then extract pathway_network.get_connectivity_matrices(n_layers)[0].index which
index valid features and used in func util.train_validation_split()
"""
pathway_network = dataframes_to_pathway_network(data_matrix=data_matrix, pathway_df=pathways, mapping_df=mapping)
valid_features_index = [i for i in pathway_network.get_connectivity_matrices(n_layers)[0].index]
valid_features_amount = len(valid_features_index)


# split sample into train and val
X_train, X_val, T_train, T_val, y_train, y_val = train_validation_split(feature_data_dir, label_data_dir,
                                                                        valid_features_index=valid_features_index)
