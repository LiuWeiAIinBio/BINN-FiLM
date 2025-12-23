import torch
import pandas as pd
from torch import nn
from torch.nn.utils import prune
import collections
from pathway_network import dataframes_to_pathway_network
from typing import List, Tuple, Any


class BINN(nn.Module):
    def __init__(self,
                 data_matrix: pd.DataFrame = None,
                 activation: str = "tanh",
                 n_layers: int = 4,
                 n_outputs: int = 2,
                 dropout: float = 0,
                 mapping: pd.DataFrame = None,
                 pathways: pd.DataFrame = None,
                 entity_col: str = "Protein",
                 input_col: str = "input",
                 translation_col: str = "translation",
                 target_col: str = "target",
                 source_col: str = "source"):

        super().__init__()
        # self.film = SampleFiLMGenerator()

        # Build connectivity from the pathway network
        pn = dataframes_to_pathway_network(data_matrix=data_matrix,
                                           pathway_df=pathways,
                                           mapping_df=mapping,
                                           input_col=input_col,
                                           target_col=target_col,
                                           source_col=source_col,
                                           entity_col=entity_col,
                                           translation_col=translation_col)

        # The connectivity matrices for each layer
        self.connectivity_matrices = pn.get_connectivity_matrices(n_layers=n_layers)

        # Collect layer sizes
        layer_sizes = []
        self.layer_names = []

        # First matrix => input layer size
        mat_first = self.connectivity_matrices[0]
        in_features, _ = mat_first.shape
        layer_sizes.append(in_features)

        self.inputs = mat_first.index.tolist()  # feature names
        self.layer_names.append(mat_first.index.tolist())

        # Additional layers
        for mat in self.connectivity_matrices[1:]:
            i, _ = mat.shape
            layer_sizes.append(i)
            self.layer_names.append(mat.index.tolist())

        # main layer
        self.layers = _generate_sequential(layer_sizes,
                                           self.connectivity_matrices,
                                           activation=activation,
                                           n_outputs=n_outputs,
                                           dropout=dropout,
                                           bias=True)

    #     # FiLM
    #     modulation_dims = [layer.num_features for layer in self.layers if isinstance(layer, nn.BatchNorm1d)]
    #
    #     print(modulation_dims)
    #
    #     self.modulation_heads = nn.ModuleList()
    #
    #     for dim in modulation_dims:
    #         self.modulation_heads.append(nn.Sequential(nn.Linear(3, 128),
    #                                                    nn.Linear(128, 128),
    #                                                    nn.Linear(128, 2 * dim)))  # 每个头输出 2*dim（gamma和beta各占dim维度）
    #
    #     # Weight init
    #     self.apply(_init_weights)
    #
    #
    # def forward(self, x: torch.Tensor, t: torch.Tensor) -> list[Any]:
    #     # FiLM
    #     modulation_params = []
    #     for head in self.modulation_heads:
    #         # embed task_ID
    #         t_encoded = torch.zeros(*t.shape, 3, device=t.device, dtype=t.dtype)
    #         mask_0 = (t == 0)
    #         mask_1 = (t == 1)
    #         mask_2 = (t == 2)
    #         t_encoded[mask_0] = torch.tensor([1, 0, 0], device=t.device, dtype=t.dtype)
    #         t_encoded[mask_1] = torch.tensor([0, 1, 0], device=t.device, dtype=t.dtype)
    #         t_encoded[mask_2] = torch.tensor([0, 0, 1], device=t.device, dtype=t.dtype)
    #
    #         out = head(t_encoded)  # (batch_size, 2*dim)
    #         gamma, beta = torch.chunk(out, 2, dim=1)  # 拆分为gamma和beta
    #         # gamma = gamma + 1
    #         modulation_params.append((gamma, beta))
    #
    #     # main network
    #     offset = 0
    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, nn.BatchNorm1d):
    #             x = layer(x)
    #             x = modulation_params[offset][0] * x + modulation_params[offset][1]
    #             offset += 1
    #         else:
    #             x = layer(x)
    #     return x

        # FiLM
        modulation_dims = [layer.num_features for layer in self.layers if isinstance(layer, nn.BatchNorm1d)]

        print("各层输出数量：", modulation_dims)

        self.modulation_heads = nn.ModuleList()

        for dim in modulation_dims:
            self.modulation_heads.append(nn.Sequential(nn.Linear(3, 128),
                                                       nn.Linear(128, 128),
                                                       nn.Linear(128, 2 * dim)))  # 每个头输出 2*dim（gamma和beta各占dim维度）

        # self.modulation_head = nn.Sequential(
        #     nn.Linear(16, 128),
        #     nn.Linear(128, 128),
        #     nn.Linear(128, 2 * sum(self.modulation_dims))
        # )  # 每个头输出 2*（gamma和beta）

        # Weight init
        self.apply(_init_weights)

        # self.task_embed = nn.Embedding(num_embeddings=3, embedding_dim=16)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> list[Any]:
        # FiLM: 单独调制头
        modulation_params = []
        for head in self.modulation_heads:
            # embed task_ID
            t_encoded = torch.zeros(*t.shape, 3, device=t.device, dtype=t.dtype)
            mask_0 = (t == 0)
            mask_1 = (t == 1)
            mask_2 = (t == 2)
            t_encoded[mask_0] = torch.tensor([1, 0, 0], device=t.device, dtype=t.dtype)
            t_encoded[mask_1] = torch.tensor([0, 1, 0], device=t.device, dtype=t.dtype)
            t_encoded[mask_2] = torch.tensor([0, 0, 1], device=t.device, dtype=t.dtype)

            out = head(t_encoded)  # (batch_size, 2*dim)
            gamma, beta = torch.chunk(out, 2, dim=1)  # 拆分为gamma和beta
            gamma = gamma + 1
            modulation_params.append((gamma, beta))

        # main network
        offset = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
                x = modulation_params[offset][0] * x + modulation_params[offset][1]
                offset += 1
            else:
                x = layer(x)
        return x

        # # FiLM: 共享调制头
        # # embed task_ID
        # t_encoded = self.task_embed(t.long())  # (batch_size, 2)
        #
        # out = self.modulation_head(t_encoded)  # (batch_size, 2*)
        #
        # # 按每层输出维度拆分 γ/β 参数
        # split_sizes = [2 * dim for dim in self.modulation_dims]
        # out_splits = torch.split(out, split_sizes, dim=1)  # 拆成每层的 (batch, 2*dim)
        # modulation_params = [torch.chunk(o, 2, dim=1) for o in out_splits]  # [(γ, β), (γ, β), ...]
        #
        # # main network
        # offset = 0
        # for layer in self.layers:
        #     if isinstance(layer, nn.BatchNorm1d):
        #         x = layer(x)
        #         gamma, beta = modulation_params[offset]
        #         x = (1 + gamma) * x + beta
        #         offset += 1
        #     else:
        #         x = layer(x)
        # return x


def _init_weights(m):
    """Initialize Linear layers with Xavier uniform."""
    if isinstance(m, nn.Linear):
        tanh_gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(m.weight, gain=tanh_gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _append_activation(layers, activation, i):
    if activation == "tanh":
        layers.append((f"Tanh_{i}", nn.Tanh()))
    elif activation == "relu":
        layers.append((f"ReLU_{i}", nn.ReLU()))
    elif activation == "leaky relu":
        layers.append((f"LeakyReLU_{i}", nn.LeakyReLU()))
    elif activation == "sigmoid":
        layers.append((f"Sigmoid_{i}", nn.Sigmoid()))
    elif activation == "elu":
        layers.append((f"Elu_{i}", nn.ELU()))
    elif activation == "hardsigmoid":
        layers.append((f"HardSigmoid_{i}", nn.Hardsigmoid()))
    return layers


def _generate_sequential(
        layer_sizes,
        connectivity_matrices=None,
        activation: str = "tanh",
        n_outputs: int = 2,
        dropout: float = 0,
        bias: bool = True):
    """
    Standard MLP layers, each with optional pruning from connectivity_matrices,
    plus a final output layer of size n_outputs.
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        lin = nn.Linear(in_size, out_size, bias=bias)
        layers.append((f"Linear_{i}", lin))
        layers.append((f"BatchNorm_{i}", nn.BatchNorm1d(out_size)))

        # Prune if a connectivity matrix is provided
        if connectivity_matrices is not None:
            mask = torch.tensor(connectivity_matrices[i].T.values, dtype=torch.float32)
            prune.custom_from_mask(lin, name="weight", mask=mask)

        # Activation
        _append_activation(layers, activation, i)

        # Dropout
        layers.append((f"Dropout_{i}", nn.Dropout(dropout)))

    # Final output layer
    layers.append(("Output", nn.Linear(layer_sizes[-1], n_outputs, bias=bias)))

    return nn.Sequential(collections.OrderedDict(layers))
