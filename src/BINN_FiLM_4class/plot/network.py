import pandas as pd
import networkx as nx
from plot.util import load_default_mapping, build_mapping_dict, rename_node_by_layer

def binn_to_edges_dataframe(
    dataframe,
    top_n=5,
    layer_normalization_value=1,
    input_entity_mapping=None,
    pathways_mapping=None,
    input_entity_key_col="input_id",
    input_entity_val_col="input_name",
    pathways_key_col="pathway_id",
    pathways_val_col="pathway_name",
):
    """
    从原始输入 dataframe 构建经过筛选和标准化的边表，保留以下列：
    ['source_layer', 'target_layer', 'source_node', 'target_node', 'importance', 'normalized_importance']

    参数与 visualize_binn() 一致，但仅返回整理后的 DataFrame，不绘图。
    """
    # ----------------------------------------------------------
    # 1) 构建映射字典
    # ----------------------------------------------------------
    if isinstance(input_entity_mapping, str):
        input_entity_df = load_default_mapping(input_entity_mapping)
        input_entity_dict = build_mapping_dict(
            input_entity_df, key_col=input_entity_key_col, val_col=input_entity_val_col
        )
    elif isinstance(input_entity_mapping, pd.DataFrame):
        input_entity_dict = build_mapping_dict(
            input_entity_mapping, key_col=input_entity_key_col, val_col=input_entity_val_col
        )
    else:
        input_entity_dict = None

    if isinstance(pathways_mapping, str):
        pathways_df = load_default_mapping(pathways_mapping)
        pathways_dict = build_mapping_dict(
            pathways_df, key_col=pathways_key_col, val_col=pathways_val_col
        )
    elif isinstance(pathways_mapping, pd.DataFrame):
        pathways_dict = build_mapping_dict(
            pathways_mapping, key_col=pathways_key_col, val_col=pathways_val_col
        )
    else:
        pathways_dict = None

    # ----------------------------------------------------------
    # 2) 准备层级、tuple 信息
    # ----------------------------------------------------------
    dataframe["source_layer"] = dataframe["source_layer"].astype(str)
    dataframe["target_layer"] = dataframe["target_layer"].astype(str)
    dataframe["source_tuple"] = list(zip(dataframe["source_node"], dataframe["source_layer"]))
    dataframe["target_tuple"] = list(zip(dataframe["target_node"], dataframe["target_layer"]))

    value_column = (
        "normalized_importance" if "normalized_importance" in dataframe.columns else "importance"
    )

    # ----------------------------------------------------------
    # 3) 计算每层 top_n 节点
    # ----------------------------------------------------------
    source_importance_df = (
        dataframe.groupby(["source_tuple", "source_layer"])[value_column]
        .sum()
        .reset_index()
    )

    source_importance_df[value_column] = source_importance_df.groupby("source_layer")[
        value_column
    ].transform(lambda x: x / x.sum() * layer_normalization_value)

    if isinstance(top_n, int):
        top_n = {layer: top_n for layer in source_importance_df["source_layer"].unique()}

    def pick_top(group):
        n = top_n.get(group.name, 0)
        return group.nlargest(n, value_column)

    top_nodes_by_layer = (
        source_importance_df.groupby("source_layer").apply(pick_top).reset_index(drop=True)
    )
    top_source_nodes = set(top_nodes_by_layer["source_tuple"])

    # ----------------------------------------------------------
    # 4) sink 节点定义与替换
    # ----------------------------------------------------------
    all_layers = set(dataframe["source_layer"].unique()).union(
        set(dataframe["target_layer"].unique())
    )
    final_layer = max(int(l) for l in all_layers)
    sink_nodes = {
        l: ("output_node", l) if int(l) == final_layer else ("sink", l) for l in all_layers
    }

    # ----------------------------------------------------------
    # 5) 构建图并累计权重
    # ----------------------------------------------------------
    G = nx.DiGraph()
    for _, row in dataframe.iterrows():
        s_tuple = row["source_tuple"]
        t_tuple = row["target_tuple"]
        s_layer, t_layer = row["source_layer"], row["target_layer"]
        importance = row[value_column]

        new_source = s_tuple if s_tuple in top_source_nodes else sink_nodes[s_layer]
        new_target = t_tuple if t_tuple in top_source_nodes else sink_nodes[t_layer]

        if G.has_edge(new_source, new_target):
            G[new_source][new_target]["weight"] += importance
        else:
            G.add_edge(new_source, new_target, weight=importance)

    # ----------------------------------------------------------
    # 6) 生成边表 DataFrame
    # ----------------------------------------------------------
    edge_records = []
    for s, t, attrs in G.edges(data=True):
        record = {
            "source_layer": s[1],
            "target_layer": t[1],
            "source_node": s[0],
            "target_node": t[0],
            "importance": attrs.get("weight", 0.0),
        }
        edge_records.append(record)

    edges_df = pd.DataFrame(edge_records)

    # 层内标准化 normalized_importance
    edges_df["normalized_importance"] = edges_df.groupby("source_layer")["importance"].transform(
        lambda x: x / x.max() if x.max() > 0 else x
    )

    return edges_df
