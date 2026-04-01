# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch_geometric.datasets import CitationFull, Amazon, GitHub
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected

import settings
from datasets.datasplit import DataSplit
from utils import logger


def _download_github_manually(raw_dir: Path) -> None:
    """Build GitHub/raw/github.npz from MUSAE raw files as fallback."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    base = "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input"
    edges_url = f"{base}/edges/git_edges.csv"
    target_url = f"{base}/target/git_target.csv"
    features_url = f"{base}/features/git.json"

    logger.info("Fallback GitHub download: loading target file...")
    with urlopen(target_url) as f:
        target_rows = list(csv.DictReader((line.decode("utf-8") for line in f)))

    id_to_idx = {int(row["id"]): i for i, row in enumerate(target_rows)}
    y = np.array([int(row["ml_target"]) for row in target_rows], dtype=np.int64)
    num_nodes = len(target_rows)

    logger.info("Fallback GitHub download: loading feature json...")
    with urlopen(features_url) as f:
        features_json = json.load(f)

    max_feature_id = -1
    for feat_list in features_json.values():
        if feat_list:
            max_feature_id = max(max_feature_id, max(int(v) for v in feat_list))
    num_features = max_feature_id + 1 if max_feature_id >= 0 else 0

    x = np.zeros((num_nodes, num_features), dtype=np.float32)
    for node_id_str, feat_list in features_json.items():
        node_id = int(node_id_str)
        if node_id not in id_to_idx:
            continue
        node_idx = id_to_idx[node_id]
        for feat in feat_list:
            x[node_idx, int(feat)] = 1.0

    logger.info("Fallback GitHub download: loading edges file...")
    edge_pairs = []
    with urlopen(edges_url) as f:
        for row in csv.DictReader((line.decode("utf-8") for line in f)):
            s = int(row["id_1"])
            t = int(row["id_2"])
            if s in id_to_idx and t in id_to_idx:
                edge_pairs.append([id_to_idx[s], id_to_idx[t]])

    edges = np.array(edge_pairs, dtype=np.int64)
    if edges.size > 0:
        rev_edges = edges[:, [1, 0]]
        edges = np.concatenate([edges, rev_edges], axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        edges = np.unique(edges, axis=0)

    out_path = raw_dir / "github.npz"
    np.savez(out_path, features=x, target=y, edges=edges)
    logger.info(f"Fallback GitHub file written to {out_path.resolve()}")


def load_node_classification_data(args, split_i):
    dataset = load_dataset(args.dataset)
    data = dataset[0]

    data_split = DataSplit(dataset_name=args.dataset, num_splits=5)
    node_split = data_split.load_node_split(split_i=split_i)
    logger.info(f"loaded node split from {data_split.node_split_root.resolve()}")

    data.train_mask = torch.zeros(data.num_nodes).bool()
    data.train_mask[node_split['train_node_index']] = True
    data.val_mask = torch.zeros(data.num_nodes).bool()
    data.val_mask[node_split['val_node_index']] = True
    data.test_mask = torch.zeros(data.num_nodes).bool()
    data.test_mask[node_split['test_node_index']] = True

    if not hasattr(data, "num_classes") or data.num_classes is None:
        data.num_classes = dataset.num_classes
    if not hasattr(data, "num_features") or data.num_features is None:
        data.num_features = dataset.num_features

    print()
    print(f'Dataset ({args.dataset}):')
    print('================================================================================')
    print(data)
    print(f"- Number of classes: {data.num_classes}")
    print(f"- Number of training nodes: {data.train_mask.sum()}")
    print(f"- Number of validation nodes: {data.val_mask.sum()}")
    print(f"- Number of testing nodes: {data.test_mask.sum()}")
    print('================================================================================')

    return data


def load_link_prediction_data(args, split_i):
    dataset = load_dataset(args.dataset)
    data = dataset[0]

    if not hasattr(data, "num_classes") or data.num_classes is None:
        data.num_classes = dataset.num_classes
    if not hasattr(data, "num_features") or data.num_features is None:
        data.num_features = dataset.num_features

    data_split = DataSplit(dataset_name=args.dataset, num_splits=5)
    edge_split = data_split.load_edge_split(split_i=split_i)
    logger.info(f"loaded edge split from {data_split.edge_split_root.resolve()}")
    train_data, val_data, test_data = edge_split['train_data'], edge_split['val_data'], edge_split['test_data']

    print()
    print(f'Dataset ({args.dataset}):')
    print('================================================================================')
    print("Raw Data:", data)
    print("Train Data:", train_data)
    print("Validation Data:", val_data)
    print("Testing Data:", test_data)
    print('================================================================================')

    return train_data, val_data, test_data, data


def load_dataset(dataset):
    if dataset.startswith("CitationFull"):
        dataset = CitationFull(
            root=f'{settings.DATA_ROOT}/CitationFull',
            name=dataset.strip().split("-")[1],
            transform=NormalizeFeatures()
        )
    elif dataset.startswith("Amazon"):
        dataset = Amazon(
            root=f'{settings.DATA_ROOT}/Amazon',
            name=dataset.strip().split("-")[1],
            transform=NormalizeFeatures()
        )
    elif dataset == "GitHub":
        github_root = Path(f'{settings.DATA_ROOT}/GitHub')
        try:
            dataset = GitHub(
                root=str(github_root),
                transform=NormalizeFeatures()
            )
        except Exception as e:
            logger.warning(f"PyG GitHub download failed ({type(e).__name__}: {e}); trying manual fallback.")
            _download_github_manually(github_root / "raw")
            dataset = GitHub(
                root=str(github_root),
                transform=NormalizeFeatures()
            )
    else:
        raise ValueError(f"Unavailable dataset: {dataset}")

    assert len(dataset) == 1, len(dataset)
    assert is_undirected(dataset[0].edge_index)

    return dataset
