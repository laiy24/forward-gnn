from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data


class BaseNodeGNNModel(torch.nn.Module, ABC):
	def __init__(self, num_classes: int, args: Any):
		super().__init__()
		self.num_classes = num_classes
		self.args = args

		# Shared training/eval hyperparameters
		self.theta = args.ff_theta
		self.patience = args.patience
		self.val_every = args.val_every
		self.append_label = args.append_label
		self.device = args.device
		self.model_type = args.model

		self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
		self.augmenter: Augmentor | None = None
		self.aug_graph: Data | None = None

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
		# Some GNNs require edge type, some do not.
		for i, layer in enumerate(self.layers):
			if edge_type is not None:
				try:
					x = layer(x, edge_index, edge_type)
				except TypeError:
					x = layer(x, edge_index)
			else:
				x = layer(x, edge_index)
		return x
	
	@abstractmethod
	def forward_train(self, *args, **kwargs) -> Any:
		"""Run one training step for this model."""

	@abstractmethod
	def eval_model(self, *args, **kwargs) -> Any:
		"""Evaluates the model and returns metric"""
		
	@staticmethod
	def _to_percent_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
		"""Returns accuracy as percentage in [0, 100]."""
		assert pred.shape == target.shape, (pred.shape, target.shape)
		correct = (pred == target).int().sum().item()
		return 100.0 * correct / max(1, target.shape[0])

	@staticmethod
	def _resolve_last_eval_layer(last_eval_layer: int, num_layers: int) -> int:
		"""Normalizes a possibly-negative layer index."""
		if last_eval_layer < 0:
			last_eval_layer = num_layers - 1
		if last_eval_layer >= num_layers:
			raise ValueError(f"last_eval_layer out of range: {last_eval_layer} for {num_layers} layers")
		return last_eval_layer


class Augmentor:
	def __init__(
		self,
		data: Data,
		append_label: Union[str, None],
		edge_direction: str,
		device: torch.device,
		non_augmented_nodes_style: str = "uniform", # options: "uniform", "zero"
	):
		assert non_augmented_nodes_style in ["zero", "uniform"], non_augmented_nodes_style
		assert edge_direction in ["unidirection", "bidirection"], edge_direction
		self.append_label = append_label
		self.edge_direction = edge_direction
		self.non_augmented_nodes_style = non_augmented_nodes_style
		x, y, edge_index = data.x, data.y, data.edge_index
		assert isinstance(x, torch.Tensor), type(x)
		assert isinstance(y, torch.Tensor), type(y)
		assert isinstance(edge_index, torch.Tensor), type(edge_index)
		self.x: torch.Tensor = x
		self.y: torch.Tensor = y.long()
		self.edge_index: torch.Tensor = edge_index

		y = self.y
		assert y.min() == 0, y.min()
		self.device = device

		self.num_classes = int((y.max() - y.min() + 1).item())
		self.num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(x.shape[0])

		# Generate virtual nodes
		self.onehot_labels = torch.eye(self.num_classes).to(device)
		self.real_node: torch.Tensor = torch.cat([
			torch.ones(self.num_nodes), # real
			torch.zeros(self.num_classes), # virtual
		]).bool().to(device)
		self.vn_ids = torch.arange(
			self.num_nodes,
			self.num_nodes + self.num_classes,
			device=device,
		)
		self.vn_features = nn.Embedding(
			self.num_classes,
			x.shape[1], # feature dimension
		).weight.clone().detach().to(device) # dimension: (num_classes, feature_dim), value initialized with random embedding and fixed during training
		self.aug_x = torch.cat([x, self.vn_features], dim=0)
		self.aug_y = torch.cat([
			y,
			torch.full((self.num_classes,), fill_value=-1, device=device, dtype=y.dtype),
		], dim=0)

	def augment(
		self,
		labels: Optional[torch.Tensor | int | float] = None,
		aug_node_mask: Optional[torch.Tensor] = None,
		node_feature_labels: Optional[torch.Tensor] = None,
	) -> Data:
		'''building the edge between real nodes and virtual nodes, and optionally appending one-hot label features to the node features.'''
		# 1) Normalize inputs and move once to target device.
		if labels is None:
			labels = self.y
		assert isinstance(labels, torch.Tensor), type(labels)
		assert labels.shape == self.y.shape, (labels.shape, self.y.shape)
		labels = labels.to(device=self.device, dtype=torch.long, non_blocking=True)

		if node_feature_labels is not None:
			node_feature_labels = node_feature_labels.to(device=self.device, dtype=torch.long, non_blocking=True)
		if aug_node_mask is None:
			aug_node_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
		else:
			assert isinstance(aug_node_mask, torch.Tensor), type(aug_node_mask)
			assert len(aug_node_mask) == len(self.x)
			aug_node_mask = aug_node_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

		# Reused views/index tensors to avoid repeated gather ops.
		masked_nodes = torch.arange(self.num_nodes, device=self.device)[aug_node_mask]
		masked_labels = labels[aug_node_mask]
		masked_virtual_nodes = self.vn_ids[masked_labels]

		# build edges
		'''1. building edges between real nodes and virtual nodes.
		   2. assign edge types: real edges have edge type 0, virtual edges have edge type in [1, num_classes] or [1, 2*num_classes] depending on directionality.'''
		if self.edge_direction == "bidirection":
			virtual_src = torch.cat([masked_nodes, masked_virtual_nodes])
			virtual_dst = torch.cat([masked_virtual_nodes, masked_nodes])
			virtual_edge_index = torch.stack([virtual_src, virtual_dst], dim=0)
			augmented_edge_index = torch.cat([self.edge_index, virtual_edge_index], dim=1)
			edge_type_bank = torch.arange(1, 1 + 2 * self.num_classes, device=self.device)
			virtual_edge_type = torch.cat([
				edge_type_bank[masked_labels],
				edge_type_bank[masked_labels + self.num_classes],
			])
		else:
			# real node -> virtual node only, no back edge
			virtual_src = masked_nodes
			virtual_dst = masked_virtual_nodes
			virtual_edge_index = torch.stack([virtual_src, virtual_dst], dim=0)
			augmented_edge_index = torch.cat([self.edge_index, virtual_edge_index], dim=1)
			edge_type_bank = torch.arange(1, 1 + self.num_classes, device=self.device)
			virtual_edge_type = edge_type_bank[masked_labels]
        # real edges have edge type 0, virtual edges have edge type in [1, num_classes] or [1, 2*num_classes] depending on directionality
		augmented_edge_type = torch.cat([
			torch.zeros(self.edge_index.shape[1], dtype=torch.long, device=self.device), 
			virtual_edge_type], 
			dim=0
		)

		# label appending
		# contruct one-hot for aug nodes
		'''put the label (one hot) right after each node features as new features'''
		real_node_one_hot = torch.zeros(self.num_nodes, self.num_classes, device=self.device)
		if node_feature_labels is None:
			node_feature_labels = labels
		real_node_one_hot[aug_node_mask] = self.onehot_labels[node_feature_labels[aug_node_mask]]

		# for non-augmented nodes, either fill with uniform distribution or zeros depending on non_augmented_nodes_style
		# sample uniform for 3 classes: [0.33, 0.33, 0.33]
		# sample non uniform for 3 classes: [0.0, 0.0, 0.0]
		if self.non_augmented_nodes_style == "uniform":
			non_aug_count = int((~aug_node_mask).sum().item())
			if non_aug_count > 0:
				uniform_row = torch.full((self.num_classes,), fill_value=1 / self.num_classes, device=self.device)
				real_node_one_hot[~aug_node_mask] = uniform_row.repeat(non_aug_count, 1)
		else:
			real_node_one_hot[~aug_node_mask] = 0.0
		all_node_one_hot = torch.cat([
			real_node_one_hot, 
			self.onehot_labels # virtial node one hot label
		], dim=0)

		# append label
		if self.append_label is not None and self.append_label != "none":
			augmented_node_feats = torch.cat([self.aug_x, all_node_one_hot], dim=1)
		else:
			augmented_node_feats = self.aug_x # user original features for real nodes and vn_features (random embedding) for virtual nodes

        # all 1 for real node and append 0 for virtual node in tthe end.
		augmented_node_mask = torch.cat([
			aug_node_mask,
			torch.zeros(self.num_classes, dtype=torch.bool, device=self.device),
		], dim=0)

		return Data(
			x=augmented_node_feats,
			node_one_hot_labels=all_node_one_hot,
			y=self.aug_y,
			edge_index=augmented_edge_index,
			edge_type=augmented_edge_type,
			num_nodes=self.num_nodes + self.num_classes,
			aug_node_mask=augmented_node_mask,
			real_node_mask=self.real_node,
		)
	
class LabelAppendingAugmentor:
	def __init__(
		self,
		label_names: List,
		device: torch.device,
		non_augmented_nodes_style: str = "uniform", # options: "uniform", "zero"
	):
		self.label_names = label_names
		self.device = device
		self.non_augmented_nodes_style = non_augmented_nodes_style
		self.onehot_labels = torch.stack([
			torch.nn.functional.one_hot(torch.tensor(i, device=device), num_classes=len(label_names))
			for i in range(len(label_names))
		]).float().to(device)

	@torch.no_grad()
	def train_data_append(
		self,
		x: torch.Tensor,
		y: torch.Tensor,
		train_mask: torch.Tensor,
		num_negs = 1
	) -> tuple[torch.Tensor, list[torch.Tensor]]:
		assert x.shape[0] == y.shape[0] == train_mask.shape[0], (x.shape, y.shape, train_mask.shape)
		num_nodes = x.shape[0]
		num_classes = len(self.label_names)
		num_negs = min(num_negs, num_classes - 1)
		num_train_nodes = int(train_mask.sum().item())
		num_non_train_nodes = num_nodes - num_train_nodes

		'''positive'''
		# construct one-hot for all nodes
		pos_node_one_hot = torch.zeros((num_nodes, num_classes), device=self.device)
		pos_node_one_hot[train_mask] = self.onehot_labels[y[train_mask].long()]
		# for non-augmented nodes, either fill with uniform distribution or zeros depending on non_augmented_nodes_style
		if self.non_augmented_nodes_style == "uniform":
			if num_non_train_nodes > 0:
				uniform_row = torch.full((num_classes,), fill_value=1 / num_classes, device=self.device)
				pos_node_one_hot[~train_mask] = uniform_row.repeat(num_non_train_nodes, 1)
		else:
			pos_node_one_hot[~train_mask] = 0.0
		pos_x = torch.cat([x, pos_node_one_hot], dim=1)

		'''negative'''
        # negative graph can has any lable except the real true one
        # we randomly select the lable in this case
		weights = torch.ones(
			(num_nodes, num_classes), 
			device=self.device
		)
		# set the probability of the TRUE class to 0.0 possibility
		weights[torch.arange(num_nodes), y] = 0.0
        # randomly pick classes based on weights
		neg_classes = torch.multinomial(weights, num_samples=num_negs, replacement=False)

		# neg feature
		neg_x_list = []
		for i in range(min(num_negs, neg_classes.shape[1])):
			neg_class_i = neg_classes[:, i]
			neg_node_one_hot = torch.zeros((num_nodes, num_classes), device=self.device)
			neg_node_one_hot[train_mask] = self.onehot_labels[neg_class_i[train_mask].long()]
			if self.non_augmented_nodes_style == "uniform":
				if num_non_train_nodes > 0:
					uniform_row = torch.full((num_classes,), fill_value=1 / num_classes, device=self.device)
					neg_node_one_hot[~train_mask] = uniform_row.repeat(num_non_train_nodes, 1)
			else:
				neg_node_one_hot[~train_mask] = 0.0
			neg_x_list.append(torch.cat([x, neg_node_one_hot], dim=1))

		return torch.cat([x, pos_node_one_hot], dim=1), neg_x_list
	
	@torch.no_grad()
	def eval_data_append(
		self,
		x: torch.Tensor,
		y: torch.Tensor,
		eval_calss: int,
		train_mask: torch.Tensor,
		eval_mask: torch.Tensor,
	) -> torch.Tensor:
		num_nodes = x.shape[0]
		num_classes = len(self.label_names)

		y_eval = torch.zeros((num_nodes,num_classes), dtype=torch.long, device=self.device)
		y_eval[train_mask] = self.onehot_labels[y[train_mask].long()]
		y_eval[eval_mask] = self.onehot_labels[eval_calss].repeat(int(eval_mask.sum().item()), 1)
		if self.non_augmented_nodes_style == "uniform":
			uniform_row = torch.full((num_classes,), fill_value=1 / num_classes, device=self.device)
			y_eval[~(train_mask | eval_mask)] = uniform_row.repeat(int((~(train_mask | eval_mask)).sum().item()), 1)
		else:
			y_eval[~(train_mask | eval_mask)] = 0.0
		return torch.cat([x, y_eval], dim=1)

