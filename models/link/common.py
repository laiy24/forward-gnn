from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data


class BaseLinkGNNModel(torch.nn.Module, ABC):
	def __init__(self, args: Any):
		super().__init__()
		self.args = args

		# Shared training/eval hyperparameters
		self.theta = args.ff_theta
		self.patience = args.patience
		self.val_every = args.val_every
		self.append_label = args.append_label
		self.device = args.device
		self.model_type = args.model

		self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
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
	def _resolve_last_eval_layer(last_eval_layer: int, num_layers: int) -> int:
		"""Normalizes a possibly-negative layer index."""
		if last_eval_layer < 0:
			last_eval_layer = num_layers - 1
		if last_eval_layer >= num_layers:
			raise ValueError(f"last_eval_layer out of range: {last_eval_layer} for {num_layers} layers")
		return last_eval_layer