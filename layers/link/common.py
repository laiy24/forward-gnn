from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from models.shared.loss import forwardforward_loss_fn



class BaseLinkLayer(torch.nn.Module, ABC):
	def __init__(self, gnn_layer: torch.nn.Module, optimizer_name: str, optimizer_kwargs: dict):
		super().__init__()
		self.gnn_layer = gnn_layer
		self.optimizer = getattr(torch.optim, optimizer_name)(
			gnn_layer.parameters(), **optimizer_kwargs
		)
		self.loss = nn.BCEWithLogitsLoss(reduction='none')
		
	@staticmethod
	def link_predict(z, label_index):
		pred = z[label_index[0]] * z[label_index[1]]
		return pred

	@staticmethod
	def forwardforward_loss(z, out, edge_label, theta):
		is_pos = edge_label == 1
		pos_out = out[is_pos]
		pos_loss, cumul_pos_logits = forwardforward_loss_fn(pos_out, theta, target=1.0)

		is_neg = edge_label == 0
		neg_out = out[is_neg]
		neg_loss, cumul_neg_logits = forwardforward_loss_fn(neg_out, theta, target=0.0)
		return pos_loss + neg_loss, (pos_out, neg_out), (cumul_pos_logits, cumul_neg_logits)

	def forward_loss(self, out, edge_label):
		# BCEWithLogitsLoss, sigmoid to probability then binary cross entropy loss
		loss = self.loss(out.sum(dim=1), edge_label)
		
		is_pos = edge_label == 1
		pos_loss = loss[is_pos]
		is_neg = edge_label == 0
		neg_loss = loss[is_neg]
		
		with torch.no_grad():
			cumul_pos_logits = pos_loss.exp().mean().item()
			cumul_neg_logits = (1 - neg_loss.exp()).mean().item()
		return loss.mean(), (out[is_pos], out[is_neg]), (cumul_pos_logits, cumul_neg_logits)



	def _forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
		try:
			return self.gnn_layer(x, edge_index)
		except TypeError:
			if edge_type is None:
				raise
			return self.gnn_layer(x, edge_index, edge_type)

	def clear_cached_propagate(self) -> None:
		"""Clear cached propagation if underlying conv supports it."""
		gnn = getattr(self.gnn_layer, "gnn", None)
		if gnn is not None and hasattr(gnn, "clear_cached"):
			gnn.clear_cached()
	
	@abstractmethod
	def forward(self, *args, **kwargs) -> Any:
		"""Run one training step for this layer."""

	@abstractmethod
	def forward_train(self, *args, **kwargs) -> Any:
		"""Run one training step for this layer."""

	@abstractmethod
	def forward_predict(self, *args, **kwargs) -> Any:
		"""Run one prediction step for this layer."""


class LayerNormalization(torch.nn.Module):
    """Applies row wise normalization"""
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index: Optional[torch.Tensor] = None):
        l2_norm = (
            torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1, keepdim=True)
            + 1e-8
        )
        return x / l2_norm

    def forward_train(
            self,
            train_data: Data,
            theta: float,
            **kwargs
    ):
        with torch.no_grad():
            output = self(train_data.x)
        return output, None

    @torch.no_grad()
    def forward_predict(
            self,
            input_feats: torch.Tensor,
            edge_index: torch.Tensor,
            edge_label_index: torch.Tensor,
            theta: float,
    ):
        output = self(input_feats)
        return output, torch.zeros(edge_label_index.shape[1], device=edge_label_index.device)
