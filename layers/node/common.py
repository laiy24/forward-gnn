import torch
from abc import ABC, abstractmethod
from typing import Any, Optional    

class LayerNormalization(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
		del edge_index 
		l2_norm = torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1, keepdim=True) + 1e-8 # L2 norm of each node's features
		return x / l2_norm
	
	def forward_train(
            self,
            input_tensor: torch.Tensor,
            signs: torch.Tensor,
            theta: float,
            pos_edge_index: Optional[torch.Tensor] = None,
            neg_edge_index: Optional[torch.Tensor] = None
    ):
		with torch.no_grad():
			output = self()
		return output, None

	@torch.no_grad()
	def forward_predict(
			self,
			input_feats: torch.Tensor,
			theta: float,
			edge_index: Optional[torch.Tensor] = None,
			edge_type: Optional[torch.Tensor] = None
	):
		output = self(input_feats)
		return output, torch.zeros(input_feats.shape[0], device=input_feats.device)




class BaseNodeLayer(torch.nn.Module, ABC):
	def __init__(self, gnn_layer: torch.nn.Module, optimizer_name: str, optimizer_kwargs: dict):
		super().__init__()
		self.gnn_layer = gnn_layer
		self.optimizer = getattr(torch.optim, optimizer_name)(
			gnn_layer.parameters(), **optimizer_kwargs
		)

	def _forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
		# Some GNNs require edge type, some do not.
		if edge_type is not None:
			try:
				return self.gnn_layer(x, edge_index, edge_type)
			except TypeError:
				return self.gnn_layer(x, edge_index)
		return self.gnn_layer(x, edge_index)

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