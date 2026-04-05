import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from layers.link.common import BaseLinkLayer, LayerNormalization

class LinkForwardLayer(BaseLinkLayer):
    def __init__(
        self,
        gnn_layer,
        optimizer_name: str,
        optimizer_kwargs: dict,
        args: argparse.Namespace,
    ):		
        super().__init__(gnn_layer, optimizer_name, optimizer_kwargs)
        self.args = args
        self.temperature = args.temperature

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
        return super()._forward(x, edge_index, edge_type)
    
    def forward_train(
            self,
            data: Data,
            theta: float,
    ):
        assert data.edge_index is not None
        assert data.x is not None
        self.optimizer.zero_grad()
        z = self.forward(data.x, data.edge_index)
        out = self.link_predict(z, data.edge_label_index)

        # update parameters
        loss, outs, logits = super().forward_loss(out, data.edge_label.float())
        loss.backward()
        self.optimizer.step()
        return outs, logits
    
    @torch.no_grad()
    def forward_predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label_index: torch.Tensor,
        theta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward(x, edge_index)
        out = self.link_predict(z, label_index)
        score = out.sum(dim=1).sigmoid()
        return z, score


class LinkForwardTopDownLayer(BaseLinkLayer):
    def __init__(
        self,
        gnn_layer,
        optimizer_name: str,
        optimizer_kwargs: dict,
        args: argparse.Namespace,
    ):		
        super().__init__(gnn_layer, optimizer_name, optimizer_kwargs)
        self.args = args
        self.temperature = args.temperature
        self.norm = LayerNormalization()

    def forward(self, x_prev: torch.Tensor, x_next: Optional[torch.Tensor], edge_index: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
        if x_next is not None:
            x = torch.cat((self.norm(x_prev), self.norm(x_next)), dim=1).detach()
        else:
            x = self.norm(x_prev).detach()
        return super()._forward(x, edge_index, edge_type)
    
    def forward_train(
            self,
            x_prev: torch.Tensor,
            x_next: Optional[torch.Tensor],
            data: Data,
    ):
        assert data.edge_index is not None
        assert data.x is not None
        assert data.edge_label is not None
        self.optimizer.zero_grad()
        z = self.forward(x_prev, x_next, data.edge_index)
        out = self.link_predict(z, data.edge_label_index).sum(dim=1)
        losses = self.loss(out, data.edge_label.float())

        # update parameters
        loss = losses.mean()
        loss.backward()
        self.optimizer.step()

        # calculate logits
        pos_mask = data.edge_label == 1
        neg_mask = data.edge_label == 0
        cumul_logits_pos = losses[pos_mask].exp().mean().item()
        cumul_logits_neg = (1-losses[neg_mask].exp()).mean().item()
        logits = [cumul_logits_pos, cumul_logits_neg]
        return z, (out[pos_mask], out[neg_mask]), logits
    
    @torch.no_grad()
    def forward_predict(
        self,
        x_prev: torch.Tensor,
        x_next: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        label_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward(x_prev, x_next, edge_index)
        out = self.link_predict(z, label_index).sum(dim=1)
        return z, out


