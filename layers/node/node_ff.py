import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from layers.node.common import BaseNodeLayer, LayerNormalization
from models.shared.loss import forwardforward_loss_fn

class FFLabelAppendLayer(BaseNodeLayer):
    def __init__(
        self,
        gnn_layer,
        optimizer_name: str,
        optimizer_kwargs: dict,
        args: argparse.Namespace,
    ):
        super().__init__(gnn_layer=gnn_layer, optimizer_name=optimizer_name, optimizer_kwargs=optimizer_kwargs)
        self.args = args
        self.norm = LayerNormalization()
        self.loss = nn.CrossEntropyLoss()
        self.temperature = args.temperature

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
        return super()._forward(x, edge_index, edge_type)

    def forward_train(
        self,
        pos_x: torch.Tensor,
        neg_x_list: List[torch.Tensor],
        theta: float,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        train_mask: torch.Tensor,
    ):
        target_count = int(train_mask.sum().item())
        pos_target = torch.ones(target_count, device=pos_x.device)
        neg_target = torch.zeros(target_count, device=pos_x.device)

        # forward
        neg_layers_out = []
        for neg_x in neg_x_list:
            neg_layers_out.append(
                self.forward(
                    neg_x,
                    neg_edge_index
                )
            )
        pos_layer_out = self.forward(
            pos_x,
            pos_edge_index
        )


        #caculate loss
        cuml_logits_neg_sum = 0.0
        num_negs_total = 0
        loss_neg_parts: list[torch.Tensor] = []

        for _, new_layer_out in enumerate(neg_layers_out):
            neg_loss, neg_accumulated_logits = forwardforward_loss_fn(
                new_layer_out[train_mask],
                theta,
                neg_target,
            )
            loss_neg_parts.append(neg_loss)

            num_negs_total += target_count
            cuml_logits_neg_sum += float(neg_accumulated_logits) * target_count
        neg_loss = torch.stack(loss_neg_parts).mean()
        cuml_logits_neg = cuml_logits_neg_sum / num_negs_total
        pos_loss, cuml_logits_pos = forwardforward_loss_fn(pos_layer_out[train_mask], theta, pos_target)
        logits = [cuml_logits_pos, cuml_logits_neg]

        # update parameters
        self.optimizer.zero_grad()
        loss = pos_loss + neg_loss
        loss.backward()
        self.optimizer.step()
        return (pos_layer_out, neg_layers_out), logits
    
    @torch.no_grad()
    def forward_predict(
        self,
        x: torch.Tensor,
        theta: float,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_out = self.forward(
            x.detach(),
            edge_index,
            edge_type,
        )
        goodness = torch.square(layer_out).mean(dim=1) - theta
        return layer_out, goodness



class FFVirtualNodeLayer(BaseNodeLayer):
    def __init__(
        self,
        gnn_layer,
        optimizer_name: str,
        optimizer_kwargs: dict,
        args: argparse.Namespace,
    ):
        super().__init__(gnn_layer=gnn_layer, optimizer_name=optimizer_name, optimizer_kwargs=optimizer_kwargs)
        self.args = args
        self.norm = LayerNormalization()
        self.loss = nn.CrossEntropyLoss()
        self.temperature = args.temperature

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: Optional[torch.Tensor] = None):
        return super()._forward(x, edge_index, edge_type)

    def forward_train(
        self,
        pos_graph: Data,
        neg_graphs: List[Data],
        theta: float,
    ):
        assert pos_graph.x is not None
        assert pos_graph.edge_index is not None
        assert pos_graph.aug_node_mask is not None
        assert len(neg_graphs) > 0

        pos_x = pos_graph.x
        pos_edge_index = pos_graph.edge_index
        pos_aug_mask = pos_graph.aug_node_mask

        pos_target = torch.ones(pos_aug_mask.sum(), device=pos_x.device)
        neg_target = torch.zeros(neg_graphs[0].aug_node_mask.sum(), device=pos_x.device)
        
        # forward
        neg_layers_out = []
        for neg_graph in neg_graphs:
            assert neg_graph.x is not None
            assert neg_graph.edge_index is not None
            neg_layers_out.append(
                self.forward(
                    neg_graph.x,
                    neg_graph.edge_index,
                    neg_graph.edge_type,
                )
            )
        pos_layer_out = self.forward(
            pos_x,
            pos_edge_index,
            pos_graph.edge_type,
        )


        #caculate loss
        cuml_logits_neg_sum = 0.0
        num_negs_total = 0
        loss_neg_parts: list[torch.Tensor] = []

        for neg_graph, neg_out in zip(neg_graphs, neg_layers_out):
            neg_mask = neg_graph.aug_node_mask
            neg_loss, neg_accumulated_logits = forwardforward_loss_fn(
                neg_out[neg_mask],
                theta,
                neg_target,
            )
            loss_neg_parts.append(neg_loss)

            num_negs = int(neg_mask.sum().item())
            num_negs_total += num_negs
            cuml_logits_neg_sum += float(neg_accumulated_logits) * num_negs
        neg_loss = torch.stack(loss_neg_parts).mean()
        cuml_logits_neg = cuml_logits_neg_sum / num_negs_total
        pos_loss, cuml_logits_pos = forwardforward_loss_fn(pos_layer_out[pos_aug_mask], theta, pos_target)
        logits = [cuml_logits_pos, cuml_logits_neg]

        # update parameters
        self.optimizer.zero_grad()
        loss = pos_loss + neg_loss
        loss.backward()
        self.optimizer.step()
        return (pos_layer_out, neg_layers_out), logits
    
    @torch.no_grad()
    def forward_predict(
        self,
        x: torch.Tensor,
        theta: float,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_out = self.forward(
            x.detach(),
            edge_index,
            edge_type,
        )
        goodness = torch.square(layer_out).mean(dim=1) - theta
        return layer_out, goodness
