import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from layers.node.common import BaseNodeLayer, LayerNormalization

class NodeSingleForwardLayer(BaseNodeLayer):
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
        x = self.norm(x, edge_index)
        return super()._forward(x, edge_index, edge_type)

    def forward_train(
        self,
        x: torch.Tensor,
        augmenter,
        train_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> float:
        layer_out = self.forward(
            x.detach(),
            edge_index,
            edge_type,
        )
        real_node_mask = augmenter.real_node

        logits = torch.mm(
            layer_out[real_node_mask][train_mask], #train
            layer_out[~real_node_mask].t() #virtual
        ) # similarity score
        logits = logits / self.temperature

        # update parameters
        train_y = augmenter.y[train_mask]
        self.optimizer.zero_grad()
        loss = self.loss(logits, train_y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def forward_predict(
        self,
        x: torch.Tensor,
        augmenter,
        eval_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        layer_out = self.forward(
            x.detach(),
            edge_index,
            edge_type,
        )
        real_node_mask = augmenter.real_node

        logits = torch.mm(
            layer_out[real_node_mask][eval_mask], #eval
            layer_out[~real_node_mask].t() #virtual
        ) # similarity score
        logits = logits / self.temperature
        prob = torch.softmax(logits, dim=1)
        return prob

class NodeSFTop2LossLayer(BaseNodeLayer):
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
        states: List[torch.Tensor],
        layer_index: int,
        augmenter,
        train_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float]:
        input = states[layer_index]
        cur_layer_out = self.forward(
            input.detach(),
            edge_index,
            edge_type
        )
        states_upper_layer = states[layer_index + 2:]
        merge_states = torch.stack([cur_layer_out] + \
            [state.detach() for state in states_upper_layer])
        layer_out = torch.mean(merge_states, dim=0)

        real_node_mask = augmenter.real_node
        logits = torch.mm(
            layer_out[real_node_mask][train_mask], #train
            layer_out[~real_node_mask].t() #virtual
        ) # similarity score
        logits = logits / self.temperature

        # update parameters
        train_y = augmenter.y[train_mask]
        self.optimizer.zero_grad()
        loss = self.loss(logits, train_y)
        loss.backward()
        self.optimizer.step()
        return cur_layer_out, loss.item()
    
    @torch.no_grad()
    def forward_predict(
        self,
        states: List[torch.Tensor],
        layer_index: int,
        augmenter,
        eval_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = states[layer_index]
        cur_layer_out = self.forward(
            input.detach(),
            edge_index,
            edge_type
        )
        states_upper_layer = states[layer_index + 2:]
        merge_states = torch.stack([cur_layer_out] + \
            [state.detach() for state in states_upper_layer])
        layer_out = torch.mean(merge_states, dim=0)

        real_node_mask = augmenter.real_node
        logits = torch.mm(
            layer_out[real_node_mask][eval_mask], #eval
            layer_out[~real_node_mask].t() #virtual
        ) # similarity score
        logits = logits / self.temperature
        prob = torch.softmax(logits, dim=1)
        return prob
 

class NodeSFTop2InputLayer(BaseNodeLayer):
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

    def forward(self, x_prev, x_next, edge_index, edge_type=None):
        # top2input use both previuos layer and next layer output as input
        x = torch.cat((self.norm(x_prev), self.norm(x_next)), dim=1).detach()
        return super()._forward(x, edge_index, edge_type)

    def forward_train(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        augmenter,
        train_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float]:
        layer_out = self.forward(
            x_prev.detach(),
            x_next.detach(),
            edge_index,
            edge_type
        )
        real_node_mask = augmenter.real_node

        logits = torch.mm(
            layer_out[real_node_mask][train_mask], #train
            layer_out[~real_node_mask].t() #virtual
        ) # similarity score
        logits = logits / self.temperature

        # update parameters
        train_y = augmenter.y[train_mask]
        self.optimizer.zero_grad()
        loss = self.loss(logits, train_y)
        loss.backward()
        self.optimizer.step()
        return layer_out, loss.item()
    
    @torch.no_grad()
    def forward_predict(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        augmenter,
        eval_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_out = self.forward(
            x_prev.detach(),
            x_next.detach(),
            edge_index,
            edge_type,
        )
        real_node_mask = augmenter.real_node

        logits = torch.mm(
            layer_out[real_node_mask][eval_mask], #eval
            layer_out[~real_node_mask].t() #virtual
        ) # similarity score
        logits = logits / self.temperature
        prob = torch.softmax(logits, dim=1)
        return layer_out, prob
 