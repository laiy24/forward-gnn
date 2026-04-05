import traceback
from functools import partial
from timeit import default_timer as timer
from typing import Dict, List, Optional, Any, Set, cast

import torch
from torch_geometric.data import Data
from tqdm import tqdm


from models.shared.utils import load_perf_dict, PerformanceManager
from models.node.common import BaseNodeGNNModel, Augmentor, LabelAppendingAugmentor
from layers.node.common import LayerNormalization
from layers.node.node_ff import FFVirtualNodeLayer, FFLabelAppendLayer
from layers.conv_layer import ConvLayer
from utils import ResultManager, logger
from utils.train_utils import EarlyStopping


class NodeLabelAppendFFModel(BaseNodeGNNModel):
    def __init__(self, layer_size: list, num_classes: int, optimizer_name: str, optimizer_kwargs: dict, args: Any):
        super().__init__( num_classes, args)
        self.theta = args.ff_theta
        self.appender = LabelAppendingAugmentor(
            label_names=list(range(num_classes)),
            device=args.device,
        )

        # construct layers
        for i in range(len(layer_size) - 1):
            self.layers.append(LayerNormalization())
            # always do label appending in this model
            in_channels = layer_size[i] + num_classes if i == 0 else layer_size[i]
            self.layers.append(
                FFLabelAppendLayer(
                    gnn_layer=ConvLayer(gnn=self.model_type,
                                      in_channels=in_channels,
                                      out_channels=layer_size[i + 1]),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    args=args,
                )
            )

    def forward_train(
        self,
        data: Data,
        result_manager: ResultManager,
        run_i: int
    ):
        data = data.to(self.device)
        # set up training
        self.train()
        start = timer()
        train_epochs, best_val_epochs = [], []
        get_perf_dict = partial(load_perf_dict, start, train_epochs, best_val_epochs)


        # build new graph with virtual nodes
        assert data.x is not None
        assert data.edge_index is not None
        assert data.y is not None
        assert isinstance(data.y, torch.Tensor)
        pos_edge_index, neg_edge_index = data.edge_index, data.edge_index  # using the same edges for positive and negative samples
        assert pos_edge_index is not None and neg_edge_index is not None
        pos_x, neg_x_list = self.appender.train_data_append(
            data.x,
            data.y,
            data.train_mask,
            self.args.num_negs,
        )

        # start training
        for i, layer in enumerate(self.layers):
            # normalization layer
            if isinstance(layer, LayerNormalization):
                with torch.no_grad():
                    pos_x = layer(pos_x).detach()
                    neg_x_list = [layer(neg_x).detach() for neg_x in neg_x_list]
                continue

            epochs, epoch = tqdm(range(self.args.epochs)), -1
            stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
            perf_log = PerformanceManager("Accuracy")
            try:
                # clear the cached propagation results from the previous layer, if applicable
                clear_cached_propagate = getattr(layer, "clear_cached_propagate", None)
                if callable(clear_cached_propagate):
                    clear_cached_propagate()

                for epoch in epochs:
                    self.train()
                    _,logits = cast(FFLabelAppendLayer, layer).forward_train(
                        pos_x, neg_x_list, self.theta,
                        pos_edge_index, neg_edge_index,
                        data.train_mask
                    )
                    seperation = logits[0] - logits[1]
                    
                    # validation + testing
                    if epoch > self.args.val_from and (epoch + 1) % self.val_every == 0 \
                            and stopper is not None:
                        val_accuracy, _ = self.eval_model(
                            data=data,
                            train_mask=data.train_mask,
                            eval_mask=data.val_mask,
                            last_layer=i
                        )
                        if perf_log.update_val_perf(
                            val_perf=val_accuracy,
                            epoch=epoch,
                        ):  # perf improved, update test perf as well
                            test_accuracy, _ = self.eval_model(
                                data=data,
                                train_mask=data.train_mask | data.val_mask,
                                eval_mask=data.test_mask,
                                last_layer=i
                            )
                            perf_log.update_test_perf(
                                test_perf=test_accuracy,
                                epoch=epoch,
                            )

                        # checking early stop signal
                        if stopper.step(val_accuracy, layer):
                            print(f"[Layer {i}: Epoch-{epoch}] Early stop!")
                            break

                    epochs.set_description(
                        f"[Layer {i}: Epoch-{epoch}] "
                        f"Goodness: pos={logits[0]:.4f}, neg={logits[1]:.4f} | "
                        f"Separation: {seperation:.4f} | "
                        f"{' | '.join([perf_log.val_perf_summary(), perf_log.test_perf_summary()])}"
                    )
                
                # load best model for this layer (if applicable)
                if stopper is not None and stopper.best_score is not None:
                    stopper.load_checkpoint(layer)
                logger.info(f"Finished training layer {i + 1} / {len(self.layers)}.\n")
            except KeyboardInterrupt:
                print(f"\n=== LAYER-{i} TRAINING INTERRUPTED AT EPOCH-{epoch}! ===\n")
                if stopper is not None and stopper.best_score is not None:
                    stopper.load_checkpoint(layer)
            except Exception:
                traceback.print_exc()
                raise

            # adjust the graph input for next layer
            with torch.no_grad():
                pos_x = layer(pos_x, pos_edge_index)
                neg_x_list = [layer(neg_x, neg_edge_index) for neg_x in neg_x_list]

            # logging performance and saving results
            train_epochs.append(epoch)
            best_val_epochs.append(perf_log.best_val_epoch)
            test_acc, _ = self.eval_model(
                                data=data,
                                train_mask=data.train_mask | data.val_mask,
                                eval_mask=data.test_mask,
                                last_layer=i
                            )
            print(f"[Layer-{i}] Test Accuracy : {test_acc:.6f}%\n")
            result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_acc), num_layers=(i + 1) // 2) # skip normalization layers
            

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            data: Data,
            train_mask: torch.BoolTensor,
            eval_mask: torch.BoolTensor,
            last_layer: int = -1
    ):
        self.eval()
        assert data.x is not None
        assert data.edge_index is not None
        assert data.y is not None
        assert isinstance(data.y, torch.Tensor)
        last_layer = self._resolve_last_eval_layer(last_layer, num_layers=len(self.layers))


        accumulate_from = 0 if last_layer >= 1 else 0
        accumulated_probs_list: list[torch.Tensor] = []
        num_nodes = data.num_nodes
        if num_nodes is None:
            num_nodes = int(data.y.size(0))

        for each_class in range(self.num_classes):
            # generate eval graphs if not provided
            node_feats = self.appender.eval_data_append(
                data.x,
                data.y,
                each_class,
                train_mask,
                eval_mask
            )            
            accumulated_probs = torch.zeros(num_nodes, device=self.args.device)
            for i, layer in enumerate(self.layers):
                if i <= last_layer:
                    if isinstance(layer, LayerNormalization):
                        node_feats, prob = cast(LayerNormalization, layer).forward_predict(
                            node_feats,
                            self.theta,
                            data.edge_index,
                        )
                    else:
                        node_feats, prob = cast(FFLabelAppendLayer, layer).forward_predict(
                            node_feats,
                            self.theta,
                            data.edge_index,
                        )

                    if i >= accumulate_from:
                        accumulated_probs += prob.detach()
            accumulated_probs_list.append(accumulated_probs.view(-1, 1))

        if not accumulated_probs_list:
            raise RuntimeError("No probabilities accumulated for evaluation.")
        pred = torch.cat(accumulated_probs_list, dim=1)[eval_mask].argmax(dim=1)
        target = data.y[eval_mask]
        acc = self._to_percent_accuracy(pred, target)

        return acc, accumulated_probs_list

class NodeVirtualNodeFFModel(BaseNodeGNNModel):
    def __init__(self, layer_size: list, num_classes: int, optimizer_name: str, optimizer_kwargs: dict, args: Any):
        super().__init__( num_classes, args)
        self.theta = args.ff_theta

        # construct layers
        for i in range(len(layer_size) - 1):
            self.layers.append(LayerNormalization())
            if self.append_label == "input":
                if i == 0:
                    in_channels = layer_size[i] + num_classes
                else:
                    in_channels = layer_size[i]
            elif self.append_label == "all":
                in_channels = layer_size[i] + num_classes
            else:  # don't append one-hot label to node features
                in_channels = layer_size[i]
            self.layers.append(
                FFVirtualNodeLayer(
                    gnn_layer=ConvLayer(gnn=self.model_type,
                                      in_channels=in_channels,
                                      out_channels=layer_size[i + 1]),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    args=args,
                )
            )

    def create_eval_graph(
        self, 
        labels: torch.Tensor,
        train_mask: torch.BoolTensor,
        eval_mask: torch.BoolTensor
    ) -> Dict[int, Data]:
        result = {}
        for each_class in range(self.num_classes):
            # mask out not training nodes
            class_labels = torch.full_like(labels, fill_value=-1, device=labels.device)
            class_labels[train_mask] = labels[train_mask]
            class_labels[eval_mask] = each_class

            assert self.augmenter is not None, "Augmenter must be initialized before creating eval graph."
            aug_graph = self.augmenter.augment(
                class_labels,
                aug_node_mask=train_mask | eval_mask
            )
            result[each_class] = aug_graph
            assert aug_graph.num_nodes == self.augmenter.num_nodes+self.num_classes

        return result
    
    def creat_pos_neg_graph(
        self,
        num_negs: int,
        aug_nodes_mask: torch.BoolTensor,
    ) -> tuple[Data, List[Data]]:
        assert self.augmenter is not None, "Augmenter must be initialized before creating pos/neg graphs."
        num_classes = self.augmenter.num_classes
        num_negs = min(num_negs, num_classes - 1)


        pos_graph = self.augmenter.augment(
            self.augmenter.y,
            aug_node_mask=aug_nodes_mask
        )

        # negative graph can has any lable except the real true one
        # we randomly select the lable in this case
        weights = torch.ones(
            (self.augmenter.num_nodes, num_classes), 
            device=self.args.device
        )
        # set the probability of the TRUE class to 0.0 possibility
        weights[torch.arange(self.augmenter.num_nodes), self.augmenter.y] = 0.0
        # randomly pick classes based on weights
        neg_classes = torch.multinomial(weights, num_samples=num_classes - 1, replacement=False).to(self.device) # (num_nodes, num_classes-1)
        
        neg_graphs = []
        for i in range(min(num_negs, neg_classes.shape[1])):
            neg_graph = self.augmenter.augment(
                labels=neg_classes[:, i],
                aug_node_mask=aug_nodes_mask,
                node_feature_labels=self.augmenter.y
            )
            neg_graphs.append(neg_graph)
        
        return pos_graph, neg_graphs

    def forward_train(
        self,
        data: Data,
        result_manager: ResultManager,
        run_i: int
    ):
        data = data.to(self.device)
        # set up training
        self.train()
        start = timer()
        train_epochs, best_val_epochs = [], []
        get_perf_dict = partial(load_perf_dict, start, train_epochs, best_val_epochs)


        # build new graph with virtual nodes
        self.augmenter = Augmentor(
            data,
            self.append_label,
            self.args.aug_edge_direction,
            self.device
        )
        self.aug_graph = aug_graph = self.augmenter.augment(data.y,data.train_mask)
        assert aug_graph.x is not None
        assert aug_graph.edge_index is not None
        features = aug_graph.x
        assert isinstance(data.y, torch.Tensor)
        assert data.train_mask is not None
        assert data.val_mask is not None
        assert data.test_mask is not None
        labels = data.y
        train_mask = data.train_mask

        val_graphs = self.create_eval_graph(
            labels=labels,
            train_mask=train_mask,
            eval_mask=data.val_mask,
        )
        test_graph = self.create_eval_graph(
            labels=labels,
            train_mask=train_mask,
            eval_mask=data.test_mask,
        )

        # create pos and neg graph
        pos_graph, neg_graphs = self.creat_pos_neg_graph(
            num_negs=self.args.num_negs,
            aug_nodes_mask=data.train_mask
        )

        # start training
        for i, layer in enumerate(self.layers):
            # normalization layer
            if isinstance(layer, LayerNormalization):
                pos_graph.x = layer(pos_graph.x).detach()
                for neg_graph in neg_graphs:
                    neg_graph.x = layer(neg_graph.x).detach()
                continue

            epochs, epoch = tqdm(range(self.args.epochs)), -1
            stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
            perf_log = PerformanceManager("Accuracy")
            try:
                # clear the cached propagation results from the previous layer, if applicable
                clear_cached_propagate = getattr(layer, "clear_cached_propagate", None)
                if callable(clear_cached_propagate):
                    clear_cached_propagate()

                for epoch in epochs:
                    self.train()
                    _, logits = cast(FFVirtualNodeLayer, layer).forward_train(
                        pos_graph,neg_graphs, self.theta
                    )
                    separation = logits[0] - logits[1]
                    
                    # validation + testing
                    if epoch > self.args.val_from and (epoch + 1) % self.val_every == 0 \
                            and stopper is not None:
                        val_accuracy, _ = self.eval_model(
                            data=data,
                            train_mask=data.train_mask,
                            eval_mask=data.val_mask,
                            eval_graphs=val_graphs,
                            last_layer=i
                        )
                        if perf_log.update_val_perf(
                            val_perf=val_accuracy,
                            epoch=epoch,
                        ):  # perf improved, update test perf as well
                            test_accuracy, _ = self.eval_model(
                                data=data,
                                train_mask=data.train_mask | data.val_mask,
                                eval_mask=data.test_mask,
                                eval_graphs=test_graph,
                                last_layer=i
                            )
                            perf_log.update_test_perf(
                                test_perf=test_accuracy,
                                epoch=epoch,
                            )

                        # checking early stop signal
                        if stopper.step(val_accuracy, layer):
                            print(f"[Layer {i}: Epoch-{epoch}] Early stop!")
                            break

                    epochs.set_description(
                        f"[Layer {i}: Epoch-{epoch}] Pos={logits[0]:.4f}, Neg={logits[1]:.4f} | "
                        f"TrainSep: {separation:.4f} | "
                        f"{' | '.join([perf_log.val_perf_summary(), perf_log.test_perf_summary()])}"
                    )
                
                # load best model for this layer (if applicable)
                if stopper is not None and stopper.best_score is not None:
                    stopper.load_checkpoint(layer)
                logger.info(f"Finished training layer {i + 1} / {len(self.layers)}.\n")
            except KeyboardInterrupt:
                print(f"\n=== LAYER-{i} TRAINING INTERRUPTED AT EPOCH-{epoch}! ===\n")
                if stopper is not None and stopper.best_score is not None:
                    stopper.load_checkpoint(layer)
            except Exception:
                traceback.print_exc()
                raise

            # logging performance and saving results
            train_epochs.append(epoch)
            best_val_epochs.append(perf_log.best_val_epoch)
            test_acc, _ = self.eval_model(
                                data=data,
                                train_mask=data.train_mask | data.val_mask,
                                eval_mask=data.test_mask,
                                eval_graphs=test_graph,
                                last_layer=i
                            )
            print(f"[Layer-{i}] Test Accuracy : {test_acc:.6f}%\n")
            result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_acc), num_layers=(i + 1) // 2) # skip normalization layers

            # use output from this gnn layer as the input to the next layer
            features = layer.forward(features, aug_graph.edge_index, aug_graph.edge_type)
            if self.append_label == "all":
                features = torch.cat([features, aug_graph.node_one_hot_labels], dim=1)
            features = features.detach()

            # adjust the graph input for next layer
            pos_graph.x = features
            for neg_graph in neg_graphs:
                neg_graph.x = features

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            data: Data,
            train_mask: torch.BoolTensor,
            eval_mask: torch.BoolTensor,
            eval_graphs: Optional[Dict[int, Data]] = None,
            last_layer: int = -1
    ):
        self.eval()
        assert self.aug_graph is not None
        assert self.aug_graph.real_node_mask.sum() == len(eval_mask)
        assert self.augmenter is not None
        aug_graph = self.aug_graph.to(self.args.device)
        assert aug_graph.x is not None
        assert aug_graph.edge_index is not None
        assert aug_graph.y is not None
        assert isinstance(data.y, torch.Tensor)
        last_layer = self._resolve_last_eval_layer(last_layer, num_layers=len(self.layers))

        if eval_graphs is None:
            eval_graphs = self.create_eval_graph(
                labels=data.y,
                train_mask=train_mask,
                eval_mask=eval_mask,
            )

        accumulate_from = 0 if last_layer >= 1 else 0
        accumulated_probs_list: list[torch.Tensor] = []
        num_nodes = data.num_nodes
        if num_nodes is None:
            num_nodes = int(data.y.size(0))
        for each_class in range(self.num_classes):
            aug_graph = eval_graphs[each_class].to(self.args.device)
            assert aug_graph.x is not None
            assert aug_graph.edge_index is not None
            node_feats = aug_graph.x
            accumulated_probs = torch.zeros(num_nodes, device=self.args.device)
            for i, layer in enumerate(self.layers):
                if i <= last_layer:
                    if isinstance(layer, LayerNormalization):
                        node_feats, prob = cast(LayerNormalization, layer).forward_predict(
                            node_feats,
                            self.theta,
                            aug_graph.edge_index,
                            aug_graph.edge_type,
                        )
                    else:
                        node_feats, prob = cast(FFVirtualNodeLayer, layer).forward_predict(
                            node_feats,
                            self.theta,
                            aug_graph.edge_index,
                            aug_graph.edge_type,
                        )

                    if i >= accumulate_from:
                        accumulated_probs += prob[aug_graph.real_node_mask].detach()

                    # input to next layer
                    if self.append_label == "all":
                        assert aug_graph.node_one_hot_labels is not None
                        node_feats = torch.cat([node_feats, aug_graph.node_one_hot_labels], dim=1)
            accumulated_probs_list.append(accumulated_probs.view(-1, 1))

        if not accumulated_probs_list:
            raise RuntimeError("No probabilities accumulated for evaluation.")
        pred = torch.cat(accumulated_probs_list, dim=1)[eval_mask].argmax(dim=1)
        target = data.y[eval_mask]
        acc = self._to_percent_accuracy(pred, target)

        return acc, accumulated_probs_list
