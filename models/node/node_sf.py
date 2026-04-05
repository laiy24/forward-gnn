import traceback
from functools import partial
from timeit import default_timer as timer
from typing import List, Optional, Any, cast

import torch
from torch_geometric.data import Data
from tqdm import tqdm


from models.shared.utils import load_perf_dict, PerformanceManager
from models.node.common import BaseNodeGNNModel, Augmentor
from layers.node.common import LayerNormalization
from layers.node.node_sf import NodeSingleForwardLayer, NodeSFTop2InputLayer, NodeSFTop2LossLayer
from layers.conv_layer import ConvLayer
from utils import ResultManager, logger
from utils.train_utils import EarlyStopping


class NodeSingleForwardModel(BaseNodeGNNModel):
    def __init__(self, layer_size: list, num_classes: int, optimizer_name: str, optimizer_kwargs: dict, args: Any):
        super().__init__( num_classes, args)

        # construct layers
        for i in range(len(layer_size) - 1):
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
                NodeSingleForwardLayer(
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

        # start training
        for i, layer in enumerate(self.layers):
            epochs, epoch = tqdm(range(self.args.epochs)), -1
            stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
            perf_log = PerformanceManager("Accuracy")
            try:
                # clear the cached propagation results from the previous layer, if applicable
                clear_cached_propagate = getattr(layer, "clear_cached_propagate", None)
                if callable(clear_cached_propagate):
                    clear_cached_propagate()

                for epoch in epochs:
                    layer_loss = cast(NodeSingleForwardLayer, layer).forward_train(
                        features,
                        self.augmenter,
                        train_mask=data.train_mask,
                        edge_index=aug_graph.edge_index,
                        edge_type=aug_graph.edge_type,
                    )
                    
                    # validation + testing
                    if epoch > self.args.val_from and (epoch + 1) % self.val_every == 0 \
                            and stopper is not None:
                        val_accuracy, _ = self.eval_model(eval_mask=data.val_mask, last_layer=i)
                        if perf_log.update_val_perf(
                            val_perf=val_accuracy,
                            epoch=epoch,
                        ):  # perf improved, update test perf as well
                            test_accuracy, _ = self.eval_model(eval_mask=data.test_mask, last_layer=i)
                            perf_log.update_test_perf(
                                test_perf=test_accuracy,
                                epoch=epoch,
                            )

                        # checking early stop signal
                        if stopper.step(val_accuracy, layer):
                            print(f"[Layer {i}: Epoch-{epoch}] Early stop!")
                            break

                    epochs.set_description(
                        f"[Layer {i}: Epoch-{epoch}] Loss={layer_loss:.4f} | "
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
            test_acc, _ = self.eval_model(eval_mask=data.test_mask, last_layer=i)
            print(f"[Layer-{i}] Test Accuracy : {test_acc:.6f}%\n")
            result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_acc), num_layers=i + 1)

            # use output from this gnn layer as the input to the next layer
            features = layer.forward(features, aug_graph.edge_index, aug_graph.edge_type)
            if self.append_label == "all":
                features = torch.cat([features, aug_graph.node_one_hot_labels], dim=1)
            features = features.detach()

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            eval_mask: torch.BoolTensor,
            last_layer: int = -1
    ):
        self.eval()
        assert self.aug_graph is not None
        assert self.aug_graph.real_node_mask.sum() == len(eval_mask)
        assert self.augmenter is not None
        aug_graph = self.aug_graph.to(self.args.device)
        assert aug_graph.x is not None
        assert aug_graph.edge_index is not None
        last_layer = self._resolve_last_eval_layer(last_layer, num_layers=len(self.layers))

        accumulate_from = 0 if last_layer >= 1 else 0
        node_feats = aug_graph.x
        accumulated_probs: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            if i <= last_layer:
                prob = cast(NodeSingleForwardLayer, layer).forward_predict(
                    node_feats,
                    self.augmenter,
                    eval_mask,
                    aug_graph.edge_index,
                    aug_graph.edge_type,
                )

                if i >= accumulate_from:
                    accumulated_probs.append(prob.detach())

                # input to next layer
                node_feats = layer.forward(node_feats, aug_graph.edge_index, aug_graph.edge_type)

                if self.append_label == "all":
                    assert aug_graph.node_one_hot_labels is not None
                    node_feats = torch.cat([node_feats, aug_graph.node_one_hot_labels], dim=1)

        if not accumulated_probs:
            raise RuntimeError("No probabilities accumulated for evaluation.")
        pred = torch.stack(accumulated_probs, dim=0).sum(dim=0).argmax(dim=1)
        target = self.augmenter.y[eval_mask]
        acc = self._to_percent_accuracy(pred, target)

        return acc, accumulated_probs


class NodeSFTop2LossModel(BaseNodeGNNModel):
    def __init__(self, layer_size: list, num_classes: int, optimizer_name: str, optimizer_kwargs: dict, args: Any):
        super().__init__( num_classes, args)
        self.epochs = args.epochs
        self.states = []
        self.layer_sizes = layer_size
        self.norm = LayerNormalization()


        # construct layers
        for i in range(len(layer_size) - 1):
            in_channels = layer_size[i] 
            if self.append_label == "input" and i == 0:
                if i == 0:
                    in_channels = in_channels + num_classes

            self.layers.append(
                NodeSFTop2LossLayer(
                    gnn_layer=ConvLayer(gnn=self.model_type,
                                      in_channels=in_channels,
                                      out_channels=layer_size[i + 1]),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    args=args,
                )
            )

    # forward all layers at time stamp 0
    # do one forward to fill up states
    @torch.no_grad()
    def forward_all_layer_first_time(
      self,
      x: torch.Tensor,
      edge_index: torch.Tensor,
      edge_type: Optional[torch.Tensor] = None      
    )-> List[torch.Tensor]:
        assert x.ndim == 2, x.ndim
        x_new = x
        states = [self.norm(x_new)]

        for _, layer in enumerate(self.layers):
            x_new = layer.forward(
                x=x_new, 
                edge_index=edge_index,
                edge_type=edge_type,
            )
            states.append(self.norm(x_new).detach())
        return states  


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
        get_perf_dict = partial(load_perf_dict, start)


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

        # build first time state memory
        states = self.forward_all_layer_first_time(
            x=aug_graph.x,
            edge_index=aug_graph.edge_index,
            edge_type=aug_graph.edge_type,
        )

        # start training
        epochs = tqdm(range(self.epochs))
        epoch = -1
        stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
        perf_log = PerformanceManager("Accuracy")
        try:
            for epoch in epochs:
                new_states: List[torch.Tensor] = list(states)
                new_states[0] = states[0]
                total_loss = 0.0
                for i, layer in enumerate(self.layers):
                    node_embed, loss = cast(NodeSFTop2LossLayer, layer).forward_train(
                        states=states,
                        layer_index=i,
                        augmenter=self.augmenter,
                        train_mask=data.train_mask,
                        edge_index=aug_graph.edge_index,
                        edge_type=aug_graph.edge_type,
                    )
                    new_states[i + 1] = self.norm(node_embed).detach()
                    total_loss += loss
                states = new_states
            
                # validation + testing
                if epoch > self.args.val_from and (epoch + 1) % self.val_every == 0 \
                        and stopper is not None:
                    val_accuracy, _ = self.eval_model(eval_mask=data.val_mask)
                    if perf_log.update_val_perf(
                        val_perf=val_accuracy,
                        epoch=epoch,
                    ):  # perf improved, update test perf as well
                        test_accuracy, _ = self.eval_model(eval_mask=data.test_mask)
                        perf_log.update_test_perf(
                            test_perf=test_accuracy,
                            epoch=epoch,
                        )

                    # checking early stop signal
                    if stopper.step(val_accuracy, self):
                        print(f"[Epoch-{epoch}] Early stop!")
                        break

                epochs.set_description(
                    f"[Epoch-{epoch}] Loss={total_loss:.4f} | "
                    f"{' | '.join([perf_log.val_perf_summary(), perf_log.test_perf_summary()])}"
                )
                if perf_log.best_val_epoch == epoch:
                    print(f"[Epoch-{epoch}] Loss={total_loss:.4f} | " \
                        f"{' | '.join([perf_log.val_perf_summary(), perf_log.test_perf_summary()])}"
                    )
            
            # load best model for this layer (if applicable)
            if stopper is not None and stopper.best_score is not None:
                stopper.load_checkpoint(self)
            logger.info(f"Finished training\n")
        except KeyboardInterrupt:
            print(f"\n=== TRAINING INTERRUPTED AT EPOCH-{epoch}! ===\n")
            if stopper is not None and stopper.best_score is not None:
                stopper.load_checkpoint(self)
        except Exception:
            traceback.print_exc()
            raise

        # logging performance and saving results
        test_acc, _ = self.eval_model(eval_mask=data.test_mask)
        print(f"Test Accuracy : {test_acc:.6f}%\n")

        return get_perf_dict(
            train_epochs=[epoch],
            best_val_epochs=[perf_log.best_val_epoch]
        )

    @torch.no_grad()
    def eval_model(
            self,
            eval_mask: torch.BoolTensor,
            last_layer: int = -1
    ):
        self.eval()
        assert self.aug_graph is not None
        assert self.aug_graph.real_node_mask.sum() == len(eval_mask)
        assert self.augmenter is not None
        aug_graph = self.aug_graph.to(self.args.device)
        assert aug_graph.x is not None
        assert aug_graph.edge_index is not None

        accumulated_probs = []
        storable_layers = list(range(len(self.layers)))
        # build first time state memory
        states = self.forward_all_layer_first_time(
            x=aug_graph.x,
            edge_index=aug_graph.edge_index,
            edge_type=aug_graph.edge_type,
        )

        for i, layer in enumerate(self.layers):
            prob = cast(NodeSFTop2LossLayer, layer).forward_predict(
                states=states,
                layer_index=i,
                augmenter=self.augmenter,
                eval_mask=eval_mask,
                edge_index=aug_graph.edge_index,
                edge_type=aug_graph.edge_type,
            )

            # update probs
            if i in storable_layers:
                accumulated_probs.append(prob.detach())

        if not accumulated_probs:
            raise RuntimeError("No probabilities accumulated for evaluation.")
        pred = torch.stack(accumulated_probs, dim=0).sum(dim=0).argmax(dim=1)
        target = self.augmenter.y[eval_mask]
        acc = self._to_percent_accuracy(pred, target)

        return acc, accumulated_probs

class NodeSFTop2InputModel(BaseNodeGNNModel):
    def __init__(self, layer_size: list, num_classes: int, optimizer_name: str, optimizer_kwargs: dict, args: Any):
        super().__init__( num_classes, args)
        self.epochs = args.epochs
        self.test_epochs = args.test_time_steps
        self.storable_time_steps = args.storable_time_steps
        self.states = []
        self.layer_sizes = layer_size


        # construct layers
        # don't construct the first to avoid wiring from last layer
        # Citation:
        '''Note that in Alg. 3 with the top-to-input signal path, we set the top-down signal for the topmost
        hidden layer to come from the context vector (we used one hot encoding corresponding to the node
        label for training nodes, and a uniform distribution vector of the same size for other nodes), following
        Hinton (2022). Also, in updating the GNN layers with the incorporation of top-down signals, we
        support both synchronous and asynchronous updates. With asynchronous (i.e., alternating) updates,
        even-numbered layers are updated based on the activities of odd-numbered layers, and then oddnumbered layers are updated based on the new activities of even-numbered layers. With synchronous
        updates, updates to all layers occur simultaneously'''
        for i in range(1, len(layer_size) - 1):
            # top2input model
            # use previoud and next input as the input to the gnn layer, instead of the output from the previous layer
            in_channels = layer_size[i - 1] + layer_size[i + 1] 
            if self.append_label == "input":
                if i == 1:
                    in_channels = in_channels + num_classes

            self.layers.append(
                NodeSFTop2InputLayer(
                    gnn_layer=ConvLayer(gnn=self.model_type,
                                      in_channels=in_channels,
                                      out_channels=layer_size[i]),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    args=args,
                )
            )

    # forward all layers at time stamp 0
    # at time stamp 0, input from top down signal is fake (zeros)
    @torch.no_grad()
    def forward_all_layer_first_time(
      self,
      x: torch.Tensor,
      y: torch.Tensor,
      edge_index: torch.Tensor,
      edge_type: Optional[torch.Tensor] = None      
    ):
        assert x.ndim == 2 and y.ndim == 2, (x.ndim, y.ndim)
        x_new = x
        states = [x_new]

        for i, layer in enumerate(self.layers):
            gnn_in_channels = int(getattr(layer.gnn_layer, "in_channels"))
            x_next = x.new_zeros((x.shape[0], max(0, gnn_in_channels - int(x_new.shape[1]))))

            x_new = layer.forward(
                x_prev=x_new, # output from previous layer
                x_next=x_next,
                edge_index=edge_index,
                edge_type=edge_type,
            ).detach()
            states.append(x_new)
        states.append(y)

        return states  


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
        get_perf_dict = partial(load_perf_dict, start)


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

        # build first time state memory
        assert aug_graph.node_one_hot_labels is not None
        states = self.forward_all_layer_first_time(
            x=aug_graph.x,
            y=aug_graph.node_one_hot_labels,
            edge_index=aug_graph.edge_index,
            edge_type=aug_graph.edge_type,
        )

        # start training
        epochs = tqdm(range(self.epochs))
        epoch = -1
        stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
        perf_log = PerformanceManager("Accuracy")
        x = self.aug_graph.x
        alternating_update = self.args.alternating_update
        try:
            # alternating updates even and odd layers
            if alternating_update:
                update_layers_list = [
                    [(i, layer) for i, layer in enumerate(self.layers) if i % 2 == 0],
                    [(i, layer) for i, layer in enumerate(self.layers) if i % 2 == 1]
                ]
            else:  # update all layers simultaneously
                update_layers_list = [list(enumerate(self.layers))]

            for epoch in epochs:
                new_states: List[torch.Tensor | None] = [None] * len(states)
                new_states[0] = x
                total_loss = 0.0
                for layers in update_layers_list:
                    for i, layer in layers:
                        prev_state = new_states[i]
                        next_state = new_states[i + 2]
                        # avoid grabing from layer which has not run yet
                        x_prev_src = prev_state if prev_state is not None else states[i]
                        x_next_src = next_state if next_state is not None else states[i + 2]
                        assert x_prev_src is not None
                        assert x_next_src is not None
                        x_prev = x_prev_src.float()
                        x_next = x_next_src.float()

                        # forward this layer
                        node_embed, loss = cast(NodeSFTop2InputLayer, layer).forward_train(
                            x_prev=x_prev,
                            x_next=x_next,
                            augmenter=self.augmenter,
                            train_mask=data.train_mask,
                            edge_index=aug_graph.edge_index,
                            edge_type=aug_graph.edge_type,
                        )
                        new_states[i + 1] = node_embed.detach()
                        total_loss += loss
                new_states[-1] = states[-1] # save output
                states = new_states
            
                # validation + testing
                if epoch > self.args.val_from and (epoch + 1) % self.val_every == 0 \
                        and stopper is not None:
                    val_accuracy, _ = self.eval_model(eval_mask=data.val_mask)
                    if perf_log.update_val_perf(
                        val_perf=val_accuracy,
                        epoch=epoch,
                    ):  # perf improved, update test perf as well
                        test_accuracy, _ = self.eval_model(eval_mask=data.test_mask)
                        perf_log.update_test_perf(
                            test_perf=test_accuracy,
                            epoch=epoch,
                        )

                    # checking early stop signal
                    if stopper.step(val_accuracy, self):
                        print(f"[Epoch-{epoch}] Early stop!")
                        break

                epochs.set_description(
                    f"[Epoch-{epoch}] Loss={total_loss:.4f} | "
                    f"{' | '.join([perf_log.val_perf_summary(), perf_log.test_perf_summary()])}"
                )
                if perf_log.best_val_epoch == epoch:
                    print(f"[Epoch-{epoch}] Loss={total_loss:.4f} | " \
                        f"{' | '.join([perf_log.val_perf_summary(), perf_log.test_perf_summary()])}"
                    )
            
            # load best model for this layer (if applicable)
            if stopper is not None and stopper.best_score is not None:
                stopper.load_checkpoint(self)
            logger.info(f"Finished training\n")
        except KeyboardInterrupt:
            print(f"\n=== TRAINING INTERRUPTED AT EPOCH-{epoch}! ===\n")
            if stopper is not None and stopper.best_score is not None:
                stopper.load_checkpoint(self)
        except Exception:
            traceback.print_exc()
            raise

        # logging performance and saving results
        test_acc, _ = self.eval_model(eval_mask=data.test_mask)
        print(f"Test Accuracy : {test_acc:.6f}%\n")

        return get_perf_dict(
            train_epochs=[epoch],
            best_val_epochs=[perf_log.best_val_epoch]
        )

    @torch.no_grad()
    def eval_model(
            self,
            eval_mask: torch.BoolTensor,
            last_layer: int = -1
    ):
        self.eval()
        assert self.aug_graph is not None
        assert self.aug_graph.real_node_mask.sum() == len(eval_mask)
        assert self.augmenter is not None
        aug_graph = self.aug_graph.to(self.args.device)
        assert aug_graph.x is not None
        assert aug_graph.edge_index is not None
        alternating_update = self.args.alternating_update

        # alternating updates even and odd layers
        if alternating_update:
            update_layers_list = [
                [(i, layer) for i, layer in enumerate(self.layers) if i % 2 == 0],
                [(i, layer) for i, layer in enumerate(self.layers) if i % 2 == 1]
            ]
        else:  # update all layers simultaneously
            update_layers_list = [list(enumerate(self.layers))]

        accumulated_probs = []
        storable_layers = list(range(len(self.layers)))
        # build first time state memory
        assert aug_graph.node_one_hot_labels is not None
        states = self.forward_all_layer_first_time(
            x=aug_graph.x,
            y=aug_graph.node_one_hot_labels,
            edge_index=aug_graph.edge_index,
            edge_type=aug_graph.edge_type,
        )

        for epoch in range(self.test_epochs):
            x = aug_graph.x
            new_states: List[Optional[torch.Tensor]] = [None] * len(states)
            new_states[0] = x
            for layers in update_layers_list:
                for i, layer in layers:
                    prev_state = new_states[i]
                    next_state = new_states[i + 2]
                    # avoid grabing from layer which has not run yet
                    x_prev_src = prev_state if prev_state is not None else states[i]
                    x_next_src = next_state if next_state is not None else states[i + 2]
                    assert x_prev_src is not None
                    assert x_next_src is not None
                    x_prev = x_prev_src.float()
                    x_next = x_next_src.float()

                    node_embed, prob = cast(NodeSFTop2InputLayer, layer).forward_predict(
                        x_prev=x_prev,
                        x_next=x_next,
                        augmenter=self.augmenter,
                        eval_mask=eval_mask,
                        edge_index=aug_graph.edge_index,
                        edge_type=aug_graph.edge_type,
                    )
                    new_states[i + 1] = node_embed.detach()

                    # update probs
                    if epoch in self.storable_time_steps and i in storable_layers:
                        accumulated_probs.append(prob.detach())
            new_states[-1] = states[-1] # save output
            states = new_states

        if not accumulated_probs:
            raise RuntimeError("No probabilities accumulated for evaluation.")
        pred = torch.stack(accumulated_probs, dim=0).sum(dim=0).argmax(dim=1)
        target = self.augmenter.y[eval_mask]
        acc = self._to_percent_accuracy(pred, target)

        return acc, accumulated_probs