import traceback
from functools import partial
from timeit import default_timer as timer
from typing import List, Optional, Any, cast

import torch
from torch_geometric.data import Data
from tqdm import tqdm


from models.shared.utils import load_perf_dict, PerformanceManager
from models.link.common import BaseLinkGNNModel
from layers.link.common import LayerNormalization
from layers.link.link_ff import LinkForwardLayer, LinkForwardTopDownLayer
from layers.conv_layer import ConvLayer
from utils import ResultManager, logger
from utils.train_utils import EarlyStopping
from utils.eval_utils import eval_link_prediction


class LinkForwardModel(BaseLinkGNNModel):
    def __init__(
        self,
        layer_size: List[int],
        optimizer_name: str,
        optimizer_kwargs: dict,
        args: Any,
    ):
        super().__init__(args=args)

        # construct layers
        for i in range(len(layer_size) - 1):
            self.layers.append(LayerNormalization())
            self.layers.append(
                LinkForwardLayer(
                    gnn_layer=ConvLayer(gnn=self.model_type,
                                      in_channels=layer_size[i],
                                      out_channels=layer_size[i + 1]),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    args=args,
                )
            )
    

    def forward(
            self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_type: Optional[torch.Tensor] = None
    ):
        return super().forward(x, edge_index, edge_type)

    def forward_train(
        self,
        train_data: Data,
        val_data: Data,
        test_data: Data,
        result_manager: ResultManager,
        run_i: int,
    ):
        # send data
        train_data = train_data.clone().to(self.device)
        val_data = val_data.clone().to(self.device)
        test_data = test_data.clone().to(self.device)

        # set up training
        self.train()
        start = timer()
        train_epochs, best_val_epochs = [], []
        get_perf_dict = partial(load_perf_dict, start, train_epochs, best_val_epochs)

        for i,layer in enumerate(self.layers):
            # normalization
            if isinstance(layer, LayerNormalization):
                with torch.no_grad():
                    train_data.x = layer(train_data.x).detach()
                continue

            epochs, epoch = tqdm(range(self.args.epochs)), -1
            stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
            perf_log = PerformanceManager("AUC")

            try:
                for epoch in epochs:
                    _,logits = cast(LinkForwardLayer, layer).forward_train(
                        train_data, self.theta
                    )
                    seperation = logits[0] - logits[1]
                    
                    # validation + testing
                    if epoch > self.args.val_from and (epoch + 1) % self.val_every == 0 \
                            and stopper is not None:
                        val_accuracy, _ = self.eval_model(
                            data=val_data,
                            last_layer=i
                        )
                        if perf_log.update_val_perf(
                            val_perf=val_accuracy,
                            epoch=epoch,
                        ):  # perf improved, update test perf as well
                            test_accuracy, _ = self.eval_model(
                                data=test_data,
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
                assert train_data.x is not None
                assert train_data.edge_index is not None
                train_data.x = layer(train_data.x, train_data.edge_index).detach()

            # logging performance and saving results
            train_epochs.append(epoch)
            best_val_epochs.append(perf_log.best_val_epoch)
            test_acc, _ = self.eval_model(
                                data=test_data,
                                last_layer=i
                            )
            print(f"[Layer-{i}] Test Accuracy : {test_acc:.2f}%\n")
            result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_acc), num_layers=(i + 1) // 2) # skip normalization layers
            

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            data: Data,
            last_layer: int = -1
    ):
        self.eval()
        assert data.x is not None
        assert data.edge_index is not None
        last_layer = self._resolve_last_eval_layer(last_layer, num_layers=len(self.layers))


        accumulate_from = 0 if last_layer >= 1 else 0
        accumulated_probs_list: list[torch.Tensor] = []
        x = data.x
       
        for i, layer in enumerate(self.layers):
            if i <= last_layer:
                if isinstance(layer, LayerNormalization):
                    x, prob = cast(LayerNormalization, layer).forward_predict(
                        x,
                        data.edge_index,
                        data.edge_label_index,
                        self.theta
                    )
                else:
                    x, prob = cast(LinkForwardLayer, layer).forward_predict(
                        x,
                        data.edge_index,
                        data.edge_label_index,
                        self.theta
                    )

                if i >= accumulate_from:
                    accumulated_probs_list.append(prob)

        if not accumulated_probs_list:
            raise RuntimeError("No probabilities accumulated for evaluation.")
        score = torch.stack(accumulated_probs_list).mean(dim=0)
        result = eval_link_prediction(
            data.edge_label,
            score
        )
        return result['rocauc'], accumulated_probs_list


class LinkForwardTopDownModel(BaseLinkGNNModel):
    def __init__(
        self,
        layer_size: List[int],
        optimizer_name: str,
        optimizer_kwargs: dict,
        args: Any,
    ):
        super().__init__(args=args)
        self.alternating_update = args.alternating_update
        self.epochs = args.epochs
        self.test_epochs = args.test_time_steps
        self.storable_time_steps = args.storable_time_steps
        self.states = []
        self.layer_sizes = layer_size

        # construct layers
        for i in range(1, len(layer_size) - 1):
            self.layers.append(
                LinkForwardTopDownLayer(
                    gnn_layer=ConvLayer(gnn=self.model_type,
                                      in_channels=layer_size[i - 1] + layer_size[i + 1],
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
      edge_index: torch.Tensor,
      edge_type: Optional[torch.Tensor] = None      
    ):
        assert x.ndim == 2, x.ndim
        x_new = x
        states: List[torch.Tensor | None] = [x_new]

        for i, layer in enumerate(self.layers):
            gnn_in_channels = int(getattr(layer.gnn_layer, "in_channels"))
            x_next = x.new_zeros((x.shape[0], max(0, gnn_in_channels - int(x_new.shape[1]))))

            x_new = cast(LinkForwardTopDownLayer, layer).forward(
                x_prev=x_new, # output from previous layer
                x_next=x_next,
                edge_index=edge_index,
                edge_type=edge_type,
            ).detach()
            states.append(x_new)
        states.append(None)

        return states  

    def forward_train(
        self,
        train_data: Data,
        val_data: Data,
        test_data: Data,
        result_manager: ResultManager,
        run_i: int,
    ):
        # send data
        train_data = train_data.clone().to(self.device)
        val_data = val_data.clone().to(self.device)
        test_data = test_data.clone().to(self.device)

        # set up training
        self.train()
        start = timer()
        train_epochs, best_val_epochs = [], []
        get_perf_dict = partial(load_perf_dict, start_time=start)

        # build first time state memory
        assert train_data.x is not None
        assert train_data.y is not None
        assert isinstance(train_data.y, torch.Tensor)
        assert train_data.edge_index is not None
        states = self.forward_all_layer_first_time(
            x=train_data.x,
            edge_index=train_data.edge_index,
        )


        # start training
        accmulate_pos_goodness = 0.0
        accmulate_neg_goodness = 0.0
        epochs = tqdm(range(self.epochs))
        epoch = -1
        stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
        perf_log = PerformanceManager("AUC")
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
                x = train_data.x
                new_states: List[torch.Tensor | None] = [None] * len(states)
                new_states[0] = x
                total_pos_goodness = 0.0
                total_neg_goodness = 0.0
                for layers in update_layers_list:
                    for i, layer in layers:
                        prev_state = new_states[i]
                        next_state = new_states[i + 2]
                        # avoid grabing from layer which has not run yet
                        x_prev_src = prev_state if prev_state is not None else states[i]
                        x_next_src = next_state if next_state is not None else states[i + 2]
                        assert x_prev_src is not None
                        x_prev = x_prev_src.float()
                        x_next = x_next_src.float() if x_next_src is not None else None

                        # forward this layer
                        node_embed,_, loss = cast(LinkForwardTopDownLayer, layer).forward_train(
                            x_prev=x_prev,
                            x_next=x_next,
                            data=train_data,
                        )
                        new_states[i + 1] = node_embed.detach()
                        accmulate_pos_goodness += loss[0]
                        accmulate_neg_goodness += loss[1]
                        total_pos_goodness += loss[0]
                        total_neg_goodness += loss[1]
                new_states[-1] = states[-1] # save output
                states = new_states
                # validation + testing
                if epoch > self.args.val_from and (epoch + 1) % self.val_every == 0 \
                        and stopper is not None:
                    val_accuracy, _ = self.eval_model(
                        data=val_data,
                        last_layer=-1
                    )
                    if perf_log.update_val_perf(
                        val_perf=val_accuracy,
                        epoch=epoch,
                    ):  # perf improved, update test perf as well
                        test_accuracy, _ = self.eval_model(
                            data=test_data,
                            last_layer=-1
                        )
                        perf_log.update_test_perf(
                            test_perf=test_accuracy,
                            epoch=epoch,
                        )

                    # checking early stop signal
                    if stopper.step(val_accuracy, self):
                        print(f"[Epoch-{epoch}] Early stop!")
                        break

                msg = f"[T-{epoch}] Pos={total_pos_goodness:.4f}, Neg={total_neg_goodness:.4f} | " \
                      f"TrainSep: {total_pos_goodness - total_neg_goodness:.2f} | " \
                      f"{' | '.join([perf_log.val_perf_summary(), perf_log.test_perf_summary()])}"
                epochs.set_description(msg)
                if perf_log.best_val_epoch == epoch:
                    print(msg)
            
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
        test_acc, _ = self.eval_model(data=test_data, last_layer=-1)
        print(f"Test Accuracy : {test_acc:.2f}%\n")

        return get_perf_dict(
            train_epochs=[epoch],
            best_val_epochs=[perf_log.best_val_epoch]
        )

    @torch.no_grad()
    def eval_model(
            self,
            data: Data,
            last_layer: int = -1
    ):
        self.eval()
        data = data.to(self.args.device)
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
        assert data.x is not None
        assert data.edge_index is not None
        states = self.forward_all_layer_first_time(
            x=data.x,
            edge_index=data.edge_index,
        )

        for epoch in range(self.test_epochs):
            x = data.x
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
                    x_prev = x_prev_src.float()
                    x_next = x_next_src.float() if x_next_src is not None else None

                    node_embed, out = cast(LinkForwardTopDownLayer, layer).forward_predict(
                        x_prev=x_prev,
                        x_next=x_next,
                        edge_index=data.edge_index,
                        label_index=data.edge_label_index
                    )
                    new_states[i + 1] = node_embed.detach()

                    # update probs
                    if epoch in self.storable_time_steps and i in storable_layers:
                        accumulated_probs.append(out.sigmoid().detach())
            new_states[-1] = states[-1] # save output
            states = new_states

        if not accumulated_probs:
            raise RuntimeError("No probabilities accumulated for evaluation.")
        score = torch.stack(accumulated_probs).mean(dim=0)
        result = eval_link_prediction(
            data.edge_label,
            score
        )
        return result['rocauc'], score