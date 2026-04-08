import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from datetime import datetime
from typing import Optional

import numpy as np
from torch_geometric.nn import GraphSAGE, GCN, GAT

import settings
from datasets import load_node_classification_data, load_link_prediction_data
from utils.log_utils import log_stdout, logger
from utils.train_utils import setup_cuda, set_seed, SeedManager, ResultManager
from models.bp_trainer import BPLinkPredictionTrainer, BPNodeClassificationTrainer
from models.fw_trainer import FWLinkPredictionTrainer, FWNodeClassificationTrainer
from models.node.node_sf import NodeSingleForwardModel, NodeSFTop2InputModel, NodeSFTop2LossModel
from models.node.node_ff import NodeVirtualNodeFFModel, NodeLabelAppendFFModel
from models.link.link_ff import LinkForwardModel,LinkForwardTopDownModel

def main(args):
    seed_manager = SeedManager(args.seed)
    result_manager = ResultManager(result_file_prefix=f"fw-results", args=args, seed_manager=seed_manager)

    for run_i in range(args.num_runs):
        if not args.overwrite_result and result_manager.load_run_result(run_i) is not None:
            logger.info(f"Skipping run-{run_i}: already evaluated.")
            continue

        seed_manager.set_run_i(run_i)
        set_seed(seed_manager.get_run_seed(), deterministic="gat" not in args.model.lower())  # initialize seed for each run
        print(f"\nStarting run-{run_i} of {args.model} on {args.dataset} (seed={seed_manager.get_run_seed()})\n")

        # load data
        train_data = val_data = test_data = None
        if args.task == "node-class":
            data = load_node_classification_data(args, split_i=run_i)
        elif args.task == "link-pred":
            train_data, val_data, test_data, data = load_link_prediction_data(args, split_i=run_i)
        else:
            raise ValueError(f"Invalid task: {args.task}")

        # Build model/trainer with explicit task -> training_type logic
        if args.task == "node-class":
            if args.training_type == "backprop":
                model = build_bp_model(data, args)
                trainer = BPNodeClassificationTrainer(
                    model, data, args.device, args.lr, args.epochs, args.patience, args
                )
            elif args.training_type == "forward":
                model = build_node_classification_model(model_type=args.model,
                                                        n_layers=args.num_layers,
                                                        hidden_size=args.num_hidden,
                                                        loss_fn_name=args.loss_fn_name,
                                                        lr=args.lr,
                                                        data=data,
                                                        args=args)
                trainer = FWNodeClassificationTrainer(
                    model, data, args.device, result_manager, run_i, args.lr, args.epochs, args.patience, args
                )
            else:
                raise ValueError(f"Invalid training type: {args.training_type}")
        elif args.task == "link-pred":
            if args.training_type == "backprop":
                model = build_bp_model(data, args)
                trainer = BPLinkPredictionTrainer(
                    model, train_data, val_data, test_data, args.device, args.lr, args.epochs, args.patience, args
                )
            elif args.training_type == "forward":
                model = build_link_prediction_model(model_type=args.model,
                                                    n_layers=args.num_layers,
                                                    hidden_size=args.num_hidden,
                                                    loss_fn_name=args.loss_fn_name,
                                                    lr=args.lr,
                                                    data=data,
                                                    args=args)
                trainer = FWLinkPredictionTrainer(
                    model, train_data, val_data, test_data, result_manager, run_i, args.device, args.lr, args.epochs, args.patience, args
                )
            else:
                raise ValueError(f"Invalid training type: {args.training_type}")
        else:
            raise ValueError(f"Invalid task: {args.task}")

        # print result for the run and save it
        result = trainer.train_test()
        train_epochs = result.get("train_epochs", [])
        if not isinstance(train_epochs, list):
            train_epochs = [train_epochs]

        best_val_epochs = result.get("best_val_epochs", [])
        if not isinstance(best_val_epochs, list):
            best_val_epochs = [best_val_epochs]

        perf_dict = {
            'perf': result["test_perf"],
            'train_time': result["train_time"],
            'memory_usage': result.get("memory_usage"),
            'train_epochs': train_epochs,
            'best_val_epochs': best_val_epochs,
        }
        result_manager.save_run_result(run_i, perf_dict)

    # final printing of aggregated performance across runs
    perfs = []
    for run_i in range(args.num_runs):
        run_result = result_manager.load_run_result(run_i)
        if run_result is None:
            logger.warning(f"Missing result for run-{run_i}; skipping in final aggregation.")
            continue
        perfs.append(run_result["perf"])

    if len(perfs) == 0:
        raise RuntimeError("No valid run results found to aggregate performance.")

    print(f"\nTest Performance ({len(perfs)} runs): {np.mean(perfs):.6f}%±{np.std(perfs):.4f}")

def build_bp_model(data, args):
    logger.info(f"Backpropagation {args.task} model {args.model} is used")
    assert args.num_layers >= 1, args.num_layers
    if args.task == "node-class":
        out_channels = data.num_classes
    else:
        out_channels = None  # will be set to args.num_hidden

    if args.model == "SAGE":
        model = GraphSAGE(
            in_channels=data.num_features,
            hidden_channels=args.num_hidden,
            num_layers=args.num_layers,
            out_channels=out_channels,
            dropout=0.0,
            act="relu"
        )
    elif args.model == "GCN":
        model = GCN(
            in_channels=data.num_features,
            hidden_channels=args.num_hidden,
            num_layers=args.num_layers,
            out_channels=out_channels,
            dropout=0.0,
            act="relu"
        )
    elif args.model == "GAT":
        model = GAT(
            in_channels=data.num_features,
            hidden_channels=args.num_hidden,
            heads=4,
            num_layers=args.num_layers,
            out_channels=out_channels,
            dropout=0.0,
            act="relu"
        )
    else:
        raise ValueError(f"Invalid model: {args.model}")

    print(model)
    return model

def build_node_classification_model(model_type, n_layers, hidden_size, loss_fn_name, lr, data, args):
    assert n_layers >= 1, n_layers

    if args.forward_type == "SF":  # single forward
        if args.topdown_model is not None and args.topdown_model.lower() != "none":
            if args.topdown_model == "top2input":
                logger.info(f"node sf top2input model {model_type} is used")
                layer_sizes = [data.num_features] + [hidden_size] * n_layers + [data.num_classes]

                model = NodeSFTop2InputModel(
                    layer_size=layer_sizes,
                    optimizer_name='Adam',
                    optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                    num_classes=data.num_classes,
                    args=args,
                )
            elif args.topdown_model == "top2loss":
                logger.info(f"node sf top2loss model {model_type} is used")
                layer_sizes = [data.num_features] + [hidden_size] * n_layers

                model = NodeSFTop2LossModel(
                    layer_size=layer_sizes,
                    optimizer_name='Adam',
                    optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                    num_classes=data.num_classes,
                    args=args,
                )
            else:
                raise ValueError(f"Invalid model: {model_type}")
        else:
            logger.info(f"node sf single forward model {model_type} is used")
            layer_sizes = [data.num_features] + [hidden_size] * n_layers

            model = NodeSingleForwardModel(
                layer_size=layer_sizes,
                optimizer_name='Adam',
                optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                num_classes=data.num_classes,
                args=args,
            )
    else:
        layer_sizes = [data.num_features] + [hidden_size] * n_layers

        if args.append_label is not None and args.append_label.lower() != "none":
            logger.info(f"node ff label appending model {model_type} is used")
            model = NodeLabelAppendFFModel(
                layer_size=layer_sizes,
                optimizer_name='Adam',
                optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                num_classes=data.num_classes,
                args=args,
            )
        elif args.virtual_node is not None and args.virtual_node:
            logger.info(f"node ff virtual node label appending model {model_type} is used")
            model = NodeVirtualNodeFFModel(
                layer_size=layer_sizes,
                optimizer_name='Adam',
                optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                num_classes=data.num_classes,
                args=args,
            )
        else:
            raise ValueError(f"Invalid model: {model_type}")

    print(model)
    return model


def build_link_prediction_model(model_type, n_layers, hidden_size, loss_fn_name, lr, data, args):
    assert n_layers >= 1, n_layers

    if args.topdown_model is not None and args.topdown_model.lower() != "none":
        logger.info(f"link pred top-down forward model {model_type} is used")
        layer_sizes = [data.num_features] + [hidden_size] * n_layers + [0]

        model = LinkForwardTopDownModel(
            layer_size=layer_sizes,
            optimizer_name='Adam',
            optimizer_kwargs={'lr': lr},
            args=args,
        )
    else:  # forward forward or forward learning
        logger.info(f"link pred forward learning model {model_type} is used")
        layer_sizes = [data.num_features] + [hidden_size] * n_layers
        model = LinkForwardModel(
            layer_size=layer_sizes,
            optimizer_name='Adam',
            optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
            args=args,
        )

    print(model)
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-type', type=str, default="forward", choices=["forward", "backprop"],
                        help="training type")
    parser.add_argument('--task', type=str, default="node-class", choices=["link-pred", "node-class"],
                        help="graph learning task")
    parser.add_argument('--model', type=str,
                        help="model type")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of gnn layers")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden channels")
    parser.add_argument('--dataset', type=str,
                        help="dataset name")
    parser.add_argument("--val-from", type=int, default=0,
                        help="epoch to start validation")
    parser.add_argument('--num-runs', type=int, default=1,
                        help="number of total runs. each run uses a different random seed.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=100,
                        help="seed for exp")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--val-every", type=int, default=10,
                        help="number of epochs between validation")
    parser.add_argument('--patience', type=int, default=30,
                        help='patience for early stopping (set this to negative value to not use early stopping)')
    parser.add_argument('--exp-setting', type=str, default="default",
                        help="experiment setting")
    parser.add_argument('--overwrite-result', action='store_true')
    parser.set_defaults(overwrite_result=False)

    # only needed for forward training, but we parse them for all models for simplicity
    parser.add_argument('--use-cache', type=bool, default=False,
                        help="whether to cache propagated messages for faster training (only applicable for some GNNs such as GraphSAGE and GCN)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="softmax temperature")                       
    parser.add_argument("--grad-max-norm", type=float, default=1.0,
                        help="max norm of the gradients")
    parser.add_argument('--aug-edge-direction', type=str, default="bidirection",
                        choices=["unidirection", "bidirection"])
    parser.add_argument("--test-time-steps", type=int, default=10)
    parser.add_argument("--storable-time-steps", type=str, default=None)
    parser.add_argument('--alternating-update', action='store_true')
    parser.set_defaults(alternating_update=False)
    parser.add_argument('--loss-fn-name', type=str, default=None,
                        help="loss function (only used for forward training)")
    parser.add_argument("--ff-theta", type=float, default=2.0,
                        help="theta for forward-forward training")
    parser.add_argument('--append-label', type=str, default="none",
                        choices=["none", "all", "input"])
    parser.add_argument('--virtual-node', type=bool, default=False,
                        help="whether to use virtual node")
    parser.add_argument('--topdown-model', type=str, choices=['none','top2input', 'top2loss'], default='none',
                        help="which top-down forward learning model to use (only applicable for node classification task)")
    parser.add_argument("--num-negs", type=int, default=2,
                        help="number of negative samples per postive")
    parser.add_argument('--forward-type', type=str, default="FF", choices=["FF", "FL", "SF"],
                        help="forward forward, forward-looking(link only) or single forward")


    args = parser.parse_args()
    return args


def populate_args(args):
    if args.training_type == "forward" and args.loss_fn_name is None:
        args.loss_fn_name = "forwardforward_loss_fn"

    if args.append_label.lower() == "none":
        args.append_label = None

    setup_cuda(args)

    args.results_dir = settings.RESULTS_ROOT / args.exp_setting / args.dataset / args.task
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.exp_datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
    stdout_path = args.results_dir / "stdout"
    stdout_path.mkdir(parents=True, exist_ok=True)
    log_stdout(stdout_path / f"stdout-{args.task}-{args.model}-{args.dataset}-{args.exp_datetime}.txt")

    storable_time_steps: Optional[str] = args.storable_time_steps
    if storable_time_steps is None:
        storable_time_steps = list(range(args.test_time_steps))[2:]
    else:
        try:
            storable_time_steps = list(range(int(storable_time_steps)))
        except Exception:
            storable_time_steps = list(map(int, storable_time_steps.strip().split(",")))
    assert all(t < args.test_time_steps for t in storable_time_steps), (args.test_time_steps, storable_time_steps)
    args.storable_time_steps = storable_time_steps

    from pprint import pformat
    print(f"args:\n{pformat(args.__dict__)}")
    return args


if __name__ == '__main__':
    args = parse_args()
    args = populate_args(args)
    main(args)
