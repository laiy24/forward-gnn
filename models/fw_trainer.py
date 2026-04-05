
from timeit import default_timer as timer

import torch
from tqdm import tqdm


from utils.log_utils import logger
from utils.eval_utils import eval_node_classification, eval_link_prediction
from models.node.common import BaseNodeGNNModel
from models.node.node_sf import NodeSingleForwardModel, NodeSFTop2InputModel, NodeSFTop2LossModel
from models.node.node_ff import NodeVirtualNodeFFModel, NodeLabelAppendFFModel

class FWNodeClassificationTrainer:
    def __init__(self, model, data, device, result_manager, run_i, lr, epochs, patience, args):
        self.model: BaseNodeGNNModel = model
        self.data = data
        self.device = device
        self.result_manager = result_manager
        self.run_i = run_i

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.args = args

    def train(self):
        model, data = self.model, self.data
        args = self.args
        
        model = model.to(args.device)
        return model.forward_train(data, self.result_manager, self.run_i)
    
    def test(self):
        model, data = self.model, self.data.to(self.device)
        if isinstance(model, (NodeSFTop2InputModel,
                              NodeSFTop2LossModel,
                              NodeSingleForwardModel)):
            acc, _ = model.eval_model(eval_mask=data.test_mask)
        elif isinstance(model, (NodeVirtualNodeFFModel, NodeLabelAppendFFModel)):
            acc, _ = model.eval_model(data, train_mask=data.train_mask | data.val_mask, eval_mask=data.test_mask)
        else:
             raise NotImplementedError(f"Model type {type(model)} not supported for evaluation.")
        print(f"Test Accuracy: {acc:.4f}%")
        return acc

    def train_test(self):
        result = self.train()
        test_acc = self.test()

        return {
            "test_perf": test_acc,
            "train_time": result["train_time"],
            "train_epochs": result["train_epochs"],
            'best_val_epochs': result["best_val_epochs"],
        }


class FWLinkPredictionTrainer:
    def __init__(self, model, train_data, val_data, test_data, result_manager, run_i, device, lr, epochs, patience, args):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.result_manager = result_manager
        self.run_i = run_i
        self.device = device

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.args = args

    @staticmethod
    def link_predict(z, edge_label_index):
        pred = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)
        assert len(pred) == edge_label_index.shape[1], (len(pred), edge_label_index.shape[1])
        return pred

    def train(self):
        model, train_data, val_data, test_data = self.model.to(self.device), self.train_data, self.val_data, self.test_data
        result = model.forward_train(train_data, val_data, test_data, self.result_manager, self.run_i)

        return result

    def test(self):
        model, test_data = self.model, self.test_data.to(self.device)
        acc, _ = model.eval_model(test_data)
        return acc

    def train_test(self):
        result = self.train()
        acc = self.test()

        return {
            "test_perf": acc,
            "train_time": result["train_time"],
            "train_epochs": result["train_epochs"],
            'best_val_epochs': result["best_val_epochs"],
        }

