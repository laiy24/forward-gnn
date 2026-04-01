
from timeit import default_timer as timer

import torch
from tqdm import tqdm

from utils.train_utils import EarlyStopping
from utils.log_utils import logger
from utils.eval_utils import eval_node_classification, eval_link_prediction


class NodeClassificationTrainer:
    def __init__(self, model, data, device, lr, epochs, patience, args):
        self.model = model
        self.data = data
        self.device = device

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.args = args

    def train(self):
        model, data = self.model, self.data

        """Training"""
        start = timer()
        stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
        epoch = -1

        data = data.clone().to(self.device)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        epoch_tqdm = tqdm(range(self.epochs))
        for epoch in epoch_tqdm:
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])

            loss.backward()
            optimizer.step()

            """Validation"""
            if stopper is not None and (epoch + 1) % self.args.val_every == 0:
                val_metric = 'acc'

                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)

                pred = val_out.argmax(dim=1)
                val_perf_dict = eval_node_classification(data.y, pred, data.val_mask)
                val_perf = val_perf_dict[val_metric]

                epoch_tqdm.set_description(
                    f'Epoch: {epoch:03d}, Train Loss={loss.item():.4f}, Val Acc={val_perf:.4f}'
                )

                if stopper.step(val_perf, model):
                    print(f"[Epoch-{epoch}] Early stop!")
                    break

        if stopper is not None and stopper.best_score is not None:
            stopper.load_checkpoint(model)

        train_time = timer() - start
        logger.info("Finished training")

        return train_time, epoch

    def test(self):
        model, data = self.model, self.data.to(self.device)
        model.eval()

        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_perf_dict = eval_node_classification(data.y, pred, data.test_mask)
        test_acc = test_perf_dict['acc']
        print(f"Test Accuracy: {test_acc:.6f}")

        return test_acc

    def train_test(self):
        train_time, train_epochs = self.train()
        test_acc = self.test()

        return {
            "test_perf": test_acc,
            "train_time": train_time,
            "train_epochs": train_epochs,
        }


class LinkPredictionTrainer:
    def __init__(self, model, train_data, val_data, test_data, device, lr, epochs, patience, args):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
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
        model, train_data, val_data, test_data = self.model, self.train_data, self.val_data, self.test_data
        model.train()

        """Training"""
        start = timer()
        stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
        epoch = -1

        train_data = train_data.clone().to(self.device)
        val_data = val_data.clone().to(self.device)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        epoch_tqdm = tqdm(range(self.epochs))
        for epoch in epoch_tqdm:
            model.train()

            optimizer.zero_grad()

            z = model(train_data.x, train_data.edge_index)
            out = self.link_predict(z, train_data.edge_label_index)
            loss = criterion(out, train_data.edge_label)

            loss.backward()
            optimizer.step()

            """Validation"""
            if stopper is not None and (epoch + 1) % self.args.val_every == 0:
                val_metric = 'rocauc'

                with torch.no_grad():
                    model.eval()
                    val_z = model(val_data.x, val_data.edge_index)
                    val_out = self.link_predict(val_z, val_data.edge_label_index)

                val_perf_dict = eval_link_prediction(
                    y_true=val_data.edge_label,
                    y_score=val_out.sigmoid(),
                    metrics=val_metric
                )
                val_perf = val_perf_dict[val_metric]
                epoch_tqdm.set_description(f'Epoch: {epoch:03d}, Train Loss={loss.item():.4f}, Val AUC={val_perf:.4f}')

                if stopper.step(val_perf, model):
                    print(f"[Epoch-{epoch}] Early stop!")
                    break

        if stopper is not None and stopper.best_score is not None:
            stopper.load_checkpoint(model)

        train_time = timer() - start
        logger.info("Finished training")

        return train_time, epoch

    def test(self):
        model, test_data = self.model, self.test_data.to(self.device)
        model.eval()

        z = model(test_data.x, test_data.edge_index)
        out = self.link_predict(z, test_data.edge_label_index)

        test_perf_dict = eval_link_prediction(
            y_true=test_data.edge_label,
            y_score=out.sigmoid(),
        )
        test_rocauc = test_perf_dict['rocauc']
        print(f"Test AUC: {test_rocauc:.6f}")

        return test_rocauc

    def train_test(self):
        train_time, train_epochs = self.train()
        test_rocauc = self.test()

        return {
            "test_perf": test_rocauc,
            "train_time": train_time,
            "train_epochs": train_epochs,
        }

