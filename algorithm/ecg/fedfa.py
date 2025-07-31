
from copy import deepcopy
from algorithm.ecg.fedavg import FedAvgServerHandler, FedAvgSerialClientTrainer
from utils.evaluation import calculate_accuracy, calculate_multilabel_metrics, get_pred_label, transfer_tensor_to_numpy
from utils.evaluation import Accumulator
from fedlab.utils.logger import Logger
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators

import torch
import tqdm
import pandas as pd
import numpy as np


class FedFaServerHandler(FedAvgServerHandler):
    def __init__(
            self,
            model: torch.nn.Module,
            test_loaders,
            criterion: torch.nn.Module,
            output_path: str,
            evaluator,
            communication_round: int,
            alpha: float,
            beta: float,
            gamma: float,
            server_lr: float,
            momentum_round: int = 1,
            num_clients: int = 0,
            sample_ratio: float = 1,
            device: torch.device | None = None,
            logger: Logger = None,
    ):
        super(FedFaServerHandler, self).__init__(model, test_loaders, criterion, output_path, evaluator,
                                                  communication_round, num_clients, sample_ratio, device, logger)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.server_lr = server_lr
        self.momentum_round = momentum_round
        self.momentum = torch.zeros_like(self.model_parameters)

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        acc_list = torch.tensor([ele[1] for ele in buffer])
        freq_list = torch.tensor([ele[2] for ele in buffer])
        for i in range(len(acc_list)):
            if acc_list[i] < 1e-6:
                acc_list[i] = 1e-6
        acc = acc_list / torch.sum(acc_list)
        freq = freq_list / torch.sum(freq_list)
        acc_inf = -torch.log2(acc)
        freq_inf = -torch.log2(1 - freq)
        acc_inf = acc_inf / torch.sum(acc_inf)
        freq_inf = freq_inf / torch.sum(freq_inf)
        weights = self.alpha * acc_inf + self.beta * freq_inf
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        delta = serialized_parameters - self.model_parameters
        self.momentum = self.gamma * self.momentum + (1 - self.gamma) * delta
        if self.communication_round % self.momentum_round == 0:
            serialized_parameters -= self.server_lr * self.momentum
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedFaSerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            model,
            num_clients,
            train_loaders,
            test_loaders,
            lr: float,
            gamma: float,
            criterion: torch.nn.Module,
            max_epoch: int,
            output_path: str,
            evaluators,
            optimizer_name: str = "SGD",
            device: torch.device | None = None,
            logger=None,
            personal=False
    ):
        super(FedFaSerialClientTrainer, self).__init__(
            model=model,
            num_clients=num_clients,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            lr=lr,
            criterion=criterion,
            max_epoch=max_epoch,
            output_path=output_path,
            evaluators=evaluators,
            optimizer_name=optimizer_name,
            device=device,
            logger=logger,
            personal=personal
        )
        self.freq = torch.zeros(self.num_clients)
        self.gamma = gamma
        self.momentum = [torch.zeros_like(self.model_parameters) for _ in range(self.num_clients)]

    def local_process(self, payload, id_list):
        pack = None
        model_parameters = payload[0]
        for idx in id_list:
            self.set_model(model_parameters)
            self.freq[idx] += 1
            for epoch in range(self.max_epoch):
                pack = self.train(model_parameters, epoch, idx)
                self.local_test(idx, epoch)
                self.global_test(idx, epoch)
            self.cache.append(pack)
            torch.save(
                {
                    "model": self._model.state_dict()
                },
                self.output_path + "client" + str(idx + 1) + "/model.pth"
            )

    def train(self, model_parameters, epoch, idx):
        self._model.train()

        metric = Accumulator(3)
        pred_score_list = []
        pred_label_list = []
        true_label_list = []
        train_desc = "Epoch {:2d}: train Loss {:.8f}  |  Acc:{:.2f}"
        train_bar = tqdm.tqdm(initial=0, leave=True, total=len(self.train_loaders[idx]),
                              desc=train_desc.format(epoch, 0, 0), position=0)
        for data, label in self.train_loaders[idx]:
            data, label = data.to(self._device), label.to(self._device)

            pred_score = self._model(data)
            with torch.no_grad():
                pred_score_np = transfer_tensor_to_numpy(pred_score)
                pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                true_label_np = transfer_tensor_to_numpy(label)
                pred_score_list.append(pred_score_np)
                pred_label_list.append(pred_label_np)
                true_label_list.append(true_label_np)

            loss = self.criterion(pred_score, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric.add(
                float(loss) * len(label), calculate_accuracy(pred_label_np, true_label_np), len(label)
            )
            train_bar.desc = train_desc.format(epoch, metric[0] / metric[2], metric[1] / metric[2])
            train_bar.update(1)
        train_bar.close()
        all_pred_score_np = np.concatenate(pred_score_list, axis=0)
        all_pred_label_np = np.concatenate(pred_label_list, axis=0)
        all_true_label_np = np.concatenate(true_label_list, axis=0)
        df = pd.DataFrame(all_pred_score_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/train/local_pred_score.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_pred_label_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/train/local_pred_label.csv", index=False, encoding="utf-8"
        )
        df = pd.DataFrame(all_true_label_np)
        df.to_csv(
            self.output_path + "client" + str(idx + 1) + "/train/local_true_label.csv", index=False, encoding="utf-8"
        )
        metric_dict = calculate_multilabel_metrics(all_pred_score_np, all_pred_label_np, all_true_label_np)
        metric_dict["loss"] = metric[0] / metric[2]
        self.evaluators[idx].add_dict("train", self.current_round, epoch, metric_dict)
        self._LOGGER[idx].info(f"Epoch {epoch} | Train Loss: {metric[0] / metric[2]} | Train Acc: {metric[1] / metric[2]}")

        delta = self.model_parameters - model_parameters
        self.momentum[idx] = self.gamma * self.momentum[idx] + self.lr * delta
        self.set_model(self.model_parameters - self.momentum[idx])
        return [self.model_parameters, metric_dict["micro_f1"], self.freq[idx]]
        # return [self.model_parameters, metric[1] / metric[2], self.freq[idx]]
