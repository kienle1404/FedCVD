
from collections import OrderedDict
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


def _zeros_like_ordered_dict(ordered_dict: OrderedDict) -> OrderedDict:
    """Create an OrderedDict with zero tensors matching the structure of input."""
    return OrderedDict({k: torch.zeros_like(v) for k, v in ordered_dict.items()})


def _add_ordered_dicts(a: OrderedDict, b: OrderedDict) -> OrderedDict:
    """Element-wise addition of two OrderedDicts."""
    return OrderedDict({k: a[k] + b[k] for k in a.keys()})


def _sub_ordered_dicts(a: OrderedDict, b: OrderedDict) -> OrderedDict:
    """Element-wise subtraction of two OrderedDicts."""
    return OrderedDict({k: a[k] - b[k] for k in a.keys()})


def _scale_ordered_dict(d: OrderedDict, scale: float) -> OrderedDict:
    """Scale all tensors in an OrderedDict by a scalar."""
    return OrderedDict({k: scale * v for k, v in d.items()})


def _flatten_ordered_dict(d: OrderedDict) -> torch.Tensor:
    """Flatten an OrderedDict of tensors to a single 1D tensor."""
    return torch.cat([v.flatten() for v in d.values()])


class ScaffoldServerHandler(FedAvgServerHandler):
    def __init__(
        self,
        lr: float,
        model: torch.nn.Module,
        test_loaders,
        criterion: torch.nn.Module,
        output_path: str,
        evaluator,
        communication_round: int,
        num_clients: int = 0,
        sample_ratio: float = 1,
        device: torch.device | None = None,
        logger: Logger = None,
    ):
        super(ScaffoldServerHandler, self).__init__(
            model=model,
            test_loaders=test_loaders,
            criterion=criterion,
            output_path=output_path,
            evaluator=evaluator,
            communication_round=communication_round,
            num_clients=num_clients,
            sample_ratio=sample_ratio,
            device=device,
            logger=logger
        )
        self.lr = lr
        # Initialize global_c as OrderedDict of zeros
        self.global_c = _zeros_like_ordered_dict(self.model_parameters)

    @property
    def downlink_package(self):
        return [self.model_parameters, self.global_c]

    def global_update(self, buffer):
        # unpack - dys and dcs are lists of OrderedDicts
        dys = [ele[0] for ele in buffer]
        dcs = [ele[1] for ele in buffer]

        # Aggregate using fedavg (returns OrderedDict)
        dx = Aggregators.fedavg_aggregate(dys)
        dc = Aggregators.fedavg_aggregate(dcs)

        # Apply server learning rate: scaled_dx = lr * dx
        scaled_dx = _scale_ordered_dict(dx, self.lr)

        # Update model: model = model + lr * dx
        # Filter out num_batches_tracked (Long type) which can't have Float added to it
        SerializationTool.deserialize_model(self._model, scaled_dx, mode="add", filter_keys="num_batches_tracked")

        # Update global_c: global_c += (num_clients_in_round / total_clients) * dc
        scale = 1.0 * len(dcs) / self.num_clients
        scaled_dc = _scale_ordered_dict(dc, scale)
        self.global_c = _add_ordered_dicts(self.global_c, scaled_dc)


class ScaffoldSerialClientTrainer(FedAvgSerialClientTrainer):
    def __init__(
            self,
            model,
            num_clients,
            train_loaders,
            test_loaders,
            lr: float,
            criterion: torch.nn.Module,
            max_epoch: int,
            output_path: str,
            evaluators,
            optimizer_name: str = "SGD",
            device: torch.device | None = None,
            logger=None,
            personal=False
    ):
        super(ScaffoldSerialClientTrainer, self).__init__(
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
        self.cs = [None for _ in range(self.num_clients)]

    def local_process(self, payload, id_list):
        pack = None
        model_parameters = payload[0]  # OrderedDict
        global_c = payload[1]  # OrderedDict
        for idx in id_list:
            self.set_model(model_parameters)
            frz_model = deepcopy(model_parameters)  # Deep copy to preserve original
            if self.cs[idx] is None:
                self.cs[idx] = _zeros_like_ordered_dict(model_parameters)
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, global_c, idx)
                # self.local_test(idx, epoch)
                # self.global_test(idx, epoch)

            # Compute dy = model_parameters_after - frz_model (OrderedDict)
            dy = _sub_ordered_dicts(self.model_parameters, frz_model)

            # Compute dc = -1/(K*eta) * dy - global_c
            scale = -1.0 / (self.max_epoch * len(self.train_loaders[idx]) * self.lr)
            scaled_dy = _scale_ordered_dict(dy, scale)
            dc = _sub_ordered_dicts(scaled_dy, global_c)

            # Update client control variate: cs[idx] += dc
            self.cs[idx] = _add_ordered_dicts(self.cs[idx], dc)

            self.cache.append([dy, dc])
            torch.save(
                {
                    "model": self._model.state_dict()
                },
                self.output_path + "client" + str(idx + 1) + "/model.pth"
            )

    def train(self, epoch, global_c, idx):
        self._model.train()

        # Flatten control variates to tensors for gradient correction
        global_c_flat = _flatten_ordered_dict(global_c)
        cs_flat = _flatten_ordered_dict(self.cs[idx])

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

            # Get model gradients as flattened tensor
            grad = self.model_grads

            # Move control variates to same device as gradients
            cs_flat_device = cs_flat.to(grad.device)
            global_c_flat_device = global_c_flat.to(grad.device)

            # SCAFFOLD gradient correction: grad = grad - c_i + c
            grad = grad - cs_flat_device + global_c_flat_device

            # Apply corrected gradients back to model parameters
            index = 0
            parameters = self._model.parameters()
            for p in self._model.state_dict().values():
                if p.grad is None:
                    layer_size = p.numel()
                else:
                    parameter = next(parameters)
                    layer_size = parameter.data.numel()
                    shape = parameter.grad.shape
                    parameter.grad.data[:] = grad[index:index + layer_size].view(shape)[:]
                index += layer_size

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

        return [self.model_parameters]
