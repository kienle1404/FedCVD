"""
Federated Learning with Dual Attention Heads for Personalization.

Key design:
- Global params: all model params EXCEPT 'local_att'/'local_proj' patterns
  → aggregated via FedAvg each round
- Local params: 'local_att' and 'local_proj' tensors
  → stored per-client on the server, never aggregated

Server model invariant:
  self._model always holds aggregated global params with local positions ZEROED.
  This makes the server model a clean "global-only" model at all times.

Client training:
  1. Load global params (local positions = 0) from server
  2. Load own local params on top
  3. Train jointly with single BCELoss
  4. Zero out local positions → upload as global_params (clean, no local contamination)
  5. Upload local params separately as a plain dict {name: tensor}

Communication Protocol (per round):
  DOWNLINK  Server  → Client k:  global_model_params (local=0) + client_k_local_dict
  TRAINING  Client k:             joint SGD on all params
  UPLINK    Client k → Server:    global_params (local=0) + local_dict + client_id + n_k
  AGGREGATE Server:               FedAvg(global_params); local_params[k] = local_dict_k

Simulation note:
  In this serial simulation, downlink_package bundles ALL clients' local param dicts
  into one list for convenience. In a real distributed deployment, only
  [global_params, local_k] would be transmitted to client k — no cross-client exposure.
"""

from copy import deepcopy
import tqdm
import numpy as np
import pandas as pd
import wandb
import torch
from fedlab.utils import Aggregators, SerializationTool
from algorithm.ecg.fedavg import FedAvgServerHandler, FedAvgSerialClientTrainer
from utils.evaluation import (
    Accumulator, transfer_tensor_to_numpy, calculate_accuracy,
    get_pred_label, calculate_multilabel_metrics
)

# Naming patterns that identify local (personalized) parameters
_LOCAL_PATTERNS = ('local_att', 'local_proj')


def _is_local(name: str) -> bool:
    """Return True if this parameter belongs to the local (personalized) branch."""
    return any(p in name for p in _LOCAL_PATTERNS)


# ============================================================
# Server Handler
# ============================================================

class FedDualAttServerHandler(FedAvgServerHandler):
    """
    Server handler for FedDualAtt.

    Maintains two disjoint sets of parameters:
      self._model                   — global params only; local positions always = 0
      self.local_attention_params   — list of per-client local param dicts
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loaders,
        criterion: torch.nn.Module,
        output_path: str,
        evaluator,
        communication_round: int,
        num_clients: int = 4,
        sample_ratio: float = 1.0,
        device: torch.device | None = None,
        logger=None,
    ):
        super().__init__(
            model, test_loaders, criterion, output_path, evaluator,
            communication_round, num_clients, sample_ratio, device, logger
        )

        # Initialize per-client local params from the model's initial local weights
        initial_local = {n: p.data.clone()
                         for n, p in model.named_parameters() if _is_local(n)}
        self.local_attention_params = [deepcopy(initial_local) for _ in range(num_clients)]

        # Establish server model invariant: local positions = 0
        self._zero_local_params()

    # ----------------------------------------------------------------
    # Invariant helper
    # ----------------------------------------------------------------

    def _zero_local_params(self):
        """Zero all local param positions in self._model."""
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if _is_local(name):
                    param.zero_()

    # ----------------------------------------------------------------
    # Communication
    # ----------------------------------------------------------------

    @property
    def downlink_package(self):
        """
        Payload sent to clients each round.

        Returns:
            list: [global_serialized, local_dict_0, ..., local_dict_{N-1}]
              global_serialized  — serialized self._model (local positions = 0)
              local_dict_k       — {param_name: tensor} for client k

        Simulation note: all N local dicts are bundled together.
        In production, only [global_serialized, local_dict_k] is sent to client k.
        """
        global_serialized = self.model_parameters  # invariant: local = 0
        return [global_serialized] + [deepcopy(p) for p in self.local_attention_params]

    def global_update(self, buffer):
        """
        Aggregate client updates.

        Each buffer entry: [global_params, local_dict, client_id, num_samples]
          global_params — serialized model with local positions = 0 (enforced by client)
          local_dict    — {param_name: tensor} of client's trained local params

        FedAvg on global_params: averaging zeros for local positions keeps them 0,
        so the server model invariant is automatically preserved.
        """
        global_params_list = [ele[0] for ele in buffer]
        local_dicts        = [ele[1] for ele in buffer]
        client_ids         = [ele[2] for ele in buffer]
        weights            = [ele[3] for ele in buffer]

        # Aggregate global params — local positions average to 0 (invariant preserved)
        global_aggregated = Aggregators.fedavg_aggregate(global_params_list, weights)
        SerializationTool.deserialize_model(self._model, global_aggregated)

        # Store each client's local params directly (no aggregation)
        for idx, client_id in enumerate(client_ids):
            self.local_attention_params[client_id] = local_dicts[idx]

    # ----------------------------------------------------------------
    # Evaluation (corrected: per-client local params loaded before test)
    # ----------------------------------------------------------------

    def local_test(self):
        """
        Evaluate on each client's test set using THAT client's trained local params.

        For each client k:
          1. Load client k's local params into model
          2. Evaluate on client k's test loader
          3. Restore local positions to 0 (preserve server invariant)
        """
        self._model.eval()
        l_metric_dict = {}
        eval_desc = "Local Test Loss {:.8f}  |  Acc:{:.2f}"

        for idx, item in enumerate(self.test_loaders):
            # Load this client's local params
            if self.local_attention_params[idx]:
                self._model.load_state_dict(self.local_attention_params[idx], strict=False)

            metric = Accumulator(3)
            eval_bar = tqdm.tqdm(
                initial=0, leave=True, total=len(item),
                desc=eval_desc.format(0, 0), position=0
            )
            pred_score_list, pred_label_list, true_label_list = [], [], []

            for data, label in item:
                data, label = data.to(self._device), label.to(self._device)
                with torch.no_grad():
                    pred_score    = self._model(data)
                    pred_score_np = transfer_tensor_to_numpy(pred_score)
                    pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                    true_label_np = transfer_tensor_to_numpy(label)
                    pred_score_list.append(pred_score_np)
                    pred_label_list.append(pred_label_np)
                    true_label_list.append(true_label_np)
                    loss = self.criterion(pred_score, label)
                    metric.add(
                        float(loss) * len(label),
                        calculate_accuracy(pred_label_np, true_label_np),
                        len(label)
                    )
                eval_bar.desc = eval_desc.format(metric[0] / metric[2], metric[1] / metric[2])
                eval_bar.update(1)
            eval_bar.close()

            # Restore invariant before moving to the next client
            self._zero_local_params()

            all_pred_score_np = np.concatenate(pred_score_list, axis=0)
            all_pred_label_np = np.concatenate(pred_label_list, axis=0)
            all_true_label_np = np.concatenate(true_label_list, axis=0)

            pd.DataFrame(all_pred_score_np).to_csv(
                self.output_path + f"server/local_test/local_pred_score_{idx}.csv",
                index=False, encoding="utf-8"
            )
            pd.DataFrame(all_pred_label_np).to_csv(
                self.output_path + f"server/local_test/local_pred_label_{idx}.csv",
                index=False, encoding="utf-8"
            )
            pd.DataFrame(all_true_label_np).to_csv(
                self.output_path + f"server/local_test/local_true_label_{idx}.csv",
                index=False, encoding="utf-8"
            )

            metric_dict = calculate_multilabel_metrics(
                all_pred_score_np, all_pred_label_np, all_true_label_np
            )
            metric_dict["loss"] = metric[0] / metric[2]
            l_metric_dict[str(idx)] = metric_dict
            self._LOGGER.info(
                f"Client {idx+1} Local Test | Loss: {metric[0]/metric[2]:.6f} "
                f"| Acc: {metric[1]/metric[2]:.4f}"
            )
            wandb.log(
                {
                    f"server_client{idx+1}_local_test_loss":     metric[0] / metric[2],
                    f"server_client{idx+1}_local_test_acc":      metric[1] / metric[2],
                    f"server_client{idx+1}_local_test_micro_f1": metric_dict["micro_f1"],
                    f"server_client{idx+1}_local_test_mAP":
                        float(np.average(metric_dict["average_precision_score"])),
                },
                step=self.round,
            )

        self.evaluator.add_dict("local_test", self.round, l_metric_dict)

    def save_model(self, path):
        """Save global model (local=0) and all per-client local params."""
        torch.save(
            {
                "global_model":          self._model.state_dict(),  # local positions = 0
                "local_attention_params": self.local_attention_params,
                "round":                  self.round,
            },
            path,
        )


# ============================================================
# Client Trainer
# ============================================================

class FedDualAttSerialClientTrainer(FedAvgSerialClientTrainer):
    """
    Client trainer for FedDualAtt.

    Training procedure per client per round:
      1. Load global params (local positions = 0) from server
      2. Load own local params on top → model has correct global + local state
      3. Joint SGD on ALL params (single BCELoss)
      4. Extract updated local params as plain dict
      5. Zero local positions in model → serialize as clean global_params upload
      6. Restore local params in model for evaluation/saving
    """

    def local_process(self, payload, id_list):
        """
        Process training for each client in id_list.

        Args:
            payload: [global_serialized, local_dict_0, ..., local_dict_{N-1}]
            id_list: list of client indices to train this round
        """
        global_params = payload[0]   # serialized global model (local = 0)

        for idx in id_list:
            # --- Step 1: Load global model (local positions = 0) ---
            self.set_model(global_params)

            # --- Step 2: Overwrite local positions with client's own params ---
            local_dict = payload[idx + 1]   # dict {param_name: tensor}
            if local_dict:
                self._model.load_state_dict(local_dict, strict=False)

            # --- Step 3: Train (global + local params jointly) ---
            pack = None
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, idx)
                self.local_test(idx, epoch)
                self.global_test(idx, epoch)

            # --- Step 4: Extract updated local params ---
            local_updated = {
                n: p.data.clone()
                for n, p in self._model.named_parameters()
                if _is_local(n)
            }

            # --- Step 5: Zero local positions → clean global upload ---
            with torch.no_grad():
                for name, param in self._model.named_parameters():
                    if _is_local(name):
                        param.zero_()
            global_updated = self.model_parameters   # local positions = 0

            # --- Step 6: Restore local params for eval/save ---
            if local_updated:
                self._model.load_state_dict(local_updated, strict=False)

            # Upload: [global_params (local=0), local_dict, client_id, num_samples]
            self.cache.append([global_updated, local_updated, idx, pack[1]])

            torch.save(
                {"model": self._model.state_dict()},
                self.output_path + f"client{idx+1}/model.pth",
            )
