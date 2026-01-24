"""
Federated Learning with Dual Attention Heads for Personalization.

This module implements a personalized FL algorithm that splits transformer attention
into global (shared/aggregated) and local (personalized) components.

Key Features:
- Global attention heads (4 heads): Aggregated via FedAvg across all clients
- Local attention heads (4 heads): Personalized per client, not aggregated
- All other components (ResNet, FFN, LayerNorm, FC): Shared/aggregated

Communication Protocol:
    Server → Client i: [global_params, local_params_0, ..., local_params_{N-1}]
    Client i trains: Both global and local parameters jointly with SGD
    Client i → Server: [global_params_updated, local_params_i_updated, client_id, num_samples]
    Server aggregates: FedAvg(global_params), DirectUpdate(local_params_i)
"""

from copy import deepcopy
from fedlab.utils import Aggregators
from fedlab.utils import SerializationTool
from algorithm.ecg.fedavg import FedAvgServerHandler, FedAvgSerialClientTrainer

import torch


class FedDualAttServerHandler(FedAvgServerHandler):
    """
    Server handler for Federated Dual Attention.

    Manages dual parameter sets:
    - Global parameters: Stored in self._model, aggregated via FedAvg
    - Local attention parameters: Stored per-client in self.local_attention_params
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
        super(FedDualAttServerHandler, self).__init__(
            model, test_loaders, criterion, output_path, evaluator,
            communication_round, num_clients, sample_ratio, device, logger
        )

        # Initialize local attention parameters for each client
        # Each client gets a copy of the initial local attention weights
        self.local_attention_params = []
        for _ in range(num_clients):
            local_params = self._extract_local_params(model)
            self.local_attention_params.append(local_params)

    @property
    def downlink_package(self):
        """
        Package to send to clients.

        Returns:
            List: [global_params, local_params_0, local_params_1, ..., local_params_{N-1}]
                - global_params: Serialized global model (includes all parameters)
                - local_params_i: Serialized local attention parameters for client i
        """
        # Serialize global model (contains all parameters including initial local attention)
        global_serialized = self.model_parameters

        # Serialize each client's local attention parameters
        local_serialized = []
        for local_params in self.local_attention_params:
            serialized = self._serialize_local_params(local_params)
            local_serialized.append(serialized)

        # Return: [global, local_0, local_1, ..., local_{N-1}]
        return [global_serialized, *local_serialized]

    def global_update(self, buffer):
        """
        Aggregate client updates.

        Args:
            buffer: List of client updates, each containing:
                [global_params, local_params, client_id, num_samples]
        """
        # Extract components from buffer
        global_params_list = [ele[0] for ele in buffer]  # Global parameters
        local_params_list = [ele[1] for ele in buffer]   # Local attention parameters
        client_ids = [ele[2] for ele in buffer]          # Client IDs
        weights = [ele[3] for ele in buffer]             # Sample counts

        # Aggregate global parameters using FedAvg (weighted by sample count)
        global_aggregated = Aggregators.fedavg_aggregate(global_params_list, weights)
        SerializationTool.deserialize_model(self._model, global_aggregated)

        # Update local attention parameters (no aggregation, direct assignment)
        for idx, client_id in enumerate(client_ids):
            self.local_attention_params[client_id] = self._deserialize_local_params(
                local_params_list[idx]
            )

    def _extract_local_params(self, model):
        """
        Extract only local attention parameters from model.

        Args:
            model: PyTorch model

        Returns:
            dict: Dictionary of local attention parameters
                Keys: Parameter names containing 'local_att'
                Values: Cloned parameter data
        """
        local_state = {}
        for name, param in model.named_parameters():
            if 'local_att' in name:
                local_state[name] = param.data.clone()
        return local_state

    def _serialize_local_params(self, local_params):
        """
        Serialize local attention parameters for transmission.

        This is a workaround for FedLab's SerializationTool which expects
        a full model. We create a temp model, load only local params, then serialize.

        Args:
            local_params: Dictionary of local attention parameters

        Returns:
            Serialized parameters (format: FedLab SerializationTool)
        """
        # Create temporary model
        temp_model = deepcopy(self._model)

        # Load only local parameters (strict=False allows partial loading)
        temp_model.load_state_dict(local_params, strict=False)

        # Serialize the temp model
        return SerializationTool.serialize_model(temp_model)

    def _deserialize_local_params(self, serialized_local):
        """
        Deserialize local attention parameters from client update.

        Args:
            serialized_local: Serialized local parameters from client

        Returns:
            dict: Dictionary of local attention parameters
        """
        # Create temporary model to deserialize into
        temp_model = deepcopy(self._model)
        SerializationTool.deserialize_model(temp_model, serialized_local)

        # Extract only local attention parameters
        local_state = {}
        for name, param in temp_model.named_parameters():
            if 'local_att' in name:
                local_state[name] = param.data.clone()

        return local_state

    def save_model(self, path):
        """
        Save both global model and all local attention parameters.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'global_model': self._model.state_dict(),
            'local_attention_params': self.local_attention_params,
            'round': self.round
        }, path)


class FedDualAttSerialClientTrainer(FedAvgSerialClientTrainer):
    """
    Client trainer for Federated Dual Attention.

    Handles:
    - Loading both global and client-specific local parameters
    - Training full model (global + local) jointly
    - Extracting and uploading separate parameter sets
    """

    def local_process(self, payload, id_list):
        """
        Process training for selected clients.

        Args:
            payload: Downlink package from server
                [global_params, local_params_0, local_params_1, ..., local_params_{N-1}]
            id_list: List of client IDs to train
        """
        global_params = payload[0]  # Global model parameters

        for idx in id_list:
            # Load global parameters into model
            self.set_model(global_params)

            # Load client's local attention parameters
            local_params_serialized = payload[idx + 1]
            self._load_local_params(local_params_serialized)

            # Train for specified epochs (both global and local jointly)
            pack = None
            for epoch in range(self.max_epoch):
                pack = self.train(epoch, idx)
                self.local_test(idx, epoch)
                self.global_test(idx, epoch)

            # Extract updated parameters
            global_updated = self.model_parameters
            local_updated = self._extract_and_serialize_local_params()

            # Package for upload: [global_params, local_params, client_id, num_samples]
            self.cache.append([global_updated, local_updated, idx, pack[1]])

            # Save checkpoint
            torch.save({
                "model": self._model.state_dict()
            }, self.output_path + "client" + str(idx + 1) + "/model.pth")

    def _load_local_params(self, serialized_local):
        """
        Load local attention parameters into current model.

        Args:
            serialized_local: Serialized local parameters from server
        """
        # Create temporary model to deserialize into
        temp_model = deepcopy(self._model)
        SerializationTool.deserialize_model(temp_model, serialized_local)

        # Extract local attention parameters
        local_state = {}
        for name, param in temp_model.named_parameters():
            if 'local_att' in name:
                local_state[name] = param.data

        # Load local parameters into current model (strict=False for partial loading)
        self._model.load_state_dict(local_state, strict=False)

    def _extract_and_serialize_local_params(self):
        """
        Extract and serialize local attention parameters for upload.

        Returns:
            Serialized local attention parameters
        """
        # Create temporary model
        temp_model = deepcopy(self._model)

        # Extract local attention parameters from current model
        local_state = {}
        for name, param in self._model.named_parameters():
            if 'local_att' in name:
                local_state[name] = param.data.clone()

        # Load local parameters into temp model and serialize
        temp_model.load_state_dict(local_state, strict=False)
        return SerializationTool.serialize_model(temp_model)
