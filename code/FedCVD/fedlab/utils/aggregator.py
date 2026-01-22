# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, List, Union
from collections import OrderedDict
from torch.nn import Module
# from matching.pfnm_communication import layer_group_descent as pdm_iterative_layer_group_descent
# from matching.pfnm_communication import build_init as pdm_build_init
# from matching.gaus_marginal_matching import match_local_atoms
# from matching.pfnm import layer_group_descent as pdm_multilayer_group_descent
import torch


class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"

        first_client_dict = serialized_params_list[0]
        aggregated_dict = OrderedDict()
        for key in first_client_dict.keys():
            # Robustness Check (Optional, but highly recommended for FL)
            if not all(key in client_dict for client_dict in serialized_params_list):
                raise KeyError(f"Key '{key}' not found in all client parameter dictionaries.")
            param_tensors = [client_dict[key].to(torch.float32) for client_dict in serialized_params_list]
            aggregated_tensor = torch.sum(torch.stack(param_tensors, dim=-1) * weights, dim=-1)
            original_dtype = serialized_params_list[0][key].dtype
            aggregated_dict[key] = aggregated_tensor.to(original_dtype)

        return aggregated_dict

    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator
        
        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters
