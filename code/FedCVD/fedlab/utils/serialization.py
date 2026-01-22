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
import torch


def _process_keys(keys: List[str], include_keys: Union[str, List[str], None], filter_keys: Union[str, List[str], None]) -> tuple[List[str], List[str]]:
    """
    Helper function to determine the final list of keys to keep based on include/filter rules.
    """
    # 1. Normalize inputs
    if include_keys is None:
        include_keys = []
    elif isinstance(include_keys, str):
        include_keys = [include_keys]

    if filter_keys is None:
        filter_keys = []
    elif isinstance(filter_keys, str):
        filter_keys = [filter_keys]

    final_keys = []
    skipped_keys = []

    # 2. Iterate and apply rules
    for key in keys:
        keep = True

        # Rule A: Check inclusion (if include_keys is non-empty)
        # If include_keys is provided, a key must start with at least one include_key
        if include_keys and not any(i in key for i in include_keys):
            keep = False

        # Rule B: Check exclusion (filter_keys always takes precedence over inclusion)
        # A key must NOT start with any filter_key
        if keep and any(f in key for f in filter_keys):
            keep = False

        if keep:
            final_keys.append(key)
        else:
            skipped_keys.append(key)

    return final_keys, skipped_keys

class SerializationTool(object):
    @staticmethod
    def serialize_model(
        model: Module,
        filter_keys: Union[str, List[str], None] = None,
        include_keys: Union[str, List[str], None] = None, # NEW PARAMETER
        cpu: bool = True
    ) -> OrderedDict:
        """
        Serializes the model's parameters into a new OrderedDict, with options
        for including and filtering layers and moving to CPU.

        Args:
            model (torch.nn.Module): The model to serialize.
            filter_keys (Union[str, List[str], None]): Keys to skip serialization (Exclusion rule).
            include_keys (Union[str, List[str], None]): Keys to include (Inclusion rule). If provided,
                                                       only keys matching this and not matching filter_keys will be serialized.
            cpu (bool): If True, all tensors in the serialized dictionary will be moved to the CPU.

        Returns:
            OrderedDict: A deep copy of the filtered and serialized model state_dict.
        """
        print(f"Starting to serialize model parameters to OrderedDict.")

        all_keys = list(model.state_dict().keys())
        keys_to_serialize, skipped_keys = _process_keys(all_keys, include_keys, filter_keys)

        serialized_model_dict = OrderedDict()

        for key in keys_to_serialize:
            param = model.state_dict()[key]
            param_copy = param.data.clone()

            if cpu:
                param_copy = param_copy.cpu()

            serialized_model_dict[key] = param_copy

        if skipped_keys:
            print(f"Skipping layers with name: [{', '.join(skipped_keys)}]")
        print(f"Successfully serialized parameters to OrderedDict. Total parameters serialized: {len(serialized_model_dict)}")
        return serialized_model_dict

    @staticmethod
    def deserialize_model(
        model: Module,
        serialized_model_dict: OrderedDict,
        mode: Literal["copy", "add", "sub"] = "copy",
        filter_keys: Union[str, List[str], None] = None,
        include_keys: Union[str, List[str], None] = None # NEW PARAMETER
    ) -> None:
        """
        Deserializes and assigns parameters from an OrderedDict to the model.

        Args:
            model (torch.nn.Module): The model to assign the parameters to.
            serialized_model_dict (OrderedDict): The serialized model parameters.
            mode (str): The mode of deserialization. "copy" copies the parameters.
            filter_keys (Union[str, List[str], None]): Keys to skip deserialization (Exclusion rule).
            include_keys (Union[str, List[str], None]): Keys to include (Inclusion rule).
        """
        print("Starting to deserialize parameters from OrderedDict.")

        # 1. Determine which keys from the serialized dictionary should be used
        all_serialized_keys = list(serialized_model_dict.keys())
        keys_to_deserialize, skipped_by_filter = _process_keys(all_serialized_keys, include_keys, filter_keys)

        # Create the final state_dict to load/use
        target_state_dict = OrderedDict({key: serialized_model_dict[key] for key in keys_to_deserialize})

        if mode == "copy":
            try:
                mismatched_keys = model.load_state_dict(target_state_dict, strict=False)

                # Log keys skipped by user's filter
                if skipped_by_filter:
                    print(f"Skipping layers (user filter/include): [{', '.join(skipped_by_filter)}]")

                # Log keys missed because they were not serialized
                if mismatched_keys.missing_keys:
                    print(f"The following layers were not found in the serialized dictionary: [{','.join(mismatched_keys.missing_keys)}]")

                # Log unexpected keys
                if mismatched_keys.unexpected_keys:
                    print(f"The following layers were unexpected in the model and will be ignored: [{','.join(mismatched_keys.unexpected_keys)}]")

                print(f"Parameters have been successfully copied to the model. Total layers updated: {len(target_state_dict)}")
            except RuntimeError as e:
                raise e
        else:
            # For 'add' and 'sub' modes, iterate over the model's state_dict keys
            for key in model.state_dict().keys():
                if key in target_state_dict:
                    current_param = model.state_dict()[key].data
                    target_param = target_state_dict[key].to(current_param.device).data

                    if mode == "add":
                        current_param.add_(target_param)
                    elif mode == "sub":
                        current_param.sub_(target_param)
                    else:
                        raise ValueError(f"Invalid deserialize mode {mode}, require \"copy\", \"add\" or \"sub\"!")
                # Note: We rely on the loguru logger warnings from 'copy' mode or manual logging for skipped/missing keys

            if skipped_by_filter:
                print(f"Skipping layers (user filter/include): [{', '.join(skipped_by_filter)}]")
            print(f"Successfully deserialized parameters from OrderedDict with mode '{mode}'.")
