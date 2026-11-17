# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

from transformers import PreTrainedConfig
from vllm.config import ModelConfig
from vllm.models.flint.gguf_nvfp4 import _map_name

from ..gguf_files import GGUFModelFiles
from ..weight_utils import get_gguf_tensor_names, split_stacked_experts
from .base import BaseGGUFWeightsAdapter, GGUFWeight


class FlintGGUFAdapter(BaseGGUFWeightsAdapter):
    """Map Flint GGUF weights, including sinks and stacked experts."""

    @classmethod
    def matches(cls, config: PreTrainedConfig) -> bool:
        return config.model_type == "flint"

    def build_name_map(
        self, files: GGUFModelFiles, model_config: ModelConfig
    ) -> dict[str, str]:
        result = {}
        for name in get_gguf_tensor_names(files.backbone):
            try:
                mapped, stacked = _map_name(name)
            except ValueError:
                base, suffix = name.rsplit(".", 1)
                mapped, stacked = _map_name(base)
                mapped += "." + suffix
            if stacked:
                mapped = mapped.format(0)
            result[name] = mapped
        return result

    def transform_weights(
        self, weights: Iterable[GGUFWeight], model_config: ModelConfig
    ) -> Iterable[GGUFWeight]:
        yield from split_stacked_experts(weights)
