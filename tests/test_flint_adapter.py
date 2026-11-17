# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import torch
from vllm.config import ModelConfig
from vllm.models.flint.configs import FlintConfig

from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator
from vllm_gguf_plugin.weights_adapter import get_weights_adapter


def test_flint_gguf_preserves_sinks_and_splits_quantized_experts(tmp_path):
    """Flint's sink parameter and packed expert bytes reach their HF names."""
    config = FlintConfig(num_hidden_layers=1, layer_types=["full_attention"])
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, dtype="float16"
    )
    path = tmp_path / "weights.gguf"
    writer = gguf.GGUFWriter(str(path), "flint")
    sinks = np.arange(32, dtype=np.float32)
    experts = gguf.quants.quantize(
        np.arange(2 * 4 * 32, dtype=np.float32).reshape(2, 4, 32),
        gguf.GGMLQuantizationType.Q8_0,
    )
    writer.add_tensor("blk.0.attn_sinks.weight", sinks)
    writer.add_tensor(
        "blk.0.ffn_up_exps.weight",
        experts,
        raw_dtype=gguf.GGMLQuantizationType.Q8_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    adapter = get_weights_adapter(model_config.hf_config)
    mapping = adapter.build_name_map(
        GGUFModelFiles(backbone=(str(path),)), model_config
    )
    tensors = dict(
        adapter.transform_weights(
            gguf_quant_weights_iterator(path, mapping), model_config
        )
    )
    torch.testing.assert_close(
        tensors["model.layers.0.self_attn.softmax_offset"], torch.from_numpy(sinks)
    )
    for expert in range(2):
        torch.testing.assert_close(
            tensors[f"model.layers.0.mlp.experts.{expert}.up_proj.weight"],
            torch.from_numpy(experts[expert]),
            atol=0,
            rtol=0,
        )
