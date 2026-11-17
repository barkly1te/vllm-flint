# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, quantize

from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf


@pytest.mark.parametrize("num_tokens", [2, 128])
@torch.inference_mode()
def test_moe_padding_does_not_read_negative_experts(num_tokens):
    """Padding with expert -1 contributes zero, even with NaN activations."""
    generator = np.random.default_rng(0)
    quant_type = GGMLQuantizationType.Q8_0
    w13 = torch.from_numpy(
        quantize(
            generator.normal(0, 0.01, (2, 512, 256)).astype(np.float32), quant_type
        )
    ).cuda()
    w2 = torch.from_numpy(
        quantize(
            generator.normal(0, 0.01, (2, 256, 256)).astype(np.float32), quant_type
        )
    ).cuda()
    x = torch.full((num_tokens, 256), float("nan"), dtype=torch.half, device="cuda")
    x[0] = 1
    ids = torch.full((num_tokens, 2), -1, dtype=torch.int32, device="cuda")
    ids[0] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    weights = torch.zeros((num_tokens, 2), device="cuda")
    weights[0] = 0.5
    output = _fused_moe_gguf(x, w13, w2, weights, ids, quant_type, quant_type, "silu")
    reference = _fused_moe_gguf(
        x[:1], w13, w2, weights[:1], ids[:1], quant_type, quant_type, "silu"
    )
    torch.testing.assert_close(output[:1], reference)
    torch.testing.assert_close(output[1:], torch.zeros_like(output[1:]), atol=0, rtol=0)
