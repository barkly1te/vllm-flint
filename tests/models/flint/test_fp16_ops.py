# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.flint.ops.norm import RMSNorm, silu_and_mul


@pytest.mark.parametrize("with_residual", [False, True])
def test_fp16_norm_matches_reference_before_residual_rounding(with_residual):
    """Normalize the FP32 sum, retaining a separately rounded FP16 residual."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, 128, generator=generator).half()
    residual = torch.randn(4, 128, generator=generator).half()
    norm = RMSNorm(128, eps=1e-6).half()
    values = x.double() + residual.double() if with_residual else x.double()
    expected = F.rms_norm(values, (128,), norm.weight.double(), 1e-6).half()
    if with_residual:
        output, residual_output = norm(x, residual)
        torch.testing.assert_close(residual_output, values.half(), atol=0, rtol=0)
    else:
        output = norm(x)
    torch.testing.assert_close(output, expected)


def test_fp16_swiglu_rounds_after_multiplication():
    x = torch.linspace(-10, 10, 512).reshape(4, 128).half()
    gate, up = x.double().chunk(2, dim=-1)
    expected = (F.silu(gate) * up).half()
    torch.testing.assert_close(silu_and_mul(x), expected)
