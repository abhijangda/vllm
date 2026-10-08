# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.model_executor.layers.strassen import (
    PREFILL_TOKEN_SIZES,
    _workspace,
    pad_mlp_weight,
    prepare_strassen_weight,
)


@pytest.fixture(scope="module")
def projection():
    if not envs.VLLM_STRASSEN_LIBRARY_PATH:
        pytest.skip("Set VLLM_STRASSEN_LIBRARY_PATH to the compiled library")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 is required")
    torch.manual_seed(0)
    weight = torch.randn(59136, 8192, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = pad_mlp_weight(weight, gate_up=True)
    packed, presums = prepare_strassen_weight(weight)
    workspace = _workspace(weight.device, packed.shape[1], packed.shape[0])
    return weight, packed, presums, workspace


def assert_projection_close(actual, expected):
    # BF16 Winograd pre-sums and post-sums round differently from a single GEMM.
    error = (actual.float() - expected.float()).square().mean()
    reference = expected.float().square().mean()
    assert torch.sqrt(error / reference).item() < 0.02
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("ffn_size", [1024, 1025, 29568])
def test_strassen_pads_each_gate_up_branch_to_1024(projection, ffn_size):
    weight = (
        torch.arange(2 * ffn_size, device="cuda", dtype=torch.float32)
        .to(torch.bfloat16)[:, None]
        .expand(-1, 128)
        .contiguous()
    )
    packed, presums = prepare_strassen_weight(weight)
    padded_ffn = ((ffn_size + 1023) // 1024) * 1024
    assert packed.shape == (128, 2 * padded_ffn)
    quadrants = packed.view(2, 2, 64, padded_ffn)
    for k_half in range(2):
        for branch in range(2):
            expected = weight[
                branch * ffn_size : (branch + 1) * ffn_size,
                k_half * 64 : (k_half + 1) * 64,
            ].t()
            torch.testing.assert_close(
                quadrants[k_half, branch, :, :ffn_size], expected, rtol=0, atol=0
            )
    assert torch.count_nonzero(quadrants[..., ffn_size:]).item() == 0
    assert (
        torch.count_nonzero(presums.view(4, 64, padded_ffn)[..., ffn_size:]).item() == 0
    )


@pytest.mark.parametrize("swizzle", [1, 2, 4])
@pytest.mark.parametrize("raster", [0, 1])
@pytest.mark.parametrize("tokens", PREFILL_TOKEN_SIZES)
def test_strassen_retains_padded_gate_up_projection(
    projection, swizzle, raster, tokens
):
    weight, packed, presums, workspace = projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    expected = F.linear(x, weight)
    with patch(
        "vllm.model_executor.layers.strassen.F.linear",
        side_effect=AssertionError(
            "Supported sizes must execute Strassen, not fallback"
        ),
    ):
        actual = torch.ops.vllm.strassen_bf16_linear(
            x, weight, packed, presums, workspace, swizzle, raster
        )
    assert actual.shape == (tokens, 59392)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert_projection_close(actual, expected)


@pytest.mark.parametrize("tokens", PREFILL_TOKEN_SIZES)
def test_strassen_graph_replay_uses_changed_activations(projection, tokens):
    weight, packed, presums, workspace = projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.vllm.strassen_bf16_linear(
                x, weight, packed, presums, workspace, 4, 0
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.vllm.strassen_bf16_linear(
            x, weight, packed, presums, workspace, 4, 0
        )
    for scale in (1.0, -2.0, 0.5):
        x.mul_(scale)
        graph.replay()
        assert_projection_close(actual, F.linear(x, weight))


def test_strassen_torch_compile_matches_eager(projection):
    weight, packed, presums, workspace = projection
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    compiled = torch.compile(
        torch.ops.vllm.strassen_bf16_linear, backend="eager", fullgraph=True
    )
    actual = compiled(x, weight, packed, presums, workspace, 4, 0)
    assert_projection_close(actual, F.linear(x, weight))


def test_unsafe_skip_packing_reinterprets_row_major_input_in_graph(
    projection, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING", True)
    weight, packed, presums, workspace = projection
    x = torch.randn(1024, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops.vllm.strassen_bf16_linear(x, weight, packed, presums, workspace, 2, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.vllm.strassen_bf16_linear(
            x, weight, packed, presums, workspace, 2, 0
        )
    x.mul_(0.5)
    graph.replay()
    wrong_x = x.reshape(2, 2, 512, 4096).permute(0, 2, 1, 3).reshape_as(x)
    assert_projection_close(actual, F.linear(wrong_x, weight))
    assert not torch.allclose(actual, F.linear(x, weight), rtol=0.02, atol=0.01)


@pytest.mark.parametrize("tokens", [1, 8, 16, 512, 3072])
def test_strassen_other_token_counts_use_padded_weights(projection, tokens):
    weight, packed, presums, workspace = projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    actual = torch.ops.vllm.strassen_bf16_linear(
        x, weight, packed, presums, workspace, 1, 0
    )
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=0, atol=0)


@pytest.mark.parametrize("ffn", [1024, 1025, 29568])
def test_padded_down_projection_preserves_mlp_output(ffn):
    torch.manual_seed(0)
    gate_up = torch.randn(2 * ffn, 8)
    down = torch.randn(8, ffn)
    padded_gate_up = pad_mlp_weight(gate_up, gate_up=True)
    padded_down = pad_mlp_weight(down, gate_up=False)
    padded_ffn = ((ffn + 1023) // 1024) * 1024
    assert padded_gate_up.shape == (2 * padded_ffn, 8)
    assert padded_down.shape == (8, padded_ffn)
    torch.testing.assert_close(padded_down[:, :ffn], down, rtol=0, atol=0)
    assert torch.count_nonzero(padded_down[:, ffn:]).item() == 0
    for branch in range(2):
        actual = padded_gate_up[branch * padded_ffn : (branch + 1) * padded_ffn]
        torch.testing.assert_close(
            actual[:ffn], gate_up[branch * ffn : (branch + 1) * ffn], rtol=0, atol=0
        )
        assert torch.count_nonzero(actual[ffn:]).item() == 0
    x = torch.randn(3, 8)
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    padded_gate, padded_up = F.linear(x, padded_gate_up).chunk(2, dim=-1)
    torch.testing.assert_close(
        F.linear(F.silu(padded_gate) * padded_up, padded_down),
        F.linear(F.silu(gate) * up, down),
        rtol=1e-4,
        atol=1e-3,
    )
