# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.model_executor.layers.strassen import (
    ATTENTION_KERNELS,
    CLUSTER_PARITY_TOKEN_SIZES,
    MLP2_KERNELS,
    PREFILL_TOKEN_SIZES,
    StrassenAttentionLinearMethod,
    StrassenDownLinearMethod,
    StrassenLinearMethod,
    _attention_workspace,
    _library,
    _mlp2_workspace,
    _workspace,
    load_attention_configs,
    load_mlp2_configs,
    load_strassen_configs,
    pad_mlp_token_rows,
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


@pytest.fixture(scope="module")
def down_projection(projection):
    weight = torch.randn(8192, 29568, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = pad_mlp_weight(weight, gate_up=False)
    packed, presums = prepare_strassen_weight(weight)
    return weight, packed, presums, _mlp2_workspace(weight.device)


@pytest.fixture(scope="module", params=["attention_qkv", "attention_o_proj"])
def attention_projection(projection, request):
    name = request.param
    n = 10240 if name == "attention_qkv" else 8192
    weight = torch.randn(n, 8192, device="cuda", dtype=torch.bfloat16) * 0.01
    bias = (
        torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.01
        if name == "attention_qkv"
        else None
    )
    packed, presums = prepare_strassen_weight(weight)
    return name, weight, bias, packed, presums, _attention_workspace(weight.device)


@pytest.mark.parametrize("kernel_id", range(len(ATTENTION_KERNELS)))
@pytest.mark.parametrize("tokens", PREFILL_TOKEN_SIZES)
def test_attention_variants_preserve_channels_and_bias(
    attention_projection, kernel_id, tokens
):
    name, weight, bias, packed, presums, workspace = attention_projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    expected = F.linear(x, weight, bias)
    workspace.fill_(255)
    with patch(
        "vllm.model_executor.layers.strassen.F.linear",
        side_effect=AssertionError("Supported attention sizes must use Strassen"),
    ):
        actual = torch.ops.vllm.strassen_bf16_attention(
            x, weight, bias, packed, presums, workspace, name, kernel_id, 1, 0
        )
    assert actual.shape == expected.shape and actual.is_contiguous()
    sizes = [8192, 1024, 1024] if name == "attention_qkv" else [8192]
    for a, e in zip(actual.split(sizes, -1), expected.split(sizes, -1)):
        assert_projection_close(a, e)


@pytest.mark.parametrize("kernel_id", range(len(ATTENTION_KERNELS)))
@pytest.mark.parametrize("tokens", [1024, 16384])
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("skip_bias", [False, True])
def test_attention_graph_replay_recomputes_projection(
    attention_projection, kernel_id, tokens, unsafe, skip_bias, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING", unsafe)
    monkeypatch.setattr(envs, "VLLM_STRASSEN_UNSAFE_SKIP_QKV_BIAS", skip_bias)
    name, weight, bias, packed, presums, workspace = attention_projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.vllm.strassen_bf16_attention(
                x, weight, bias, packed, presums, workspace, name, kernel_id, 2, 1
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.vllm.strassen_bf16_attention(
            x, weight, bias, packed, presums, workspace, name, kernel_id, 2, 1
        )
    for scale in (-0.5, 2.0):
        x.mul_(scale)
        workspace.fill_(255)
        graph.replay()
        reference_x = (
            x.reshape(2, 2, tokens // 2, 4096).permute(0, 2, 1, 3).reshape_as(x)
            if unsafe
            else x
        )
        reference_bias = None if skip_bias and name == "attention_qkv" else bias
        assert_projection_close(actual, F.linear(reference_x, weight, reference_bias))


@pytest.mark.parametrize("tokens", [16, 1024])
@pytest.mark.parametrize("skip_bias", [False, True])
def test_qkv_bias_is_preserved_unless_explicitly_bypassed(
    attention_projection, tokens, skip_bias, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_STRASSEN_UNSAFE_SKIP_QKV_BIAS", skip_bias)
    name, weight, bias, packed, presums, workspace = attention_projection
    x = torch.zeros(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    output = torch.ops.vllm.strassen_bf16_attention(
        x, weight, bias, packed, presums, workspace, name, 0, 1, 0
    )
    expected = torch.zeros_like(output)
    if name == "attention_qkv" and not skip_bias:
        expected += bias
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 512])
def test_attention_unsupported_rows_keep_dense_fallback(attention_projection, tokens):
    name, weight, bias, packed, presums, workspace = attention_projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    actual = torch.ops.vllm.strassen_bf16_attention(
        x, weight, bias, packed, presums, workspace, name, 0, 1, 0
    )
    torch.testing.assert_close(actual, F.linear(x, weight, bias), rtol=0, atol=0)


@pytest.mark.parametrize("drop_dense", [False, True])
def test_attention_prepares_once_and_preserves_projection_without_dense_weights(
    attention_projection, tmp_path, monkeypatch, drop_dense
):
    name, weight, bias, _, _, _ = attention_projection
    path = tmp_path / "attention.json"
    path.write_text(
        json.dumps(
            {
                str(m): {"kernel": ATTENTION_KERNELS[0], "raster": "N", "swizzle": 2}
                for m in PREFILL_TOKEN_SIZES
            }
        )
    )
    variable = (
        "VLLM_STRASSEN_QKV_CONFIG_PATH"
        if name == "attention_qkv"
        else "VLLM_STRASSEN_ATTN_OUT_CONFIG_PATH"
    )
    monkeypatch.setattr(envs, variable, str(path))
    monkeypatch.setattr(envs, "VLLM_STRASSEN_DROP_DENSE_WEIGHTS", drop_dense)
    monkeypatch.setattr(envs, "VLLM_STRASSEN_PAD_TOKEN_ROWS", True)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.prefix = f"test.{name}"
    method = StrassenAttentionLinearMethod(name)
    method.process_weights_after_loading(layer)
    assert (layer.weight is None) == drop_dense
    operation = torch.compile(method.apply, backend="eager", fullgraph=True)
    x = torch.randn(31, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    with patch(
        "vllm.model_executor.layers.strassen.prepare_strassen_weight",
        side_effect=AssertionError("Weight pre-sums must not be recomputed"),
    ):
        for scale in (1.0, -0.5):
            x.mul_(scale)
            assert_projection_close(
                operation(layer, x, bias), F.linear(x, weight, bias)
            )


def test_attention_without_dense_weight_rejects_unsupported_rows(attention_projection):
    name, _, bias, packed, presums, workspace = attention_projection
    x = torch.empty(16385, 8192, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="dense weights were released"):
        torch.ops.vllm.strassen_bf16_attention(
            x, None, bias, packed, presums, workspace, name, 0, 1, 0
        )


def test_attention_configuration_rejects_invalid_entries(tmp_path):
    path = tmp_path / "attention.json"
    data = {
        str(m): {"kernel": ATTENTION_KERNELS[0], "raster": "N", "swizzle": 2}
        for m in PREFILL_TOKEN_SIZES
    }
    data["1024"]["swizzle"] = 4
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported attention swizzle"):
        load_attention_configs(str(path))
    data["1024"]["kernel"] = "unknown"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid attention"):
        load_attention_configs(str(path))
    del data["1024"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="must cover"):
        load_attention_configs(str(path))


@pytest.mark.parametrize("kernel_id", [-1, len(ATTENTION_KERNELS)])
def test_attention_native_rejects_unknown_kernel(projection, kernel_id):
    assert (
        _library().vllm_strassen_attention_run_v1(
            None,
            None,
            None,
            None,
            None,
            None,
            1024,
            10240,
            8192,
            0,
            132,
            1,
            0,
            0,
            kernel_id,
            None,
        )
        != 0
    )


@pytest.mark.parametrize("drop_dense", [False, True])
def test_mlp2_weight_release_retains_padded_projection(
    down_projection, tmp_path, monkeypatch, drop_dense
):
    weight, _, _, _ = down_projection
    path = tmp_path / "mlp2.json"
    path.write_text(
        json.dumps(
            {
                str(m): {"kernel": MLP2_KERNELS[0], "raster": "N", "swizzle": 2}
                for m in PREFILL_TOKEN_SIZES
            }
        )
    )
    monkeypatch.setattr(envs, "VLLM_STRASSEN_MLP2_CONFIG_PATH", str(path))
    monkeypatch.setattr(envs, "VLLM_STRASSEN_DROP_DENSE_WEIGHTS", drop_dense)
    monkeypatch.setattr(envs, "VLLM_STRASSEN_PAD_TOKEN_ROWS", True)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.prefix = "test.down_proj"
    method = StrassenDownLinearMethod()
    method.process_weights_after_loading(layer)
    assert (layer.weight is None) == drop_dense
    operation = torch.compile(method.apply, backend="eager", fullgraph=True)
    x = torch.randn(31, 29696, device="cuda", dtype=torch.bfloat16) * 0.1
    with patch(
        "vllm.model_executor.layers.strassen.prepare_strassen_weight",
        side_effect=AssertionError("Weight pre-sums must not be recomputed"),
    ):
        assert_projection_close(operation(layer, x), F.linear(x, weight))


def test_mlp2_without_dense_weight_rejects_unsupported_rows(down_projection):
    _, packed, presums, workspace = down_projection
    x = torch.empty(1, 29696, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="dense weights were released"):
        torch.ops.vllm.strassen_bf16_mlp2(x, None, packed, presums, workspace, 0, 1, 0)


@pytest.mark.parametrize("kernel_id", range(len(MLP2_KERNELS)))
@pytest.mark.parametrize("tokens", PREFILL_TOKEN_SIZES)
def test_mlp2_variants_repack_activations(down_projection, kernel_id, tokens):
    weight, packed, presums, workspace = down_projection
    x = torch.randn(tokens, 29696, device="cuda", dtype=torch.bfloat16) * 0.1
    expected = F.linear(x, weight)
    # Poison the A pre-sums so incomplete K coverage cannot reuse previous data.
    workspace[: tokens * 29696 * 2].fill_(255)
    with patch(
        "vllm.model_executor.layers.strassen.F.linear",
        side_effect=AssertionError("MLP2 must execute the selected Strassen kernel"),
    ):
        actual = torch.ops.vllm.strassen_bf16_mlp2(
            x, weight, packed, presums, workspace, kernel_id, 1, 0
        )
    assert actual.shape == (tokens, 8192) and actual.is_contiguous()
    assert_projection_close(actual, expected)


@pytest.mark.parametrize("kernel_id", range(len(MLP2_KERNELS)))
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("tokens", [1024, 2048, 8192, 16384])
def test_mlp2_graph_replay_recomputes_packed_or_reinterpreted_input(
    down_projection, kernel_id, unsafe, tokens, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING", unsafe)
    weight, packed, presums, workspace = down_projection
    m, k = tokens, 29696
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.vllm.strassen_bf16_mlp2(
                x, weight, packed, presums, workspace, kernel_id, 1, 0
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.vllm.strassen_bf16_mlp2(
            x, weight, packed, presums, workspace, kernel_id, 1, 0
        )
    for scale in (-0.5, 2.0):
        x.mul_(scale)
        workspace[: m * k * 2].fill_(255)
        graph.replay()
        reference_x = (
            x.reshape(2, 2, m // 2, k // 2).permute(0, 2, 1, 3).reshape_as(x)
            if unsafe
            else x
        )
        assert_projection_close(actual, F.linear(reference_x, weight))
        if unsafe:
            assert not torch.allclose(actual, F.linear(x, weight), rtol=0.02, atol=0.01)


def test_both_strassen_mlp_projections_preserve_valid_layout(
    projection, down_projection
):
    up, up_packed, up_sums, up_workspace = projection
    down, down_packed, down_sums, down_workspace = down_projection
    x = torch.randn(1024, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    gate_up = torch.ops.vllm.strassen_bf16_linear(
        x, up, up_packed, up_sums, up_workspace, 2, 0
    )
    gate, activation = gate_up.chunk(2, -1)
    hidden = F.silu(gate) * activation
    output = torch.ops.vllm.strassen_bf16_mlp2(
        hidden, down, down_packed, down_sums, down_workspace, 0, 1, 0
    )
    expected_gate, expected_up = F.linear(x, up).chunk(2, -1)
    expected = F.linear(F.silu(expected_gate) * expected_up, down)
    assert_projection_close(output, expected)


@pytest.mark.parametrize("tokens", [1, 16, 512])
def test_mlp2_other_sizes_keep_padded_cublas_fallback(down_projection, tokens):
    weight, packed, presums, workspace = down_projection
    x = torch.randn(tokens, 29696, device="cuda", dtype=torch.bfloat16)
    actual = torch.ops.vllm.strassen_bf16_mlp2(
        x, weight, packed, presums, workspace, 0, 1, 0
    )
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=0, atol=0)


@pytest.mark.parametrize("kernel_id", [-1, len(MLP2_KERNELS)])
def test_mlp2_native_entry_rejects_unknown_kernel(down_projection, kernel_id):
    status = _library().vllm_strassen_mlp2_run_v2(
        None,
        None,
        None,
        None,
        None,
        None,
        1024,
        8192,
        29696,
        0,
        132,
        1,
        0,
        0,
        kernel_id,
        None,
    )
    assert status != 0


def test_mlp2_config_requires_complete_known_variants(tmp_path):
    path = tmp_path / "mlp2.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="must cover"):
        load_mlp2_configs(str(path))
    configs = {
        str(m): {"kernel": MLP2_KERNELS[0], "raster": "N", "swizzle": 2}
        for m in PREFILL_TOKEN_SIZES
    }
    configs["1024"]["kernel"] = "unknown"
    path.write_text(json.dumps(configs))
    with pytest.raises(ValueError, match="Invalid MLP2"):
        load_mlp2_configs(str(path))
    configs["1024"]["kernel"] = MLP2_KERNELS[0]
    configs["1024"]["swizzle"] = 4
    path.write_text(json.dumps(configs))
    with pytest.raises(ValueError, match="Unsupported MLP2 swizzle"):
        load_mlp2_configs(str(path))
    configs["1024"]["swizzle"] = 2
    path.write_text(json.dumps(configs))
    assert load_mlp2_configs(str(path))[1024] == (0, 2, 0)


def test_mlp2_rejects_swizzle_exceeding_row_cluster_count(down_projection):
    weight, packed, presums, workspace = down_projection
    x = torch.empty(1024, 29696, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Unsupported MLP2 swizzle"):
        torch.ops.vllm.strassen_bf16_mlp2(
            x, weight, packed, presums, workspace, 0, 4, 0
        )


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


@pytest.mark.parametrize(
    "cluster_parity,tokens",
    [(False, tokens) for tokens in PREFILL_TOKEN_SIZES]
    + [(True, tokens) for tokens in CLUSTER_PARITY_TOKEN_SIZES],
)
def test_strassen_graph_replay_uses_changed_activations(
    projection, tokens, cluster_parity, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_STRASSEN_CLUSTER_PARITY", cluster_parity)
    monkeypatch.setattr(envs, "VLLM_STRASSEN_CONFIG_PATH", None)
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


def test_strassen_without_dense_weight_preserves_compiled_projection(projection):
    weight, packed, presums, workspace = projection
    x = torch.randn(1024, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    compiled = torch.compile(
        torch.ops.vllm.strassen_bf16_linear, backend="eager", fullgraph=True
    )
    actual = compiled(x, None, packed, presums, workspace, 2, 0)
    assert_projection_close(actual, F.linear(x, weight))


def test_strassen_without_dense_weight_rejects_fallback(projection):
    _, packed, presums, workspace = projection
    x = torch.randn(8, 8192, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="dense weights were released"):
        torch.ops.vllm.strassen_bf16_linear(x, None, packed, presums, workspace, 2, 0)


@pytest.mark.parametrize("keep_dense", [False, True])
@pytest.mark.parametrize("tokens", [37, 1537])
def test_strassen_zero_pads_unaligned_calls(
    projection, monkeypatch, keep_dense, tokens
):
    monkeypatch.setattr(envs, "VLLM_STRASSEN_PAD_TOKEN_ROWS", True)
    weight, packed, presums, workspace = projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    actual = torch.ops.vllm.strassen_bf16_linear(
        x, weight if keep_dense else None, packed, presums, workspace, 2, 0
    )
    assert actual.shape == (tokens, 59392)
    assert_projection_close(actual, F.linear(x, weight))


@pytest.mark.parametrize("tokens,cluster_m", [(37, 2), (512, 2), (1024, 0), (1024, 3)])
def test_native_strassen_entry_rejects_invalid_launch(projection, tokens, cluster_m):
    status = _library().vllm_strassen_run_padded_v3(
        None,
        None,
        None,
        None,
        None,
        None,
        tokens,
        59392,
        8192,
        0,
        132,
        2,
        0,
        0,
        cluster_m,
        None,
    )
    assert status != 0


@pytest.mark.parametrize(
    "cluster_parity,tokens",
    [(False, 1024), (False, 2048)]
    + [(True, tokens) for tokens in (512, 1024, 1536, 2048, 2560, 15872)],
)
def test_unsafe_skip_packing_reinterprets_row_major_input_in_graph(
    projection, monkeypatch, tokens, cluster_parity
):
    monkeypatch.setattr(envs, "VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING", True)
    monkeypatch.setattr(envs, "VLLM_STRASSEN_CLUSTER_PARITY", cluster_parity)
    monkeypatch.setattr(envs, "VLLM_STRASSEN_CONFIG_PATH", None)
    weight, packed, presums, workspace = projection
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16) * 0.1
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
    wrong_x = x.reshape(2, 2, tokens // 2, 4096).permute(0, 2, 1, 3).reshape_as(x)
    assert_projection_close(actual, F.linear(wrong_x, weight))
    assert not torch.allclose(actual, F.linear(x, weight), rtol=0.02, atol=0.01)


@pytest.mark.parametrize("tokens", [1, 8, 16, 512, 1536])
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


def test_token_row_padding_zeros_graph_padding_and_new_rows():
    x = torch.randn(1025, 8)
    mask = torch.arange(1025) >= 1017
    x[mask] = float("nan")
    padded = pad_mlp_token_rows(x, mask)
    assert padded.shape == (2048, 8)
    torch.testing.assert_close(padded[:1017], x[:1017], rtol=0, atol=0)
    assert torch.count_nonzero(padded[1017:]).item() == 0


def test_token_padding_mask_updates_on_cuda_graph_replay(projection):
    x = torch.randn(1024, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1024, device="cuda", dtype=torch.bool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        padded = pad_mlp_token_rows(x, mask)
    for actual_rows in (1, 7, 64, 1023):
        mask.fill_(True)
        mask[:actual_rows].fill_(False)
        graph.replay()
        torch.testing.assert_close(
            padded[:actual_rows], x[:actual_rows], rtol=0, atol=0
        )
        assert torch.count_nonzero(padded[actual_rows:]).item() == 0


def test_token_padding_compiles_without_specializing_away_dynamic_padding():
    compilations = []

    def check_graph(graph, example_inputs):
        compilations.append(graph)
        return graph.forward

    compiled = torch.compile(
        pad_mlp_token_rows, backend=check_graph, dynamic=True, fullgraph=True
    )
    for tokens in (2048, 37, 1025):
        x = torch.randn(tokens, 8)
        mask = torch.zeros(tokens, dtype=torch.bool)
        result = compiled(x, mask)
        assert result.shape == (((tokens + 1023) // 1024) * 1024, 8)
        torch.testing.assert_close(result[:tokens], x)
        assert torch.count_nonzero(result[tokens:]).item() == 0
    assert len(compilations) == 1


def test_per_size_strassen_config_rejects_incomplete_or_invalid_entries(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="must cover"):
        load_strassen_configs(str(path))
    configs = {
        str(tokens): {"raster": "N", "swizzle": 2} for tokens in PREFILL_TOKEN_SIZES
    }
    configs["3072"]["swizzle"] = 4
    path.write_text(json.dumps(configs))
    with pytest.raises(ValueError, match="Unsupported Strassen swizzle"):
        load_strassen_configs(str(path))
    configs["3072"] = {"raster": "M", "swizzle": 1}
    path.write_text(json.dumps(configs))
    assert load_strassen_configs(str(path))[3072] == (1, 1)


@pytest.mark.parametrize("drop_dense", [False, True])
def test_weight_loading_accepts_per_size_config(
    projection, tmp_path, monkeypatch, drop_dense
):
    path = tmp_path / "configs.json"
    path.write_text(
        json.dumps(
            {
                str(tokens): {"raster": "N", "swizzle": 2}
                for tokens in PREFILL_TOKEN_SIZES
            }
        )
    )
    monkeypatch.setattr(envs, "VLLM_STRASSEN_CONFIG_PATH", str(path))
    monkeypatch.setattr(envs, "VLLM_STRASSEN_DROP_DENSE_WEIGHTS", drop_dense)
    monkeypatch.setattr(envs, "VLLM_STRASSEN_PAD_TOKEN_ROWS", True)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(projection[0], requires_grad=False)
    layer.prefix = "test.gate_up_proj"
    method = StrassenLinearMethod()
    method.process_weights_after_loading(layer)
    assert layer.strassen_packed_weight.shape == (8192, 59392)
    assert (layer.weight is None) == drop_dense
