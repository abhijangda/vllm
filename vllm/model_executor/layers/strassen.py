# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import json
from functools import cache
from math import gcd
from pathlib import Path

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
PREFILL_TOKEN_SIZES = tuple(range(1024, 16385, 1024))
CLUSTER_PARITY_TOKEN_SIZES = tuple(range(512, 16385, 512))
MLP2_KERNELS = (
    "pingpong_reduce_4x128_opt_no_2x1",
    "pingpong_reduce_4x128_opt_no_1x2",
    "cooperative_reduce_4x256_opt_no_2x1",
    "cooperative_reduce_4x256_opt_no_1x2",
)
UNSAFE_PACKING_WARNING = (
    "UNSAFE Strassen timing experiment: activation packing is DISABLED. "
    "Row-major activations are interpreted as quadrant-packed data; "
    "model outputs are INCORRECT. Do not use this mode for inference."
)


def effective_strassen_swizzle(tokens: int, swizzle: int) -> int:
    return gcd(swizzle, tokens // (2 * 128 * 2))


@cache
def load_strassen_configs(path: str) -> dict[int, tuple[int, int]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {
        str(tokens) for tokens in PREFILL_TOKEN_SIZES
    }:
        raise ValueError(
            "Strassen config must cover every 1024-token multiple to 16384"
        )
    configs = {}
    for tokens, config in data.items():
        if (
            not isinstance(config, dict)
            or config.get("raster") not in ("N", "M")
            or type(config.get("swizzle")) is not int
            or config["swizzle"] not in (1, 2, 4)
        ):
            raise ValueError(f"Invalid Strassen configuration for {tokens} tokens")
        m = int(tokens)
        swizzle = config["swizzle"]
        if effective_strassen_swizzle(m, swizzle) != swizzle:
            raise ValueError(f"Unsupported Strassen swizzle {swizzle} for {m} tokens")
        configs[m] = swizzle, int(config["raster"] == "M")
    return configs


@cache
def load_mlp2_configs(path: str) -> dict[int, tuple[int, int, int]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {
        str(tokens) for tokens in PREFILL_TOKEN_SIZES
    }:
        raise ValueError("MLP2 config must cover every 1024-token multiple to 16384")
    result = {}
    for tokens, config in data.items():
        if (
            not isinstance(config, dict)
            or config.get("kernel") not in MLP2_KERNELS
            or config.get("raster") not in ("N", "M")
            or type(config.get("swizzle")) is not int
            or config["swizzle"] not in (1, 2, 4)
        ):
            raise ValueError(f"Invalid MLP2 Strassen configuration for {tokens} tokens")
        if (
            config["kernel"].endswith("2x1")
            and effective_strassen_swizzle(int(tokens), config["swizzle"])
            != config["swizzle"]
        ):
            raise ValueError(f"Unsupported MLP2 swizzle for {tokens} tokens")
        result[int(tokens)] = (
            MLP2_KERNELS.index(config["kernel"]),
            config["swizzle"],
            int(config["raster"] == "M"),
        )
    return result


def pad_mlp_token_rows(x: torch.Tensor, is_padding: torch.Tensor) -> torch.Tensor:
    """Zero dummy rows and pad the MLP token dimension to a multiple of 1024."""
    if (
        x.ndim != 2
        or is_padding.ndim != 1
        or is_padding.shape[0] != x.shape[0]
        or is_padding.dtype != torch.bool
        or is_padding.device != x.device
    ):
        raise ValueError("MLP token padding requires a matching device-side row mask")
    x = x.masked_fill(is_padding[:, None], 0)
    return F.pad(x, (0, 0, 0, -x.shape[0] % 1024))


@cache
def _library() -> ctypes.CDLL:
    path = envs.VLLM_STRASSEN_LIBRARY_PATH
    if not path:
        raise RuntimeError("VLLM_STRASSEN_LIBRARY_PATH must name the BF16 library")
    lib = ctypes.CDLL(path)
    lib.vllm_strassen_configure.argtypes = []
    lib.vllm_strassen_configure.restype = ctypes.c_int
    lib.vllm_strassen_workspace_size.argtypes = [ctypes.c_int] * 3
    lib.vllm_strassen_workspace_size.restype = ctypes.c_size_t
    lib.vllm_strassen_prepare.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    lib.vllm_strassen_prepare.restype = ctypes.c_int
    lib.vllm_strassen_run_padded_v3.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 9 + [ctypes.c_void_p]
    )
    lib.vllm_strassen_run_padded_v3.restype = ctypes.c_int
    lib.vllm_strassen_mlp2_configure.argtypes = []
    lib.vllm_strassen_mlp2_configure.restype = ctypes.c_int
    lib.vllm_strassen_mlp2_workspace_size.argtypes = [ctypes.c_int] * 3
    lib.vllm_strassen_mlp2_workspace_size.restype = ctypes.c_size_t
    lib.vllm_strassen_mlp2_run_v2.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 9 + [ctypes.c_void_p]
    )
    lib.vllm_strassen_mlp2_run_v2.restype = ctypes.c_int
    return lib


def _check_status(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"BF16 Strassen {operation} failed (status={status})")


@cache
def _workspace(device: torch.device, n: int, k: int) -> torch.Tensor:
    lib = _library()
    with torch.accelerator.device_index(device.index):
        _check_status(lib.vllm_strassen_configure(), "configuration")
        size = max(
            lib.vllm_strassen_workspace_size(tokens, n, k)
            for tokens in CLUSTER_PARITY_TOKEN_SIZES
        )
        return torch.empty(size, dtype=torch.uint8, device=device)


@cache
def _mlp2_workspace(device: torch.device) -> torch.Tensor:
    with torch.accelerator.device_index(device.index):
        lib = _library()
        _check_status(lib.vllm_strassen_mlp2_configure(), "MLP2 configuration")
        size = max(
            lib.vllm_strassen_mlp2_workspace_size(tokens, 8192, 29696)
            for tokens in PREFILL_TOKEN_SIZES
        )
        return torch.empty(size, dtype=torch.uint8, device=device)


def pad_mlp_weight(weight: torch.Tensor, *, gate_up: bool) -> torch.Tensor:
    """Zero-pad the FFN dimension, preserving separate gate/up branches."""
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("MLP padding requires a nonempty matrix")
    if gate_up and weight.shape[0] % 2:
        raise ValueError("Gate/up weights must have equal branch widths")
    ffn = weight.shape[0] // 2 if gate_up else weight.shape[1]
    padding = -ffn % 1024
    if not padding:
        return weight
    if gate_up:
        return F.pad(weight.reshape(2, ffn, -1), (0, 0, 0, padding)).flatten(0, 1)
    return F.pad(weight, (0, padding))


def prepare_strassen_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not weight.is_cuda
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
        or min(weight.shape) <= 0
        or weight.shape[0] % 2
        or weight.shape[1] % 128
    ):
        raise ValueError("Strassen requires contiguous CUDA BF16 (N, K) weights")
    if torch.cuda.get_device_capability(weight.device) != (9, 0):
        raise ValueError("The Strassen library requires an SM90 GPU")
    n, k = weight.shape
    padded_n = 2 * ((n // 2 + 1023) // 1024) * 1024
    packed = torch.empty((k, padded_n), dtype=weight.dtype, device=weight.device)
    presums = torch.empty_like(packed)
    with torch.accelerator.device_index(weight.device.index):
        _check_status(
            _library().vllm_strassen_prepare(
                weight.data_ptr(),
                packed.data_ptr(),
                presums.data_ptr(),
                n,
                padded_n,
                k,
                torch.cuda.current_stream(weight.device).cuda_stream,
            ),
            "weight preprocessing",
        )
    return packed, presums


def strassen_bf16_linear(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    packed: torch.Tensor,
    presums: torch.Tensor,
    workspace: torch.Tensor,
    swizzle: int,
    raster: int,
) -> torch.Tensor:
    cluster_parity = envs.VLLM_STRASSEN_CLUSTER_PARITY
    supported_tokens = (
        CLUSTER_PARITY_TOKEN_SIZES if cluster_parity else PREFILL_TOKEN_SIZES
    )
    result_rows = None
    if (
        envs.VLLM_STRASSEN_PAD_TOKEN_ROWS
        and x.ndim == 2
        and 0 < x.shape[0] <= PREFILL_TOKEN_SIZES[-1]
        and x.shape[0] not in supported_tokens
    ):
        # Startup warmup can bypass graph-size padding.
        result_rows = x.shape[0]
        x = F.pad(x, (0, 0, 0, -result_rows % 1024))
    if x.ndim != 2 or x.shape[0] not in supported_tokens:
        if weight is None:
            raise RuntimeError(
                "Strassen dense weights were released; linear fallback requires "
                "retained weights. Use zero-padded supported token counts."
            )
        return F.linear(x, weight)
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Strassen prefill requires contiguous BF16 activations")
    m, k = x.shape
    n = weight.shape[0] if weight is not None else packed.shape[1]
    padded_n = packed.shape[1]
    if weight is not None and (k != weight.shape[1] or x.device != weight.device):
        raise ValueError("Strassen activation and weight dimensions/devices differ")
    cluster_m = 2
    if cluster_parity:
        if envs.VLLM_STRASSEN_CONFIG_PATH:
            raise ValueError(
                "Cluster-parity dispatch cannot use a Strassen config table"
            )
        cluster_m, swizzle, raster = (2, 4, 0) if m % 2048 == 0 else (1, 2, 1)
    elif envs.VLLM_STRASSEN_CONFIG_PATH:
        swizzle, raster = load_strassen_configs(envs.VLLM_STRASSEN_CONFIG_PATH)[m]
    if swizzle not in (1, 2, 4) or raster not in (0, 1):
        raise ValueError("Strassen requires swizzle 1/2/4 and raster 0/1")
    # Swizzles must divide the half-M row-cluster count.
    if cluster_m == 2:
        swizzle = effective_strassen_swizzle(m, swizzle)
    if (
        n % 2048
        or packed.shape != (k, n)
        or presums.shape != packed.shape
        or any(
            tensor.dtype != torch.bfloat16
            or tensor.device != x.device
            or not tensor.is_contiguous()
            for tensor in (weight, packed, presums)
            if tensor is not None
        )
        or workspace.dtype != torch.uint8
        or workspace.device != x.device
        or not workspace.is_contiguous()
        or workspace.numel() < _library().vllm_strassen_workspace_size(m, padded_n, k)
    ):
        raise ValueError("Strassen weight buffers or workspace are incompatible")
    skip_packing = envs.VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING
    if skip_packing:
        logger.warning_once(UNSAFE_PACKING_WARNING)
    packed_a = None if skip_packing else torch.empty_like(x)
    output = x.new_empty((m, n))
    logger.info_once(
        "Strassen CUDA launch: M=%d N=%d K=%d; row alignment=%d; "
        "cluster=%dx%d; raster=%d swizzle=%d; activation packing=%s",
        m,
        n,
        k,
        512 if cluster_parity else 1024,
        cluster_m,
        2 // cluster_m,
        raster,
        swizzle,
        not skip_packing,
    )
    with torch.accelerator.device_index(x.device.index):
        device_index = x.device.index
        assert device_index is not None
        properties = torch.cuda.get_device_properties(x.device)
        _check_status(
            _library().vllm_strassen_run_padded_v3(
                x.data_ptr(),
                packed_a.data_ptr() if packed_a is not None else None,
                packed.data_ptr(),
                presums.data_ptr(),
                output.data_ptr(),
                workspace.data_ptr(),
                m,
                n,
                k,
                device_index,
                properties.multi_processor_count,
                swizzle,
                raster,
                int(skip_packing),
                cluster_m,
                torch.cuda.current_stream(x.device).cuda_stream,
            ),
            "projection",
        )
    return output[:result_rows] if result_rows is not None else output


def _strassen_bf16_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    packed: torch.Tensor,
    presums: torch.Tensor,
    workspace: torch.Tensor,
    swizzle: int,
    raster: int,
) -> torch.Tensor:
    n = weight.shape[0] if weight is not None else packed.shape[1]
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    "strassen_bf16_linear",
    strassen_bf16_linear,
    mutates_args=["workspace"],
    fake_impl=_strassen_bf16_linear_fake,
)


def strassen_bf16_mlp2(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed: torch.Tensor,
    presums: torch.Tensor,
    workspace: torch.Tensor,
    kernel_id: int,
    swizzle: int,
    raster: int,
) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] not in PREFILL_TOKEN_SIZES:
        return F.linear(x, weight)
    m, k = x.shape
    if envs.VLLM_STRASSEN_MLP2_CONFIG_PATH:
        kernel_id, swizzle, raster = load_mlp2_configs(
            envs.VLLM_STRASSEN_MLP2_CONFIG_PATH
        )[m]
    if (
        x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or k != 29696
        or weight.shape != (8192, k)
        or packed.shape != (k, 8192)
        or presums.shape != packed.shape
        or any(
            t.dtype != torch.bfloat16 or t.device != x.device or not t.is_contiguous()
            for t in (weight, packed, presums)
        )
        or workspace.device != x.device
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
        or not 0 <= kernel_id < len(MLP2_KERNELS)
        or swizzle not in (1, 2, 4)
        or raster not in (0, 1)
    ):
        raise ValueError(
            "MLP2 Strassen requires compatible BF16 weights and configuration"
        )
    if (
        MLP2_KERNELS[kernel_id].endswith("2x1")
        and effective_strassen_swizzle(m, swizzle) != swizzle
    ):
        raise ValueError(f"Unsupported MLP2 swizzle {swizzle} for {m} tokens")
    lib = _library()
    if workspace.numel() < lib.vllm_strassen_mlp2_workspace_size(m, 8192, k):
        raise ValueError("MLP2 Strassen workspace is too small")
    skip_packing = envs.VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING
    if skip_packing:
        logger.warning_once(UNSAFE_PACKING_WARNING)
    packed_a = None if skip_packing else torch.empty_like(x)
    output = x.new_empty((m, 8192))
    logger.info_once(
        "Strassen MLP2 CUDA launch: M=%d N=8192 K=%d; kernel=%s; "
        "raster=%d swizzle=%d; activation packing=%s",
        m,
        k,
        MLP2_KERNELS[kernel_id],
        raster,
        swizzle,
        not skip_packing,
    )
    with torch.accelerator.device_index(x.device.index):
        device = x.device.index
        assert device is not None
        _check_status(
            lib.vllm_strassen_mlp2_run_v2(
                x.data_ptr(),
                packed_a.data_ptr() if packed_a is not None else None,
                packed.data_ptr(),
                presums.data_ptr(),
                output.data_ptr(),
                workspace.data_ptr(),
                m,
                8192,
                k,
                device,
                torch.cuda.get_device_properties(device).multi_processor_count,
                swizzle,
                raster,
                int(skip_packing),
                kernel_id,
                torch.cuda.current_stream(device).cuda_stream,
            ),
            "MLP2 projection",
        )
    return output


def _strassen_bf16_mlp2_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed: torch.Tensor,
    presums: torch.Tensor,
    workspace: torch.Tensor,
    kernel_id: int,
    swizzle: int,
    raster: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    "strassen_bf16_mlp2",
    strassen_bf16_mlp2,
    mutates_args=["workspace"],
    fake_impl=_strassen_bf16_mlp2_fake,
)


class StrassenLinearMethod(UnquantizedLinearMethod):
    """Opt-in SM90 BF16 gate/up projection for supported Qwen prefill sizes."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if envs.VLLM_STRASSEN_CLUSTER_PARITY:
            if envs.VLLM_STRASSEN_CONFIG_PATH:
                raise ValueError(
                    "Cluster-parity dispatch cannot use a Strassen config table"
                )
            logger.info_once(
                "Strassen cluster parity: multiples of 2048 tokens use "
                "cluster=2x1 raster=N swizzle=4; all other multiples of 512 "
                "use cluster=1x2 raster=M swizzle=2."
            )
        drop_dense = envs.VLLM_STRASSEN_DROP_DENSE_WEIGHTS
        if drop_dense and not envs.VLLM_STRASSEN_PAD_TOKEN_ROWS:
            raise ValueError(
                "Dropping Strassen dense weights requires "
                "VLLM_STRASSEN_PAD_TOKEN_ROWS=1"
            )
        if envs.VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING:
            logger.warning_once(UNSAFE_PACKING_WARNING)
        if envs.VLLM_STRASSEN_CONFIG_PATH:
            configs = load_strassen_configs(envs.VLLM_STRASSEN_CONFIG_PATH)
            logger.info_once(
                "Using per-token-size Strassen configurations: %s", str(configs)
            )
        self.swizzle = envs.VLLM_STRASSEN_SWIZZLE
        raster = envs.VLLM_STRASSEN_RASTER
        if self.swizzle not in (1, 2, 4) or raster not in ("N", "M"):
            raise ValueError("Strassen requires swizzle 1/2/4 and raster N/M")
        self.raster = int(raster == "M")
        if self.swizzle == 4 and not envs.VLLM_STRASSEN_CLUSTER_PARITY:
            logger.info_once(
                "Strassen caps swizzle 4 to 2 for odd multiples of 1024 tokens."
            )
        layer.weight.data = pad_mlp_weight(layer.weight.data, gate_up=True)
        packed, presums = prepare_strassen_weight(layer.weight)
        layer.register_buffer("strassen_packed_weight", packed, persistent=False)
        layer.register_buffer("strassen_weight_presums", presums, persistent=False)
        layer.register_buffer(
            "strassen_workspace",
            _workspace(layer.weight.device, packed.shape[1], packed.shape[0]),
            persistent=False,
        )
        logger.info(
            "Prepared BF16 Strassen %s: weight=%s, padded N=%d, raster=%s, "
            "swizzle=%d, prefill tokens=%s",
            layer.prefix,
            tuple(layer.weight.shape),
            packed.shape[1],
            raster,
            self.swizzle,
            CLUSTER_PARITY_TOKEN_SIZES
            if envs.VLLM_STRASSEN_CLUSTER_PARITY
            else PREFILL_TOKEN_SIZES,
        )
        if drop_dense:
            layer.register_parameter("weight", None)
            logger.info(
                "Released dense MLP1 weight for %s; packed weights and pre-sums "
                "are retained, and non-Strassen fallback is disabled.",
                layer.prefix,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("Strassen gate/up projection does not support bias")
        return torch.ops.vllm.strassen_bf16_linear(
            x,
            layer.weight,
            layer.strassen_packed_weight,
            layer.strassen_weight_presums,
            layer.strassen_workspace,
            self.swizzle,
            self.raster,
        )


class StrassenDownLinearMethod(UnquantizedLinearMethod):
    """Retain padded FFN channels with optional per-size BF16 Strassen MLP2."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.weight.data = pad_mlp_weight(layer.weight.data, gate_up=False)
        logger.info(
            "Padded Strassen down projection %s: %s", layer.prefix, layer.weight.shape
        )
        config_path = envs.VLLM_STRASSEN_MLP2_CONFIG_PATH
        self.use_strassen = bool(config_path)
        if config_path:
            load_mlp2_configs(config_path)
            if layer.weight.shape != (8192, 29696):
                raise ValueError("MLP2 Strassen requires TP=1 Qwen2.5-72B weights")
            packed, presums = prepare_strassen_weight(layer.weight)
            layer.register_buffer("strassen_packed_weight", packed, persistent=False)
            layer.register_buffer("strassen_weight_presums", presums, persistent=False)
            layer.register_buffer(
                "strassen_workspace",
                _mlp2_workspace(layer.weight.device),
                persistent=False,
            )
            logger.info("Prepared BF16 Strassen MLP2 %s", layer.prefix)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_strassen:
            return super().apply(layer, x, bias)
        if bias is not None:
            raise ValueError("Strassen MLP2 does not support bias")
        return torch.ops.vllm.strassen_bf16_mlp2(
            x,
            layer.weight,
            layer.strassen_packed_weight,
            layer.strassen_weight_presums,
            layer.strassen_workspace,
            0,
            1,
            0,
        )
