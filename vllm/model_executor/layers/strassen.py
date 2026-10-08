# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from functools import cache

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
PREFILL_TOKEN_SIZES = (1024, 2048, 4096, 8192, 16384)
UNSAFE_PACKING_WARNING = (
    "UNSAFE Strassen timing experiment: activation packing is DISABLED. "
    "Row-major activations are interpreted as quadrant-packed data; "
    "model outputs are INCORRECT. Do not use this mode for inference."
)


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
    lib.vllm_strassen_run_padded_v2.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8 + [ctypes.c_void_p]
    )
    lib.vllm_strassen_run_padded_v2.restype = ctypes.c_int
    return lib


def _check_status(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"BF16 Strassen {operation} failed (status={status})")


@cache
def _workspace(device: torch.device, n: int, k: int) -> torch.Tensor:
    lib = _library()
    with torch.cuda.device(device):
        _check_status(lib.vllm_strassen_configure(), "configuration")
        size = max(
            lib.vllm_strassen_workspace_size(tokens, n, k)
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
    with torch.cuda.device(weight.device):
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
    weight: torch.Tensor,
    packed: torch.Tensor,
    presums: torch.Tensor,
    workspace: torch.Tensor,
    swizzle: int,
    raster: int,
) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] not in PREFILL_TOKEN_SIZES:
        return F.linear(x, weight)
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Strassen prefill requires contiguous BF16 activations")
    m, k = x.shape
    n = weight.shape[0]
    padded_n = packed.shape[1]
    if k != weight.shape[1] or x.device != weight.device:
        raise ValueError("Strassen activation and weight dimensions/devices differ")
    if swizzle not in (1, 2, 4) or raster not in (0, 1):
        raise ValueError("Strassen requires swizzle 1/2/4 and raster 0/1")
    # Fused Strassen tiles cannot swizzle past the half-M row-cluster count.
    swizzle = min(swizzle, m // (2 * 128 * 2))
    if (
        weight.dtype != torch.bfloat16
        or n % 2048
        or packed.shape != (k, n)
        or presums.shape != packed.shape
        or any(
            tensor.dtype != torch.bfloat16
            or tensor.device != x.device
            or not tensor.is_contiguous()
            for tensor in (weight, packed, presums)
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
    with torch.cuda.device(x.device):
        device_index = x.device.index
        assert device_index is not None
        properties = torch.cuda.get_device_properties(x.device)
        _check_status(
            _library().vllm_strassen_run_padded_v2(
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
                torch.cuda.current_stream(x.device).cuda_stream,
            ),
            "projection",
        )
    return output


def _strassen_bf16_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed: torch.Tensor,
    presums: torch.Tensor,
    workspace: torch.Tensor,
    swizzle: int,
    raster: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    "strassen_bf16_linear",
    strassen_bf16_linear,
    mutates_args=["workspace"],
    fake_impl=_strassen_bf16_linear_fake,
)


class StrassenLinearMethod(UnquantizedLinearMethod):
    """Opt-in SM90 BF16 gate/up projection for supported Qwen prefill sizes."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if envs.VLLM_STRASSEN_UNSAFE_SKIP_ACTIVATION_PACKING:
            logger.warning_once(UNSAFE_PACKING_WARNING)
        self.swizzle = envs.VLLM_STRASSEN_SWIZZLE
        raster = envs.VLLM_STRASSEN_RASTER
        if self.swizzle not in (1, 2, 4) or raster not in ("N", "M"):
            raise ValueError("Strassen requires swizzle 1/2/4 and raster N/M")
        self.raster = int(raster == "M")
        if self.swizzle == 4:
            logger.info_once("Strassen caps swizzle 4 to 2 for 1024-token projections.")
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
            PREFILL_TOKEN_SIZES,
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
    """Retain the padded FFN width through the down projection."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.weight.data = pad_mlp_weight(layer.weight.data, gate_up=False)
        logger.info(
            "Padded Strassen down projection %s: %s", layer.prefix, layer.weight.shape
        )
