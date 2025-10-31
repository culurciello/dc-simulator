import math
from typing import Dict, Tuple

from GPU import GPUConfig
from model import ModelConfig, QuantConfig


def batch_efficiency(batch_size: int) -> float:
    """
    Empirical utilization scaling for decode batch size.
    Large batches hide kernel overheads and comm latency.
    """
    base = 0.65 + 0.08 * math.log2(max(batch_size, 1))
    return max(0.55, min(0.92, base))


def kv_bytes_per_token(model: ModelConfig, quant: QuantConfig) -> float:
    # 2 (key + value) * layers * hidden_size * bytes per element
    return 2.0 * model.num_layers * model.hidden_size * quant.kv_bytes_per_elem


def tensor_parallel_comm_bytes(model: ModelConfig, tp: int, pp: int) -> float:
    if tp <= 1:
        return 0.0
    bytes_per_activation = 2.0  # fp16 activations
    layers_per_stage = model.num_layers / pp
    # All-reduce on attention and MLP outputs per layer; approximate with 2 * hidden
    per_layer_bytes = 2.0 * model.hidden_size * bytes_per_activation
    allreduce_factor = (tp - 1) / tp
    return per_layer_bytes * layers_per_stage * allreduce_factor


def pipeline_comm_bytes(model: ModelConfig, pp: int) -> float:
    if pp <= 1:
        return 0.0
    bytes_per_activation = 2.0
    activation_size = model.hidden_size * bytes_per_activation
    boundaries = pp - 1
    return activation_size * boundaries


def moe_dispatch_bytes(model: ModelConfig) -> float:
    if not model.is_moe:
        return 0.0
    bytes_per_activation = 2.0  # assume fp16 routed activations
    return (
        model.hidden_size
        * bytes_per_activation
        * model.active_experts
        * model.moe_dispatch_factor
    )


def fits_memory(
    gpu: GPUConfig,
    model: ModelConfig,
    quant: QuantConfig,
    tp: int,
    pp: int,
    ep: int,
    batch_size: int,
    safety_margin: float = 0.15,
) -> Tuple[bool, float, Dict[str, float]]:
    total_weights = model.num_params * quant.bytes_per_param
    base_shards = tp * pp

    if model.is_moe:
        shared_weights = total_weights * model.shared_param_fraction
        expert_weights = total_weights * model.expert_param_fraction
        weights_per_gpu = (shared_weights / base_shards) + (
            expert_weights / (base_shards * ep)
        )
    else:
        weights_per_gpu = total_weights / (base_shards)

    kv_bytes_token = kv_bytes_per_token(model, quant)
    kv_per_gpu_per_token = kv_bytes_token / (tp * pp)
    tokens_cached = model.total_cached_tokens * batch_size
    kv_cache = kv_per_gpu_per_token * tokens_cached

    other_overhead = 0.05 * gpu.max_mem_bytes  # activations, optimizer states (small for inference)
    total_usage = weights_per_gpu + kv_cache + other_overhead
    limit = gpu.max_mem_bytes * (1.0 - safety_margin)

    breakdown = {
        "weights": weights_per_gpu,
        "kv_cache": kv_cache,
        "overhead": other_overhead,
        "limit": limit,
    }
    return total_usage <= limit, total_usage, breakdown


def format_gbps(value_bytes_per_second: float) -> float:
    return value_bytes_per_second / 1e9


def human_gbytes(value_bytes: float) -> float:
    return value_bytes / (1024**3)


def slugify(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def format_bound(value: float) -> str:
    if math.isinf(value):
        return "  inf"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-1):
        return f"{value:.2e}"
    return f"{value:6.1f}"

