import math
from dataclasses import dataclass
from typing import Dict, Tuple

from GPU import GPUConfig
from model import ModelConfig, QuantConfig

# Fraction of each user's KV cache kept resident in HBM at steady state.
DEFAULT_HBM_RESIDENCY = 0.5
# Per-GPU fallback bandwidth when KV cache spills past host DRAM into NVMe.
NVME_FALLBACK_BANDWIDTH = 32e9

BYTES_PER_ACTIVATION = 2.0  # assume fp16 activations for comm + stash
TRAIN_ACTIVATION_STASH = 2.5  # multiplier to cover checkpoints + backward buffers
TRAIN_OPTIMIZER_MULT = 2.0    # Adam moments (m, v) alongside weights


@dataclass(frozen=True)
class CommPattern:
    bytes_per_token: float
    messages_per_token: float

    def scaled(self, factor: float) -> "CommPattern":
        if factor == 1.0:
            return self
        return CommPattern(
            bytes_per_token=self.bytes_per_token * factor,
            messages_per_token=self.messages_per_token * factor,
        )


def batch_efficiency(batch_size: int) -> float:
    """
    Empirical utilization scaling for decode batch size.
    Large batches hide kernel overheads and comm latency.
    """
    base = 0.65 + 0.08 * math.log2(max(batch_size, 1))
    return max(0.55, min(0.92, base))


def train_efficiency(global_batch: int) -> float:
    """
    Simplified utilisation curve for training.
    Larger global batches amortise optimizer + communication overheads.
    """
    base = 0.50 + 0.05 * math.log2(max(global_batch, 1))
    return max(0.45, min(0.90, base))


def kv_bytes_per_token(model: ModelConfig, quant: QuantConfig) -> float:
    # 2 (key + value) * layers * hidden_size * bytes per element
    return 2.0 * model.num_layers * model.hidden_size * quant.kv_bytes_per_elem


def tensor_parallel_comm(model: ModelConfig, tp: int, pp: int) -> CommPattern:
    if tp <= 1:
        return CommPattern(0.0, 0.0)
    layers_per_stage = model.num_layers / pp
    # All-reduce on attention and MLP outputs per layer; approximate with 2 * hidden
    per_layer_bytes = 2.0 * model.hidden_size * BYTES_PER_ACTIVATION
    allreduce_factor = (tp - 1) / tp
    messages_per_layer = 2.0  # attention + MLP collectives
    collectives = layers_per_stage * messages_per_layer
    total_bytes = per_layer_bytes * layers_per_stage * allreduce_factor
    return CommPattern(total_bytes, collectives)


def pipeline_comm(model: ModelConfig, pp: int) -> CommPattern:
    if pp <= 1:
        return CommPattern(0.0, 0.0)
    activation_size = model.hidden_size * BYTES_PER_ACTIVATION
    boundaries = pp - 1
    return CommPattern(activation_size * boundaries, boundaries)


def moe_dispatch(model: ModelConfig) -> CommPattern:
    if not model.is_moe:
        return CommPattern(0.0, 0.0)
    total_bytes = (
        model.hidden_size
        * BYTES_PER_ACTIVATION
        * model.active_experts
        * model.moe_dispatch_factor
    )
    # moe_dispatch_factor already encodes send + gather, so treat it as the message count
    return CommPattern(total_bytes, model.moe_dispatch_factor)


def _weights_per_gpu(
    model: ModelConfig,
    quant: QuantConfig,
    tp: int,
    pp: int,
    ep: int,
) -> float:
    total_weights = model.num_params * quant.bytes_per_param
    base_shards = tp * pp

    if model.is_moe:
        shared_weights = total_weights * model.shared_param_fraction
        expert_weights = total_weights * model.expert_param_fraction
        return (shared_weights / base_shards) + (expert_weights / (base_shards * ep))
    return total_weights / base_shards


def fits_memory(
    gpu: GPUConfig,
    model: ModelConfig,
    quant: QuantConfig,
    tp: int,
    pp: int,
    ep: int,
    batch_size: int,
    hbm_residency: float = 1.0,
    safety_margin: float = 0.15,
) -> Tuple[bool, float, Dict[str, float]]:
    weights_per_gpu = _weights_per_gpu(model, quant, tp, pp, ep)

    kv_per_token = kv_bytes_per_token(model, quant) / (tp * pp)
    tokens_cached = model.total_cached_tokens * batch_size
    residency = max(0.0, min(hbm_residency, 1.0))
    kv_cache_total = kv_per_token * tokens_cached
    kv_cache = kv_cache_total * residency
    kv_cache_remote = max(0.0, kv_cache_total - kv_cache)

    other_overhead = 0.05 * gpu.max_mem_bytes  # activations, optimizer states (small for inference)
    total_usage = weights_per_gpu + kv_cache + other_overhead
    limit = gpu.max_mem_bytes * (1.0 - safety_margin)

    breakdown = {
        "weights": weights_per_gpu,
        "kv_cache": kv_cache,
        "kv_cache_remote": kv_cache_remote,
        "kv_residency": residency,
        "overhead": other_overhead,
        "limit": limit,
    }
    return total_usage <= limit, total_usage, breakdown


def training_activation_bytes_per_token(
    model: ModelConfig,
    tp: int,
    pp: int,
    stash_factor: float = TRAIN_ACTIVATION_STASH,
) -> float:
    layers_per_stage = model.num_layers / pp
    return (
        layers_per_stage
        * model.hidden_size
        * BYTES_PER_ACTIVATION
        * stash_factor
        / max(tp, 1)
    )


def _training_activation_bytes(
    model: ModelConfig,
    batch_size: int,
    tp: int,
    pp: int,
    stash_factor: float = TRAIN_ACTIVATION_STASH,
) -> float:
    tokens = batch_size * model.context_length
    per_token = training_activation_bytes_per_token(
        model=model, tp=tp, pp=pp, stash_factor=stash_factor
    )
    return tokens * per_token


def fits_training_memory(
    gpu: GPUConfig,
    model: ModelConfig,
    quant: QuantConfig,
    tp: int,
    pp: int,
    ep: int,
    batch_size: int,
    safety_margin: float = 0.12,
) -> Tuple[bool, float, Dict[str, float]]:
    weights_per_gpu = _weights_per_gpu(model, quant, tp, pp, ep)
    optimizer_states = weights_per_gpu * TRAIN_OPTIMIZER_MULT
    activations = _training_activation_bytes(model, batch_size, tp, pp)
    other_overhead = 0.03 * gpu.max_mem_bytes  # gradients, temporary buffers

    total_usage = weights_per_gpu + optimizer_states + activations + other_overhead
    limit = gpu.max_mem_bytes * (1.0 - safety_margin)

    breakdown = {
        "weights": weights_per_gpu,
        "optimizer": optimizer_states,
        "activations": activations,
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
