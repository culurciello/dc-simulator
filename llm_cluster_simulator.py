
#!/usr/bin/env python3
"""
Rack-level LLM throughput simulator.

The simulator models sustained decode throughput for Qwen3-235B across
multiple racks while accounting for
- quantized weight storage (4/8/16 bit)
- realistic KV-cache read bandwidth
- multi-level interconnect costs (GPU↔GPU, server↔server, rack↔rack)
- automatic search over tensor/pipeline parallelism plans

Outputs per-rack summaries for 8 racks of each supported hardware preset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model & hardware descriptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUConfig:
    name: str
    sustained_flops: float  # FLOPs/s (decode-mode sustained)
    hbm_bw: float           # bytes/s of effective HBM bandwidth
    max_mem_bytes: float    # bytes of HBM capacity


@dataclass(frozen=True)
class RackPreset:
    name: str
    gpu_key: str
    gpus_per_server: int
    servers_per_rack: int
    intra_server_bw: float      # bytes/s (e.g., NVLink/XGMI)
    inter_server_bw: float      # bytes/s (within rack, NVSwitch/IF link)
    inter_rack_bw: float        # bytes/s (rack-to-rack optics/IB)
    notes: str = ""


@dataclass(frozen=True)
class QuantConfig:
    bits: int
    bytes_per_param: float
    compute_scale: float
    kv_bytes_per_elem: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_params: float
    num_layers: int
    hidden_size: int
    num_heads: int
    context_length: int       # prompt + cached tokens
    generation_window: int    # planned decode horizon per request
    experts_per_layer: int = 0
    active_experts: int = 0
    expert_param_fraction: float = 0.0
    shared_param_fraction: float = 1.0
    moe_dispatch_factor: float = 2.0  # out + back traffic multiplier per token

    @property
    def flops_per_token(self) -> float:
        return 2.0 * self.active_params

    @property
    def total_cached_tokens(self) -> int:
        return self.context_length + self.generation_window

    @property
    def is_moe(self) -> bool:
        return (
            self.experts_per_layer > 0
            and self.active_experts > 0
            and self.expert_param_fraction > 0
        )

    @property
    def active_params(self) -> float:
        if not self.is_moe:
            return self.num_params
        shared = self.num_params * self.shared_param_fraction
        expert_pool = self.num_params * self.expert_param_fraction
        expert_active = expert_pool * (self.active_experts / self.experts_per_layer)
        return shared + expert_active


GPU_PRESETS: Dict[str, GPUConfig] = {
    "MI300X": GPUConfig(
        "AMD MI300X",
        sustained_flops=11.5e12,
        hbm_bw=1.9e12,
        max_mem_bytes=192e9,
    ),
    "MI355X": GPUConfig(
        "AMD MI355X",
        sustained_flops=14.5e12,
        hbm_bw=2.4e12,
        max_mem_bytes=192e9,
    ),
    "GB200": GPUConfig(
        "NVIDIA GB200",
        sustained_flops=24e12,
        hbm_bw=3.4e12,
        max_mem_bytes=192e9,
    ),
    "GB300": GPUConfig(
        "NVIDIA GB300",
        sustained_flops=32e12,
        hbm_bw=4.3e12,
        max_mem_bytes=228e9,
    ),
}


RACK_PRESETS: List[RackPreset] = [
    RackPreset(
        name="NVIDIA GB200 NVL72",
        gpu_key="GB200",
        gpus_per_server=8,
        servers_per_rack=9,
        intra_server_bw=1.8e12,     # NVLink 5 per GPU
        inter_server_bw=0.9e12,     # NVSwitch within NVL72
        inter_rack_bw=0.45e12,      # InfiniBand / NVLink Switch System
        notes="NVL cabinet with 9 servers × 8 GB200 GPUs.",
    ),
    RackPreset(
        name="NVIDIA GB300 NVL72",
        gpu_key="GB300",
        gpus_per_server=8,
        servers_per_rack=9,
        intra_server_bw=2.1e12,
        inter_server_bw=1.05e12,
        inter_rack_bw=0.6e12,
        notes="Future Blackwell-class NVL72 cabinet (estimates).",
    ),
    RackPreset(
        name="AMD MI300X OAM Rack",
        gpu_key="MI300X",
        gpus_per_server=8,
        servers_per_rack=8,
        intra_server_bw=0.95e12,    # XGMI (8 links, ~= 896 GB/s)
        inter_server_bw=0.45e12,    # Infinity Fabric over optics
        inter_rack_bw=0.25e12,      # IB/HDR or 400G-class optics
        notes="4U OAM trays with Infinity Fabric interconnect.",
    ),
    RackPreset(
        name="AMD MI355X OAM Rack",
        gpu_key="MI355X",
        gpus_per_server=8,
        servers_per_rack=8,
        intra_server_bw=1.2e12,
        inter_server_bw=0.55e12,
        inter_rack_bw=0.3e12,
        notes="Projected MI355X fabric improvements.",
    ),
]


QUANT_PRESETS: Dict[int, QuantConfig] = {
    16: QuantConfig(bits=16, bytes_per_param=2.0, compute_scale=1.0, kv_bytes_per_elem=2.0),
    8: QuantConfig(bits=8, bytes_per_param=1.0, compute_scale=1.12, kv_bytes_per_elem=1.0),
    4: QuantConfig(bits=4, bytes_per_param=0.5, compute_scale=1.18, kv_bytes_per_elem=0.75),
}


MODEL = ModelConfig(
    name="Qwen3-235B-A22B",
    num_params=235e9,
    num_layers=94,
    hidden_size=1536,
    num_heads=64,
    context_length=4096,  # up to 256K-token long-context understanding
    generation_window=1024,
    experts_per_layer=128,
    active_experts=8,
    expert_param_fraction=0.88,   # majority of params live in experts
    shared_param_fraction=0.12,   # router + shared FFNs + attention
    moe_dispatch_factor=2.0,      # send + gather
)


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Simulation kernel
# ---------------------------------------------------------------------------


def evaluate_plan(
    rack: RackPreset,
    quant: QuantConfig,
    model: ModelConfig,
    batch_size: int,
    tp: int,
    pp: int,
    ep: int,
    racks: int,
) -> Optional[Dict[str, float]]:
    if model.num_layers % pp != 0 or model.num_heads % tp != 0:
        return None
    if model.is_moe and ep <= 0:
        return None
    if model.is_moe and model.experts_per_layer % ep != 0:
        return None

    gpu = GPU_PRESETS[rack.gpu_key]
    total_gpus = racks * rack.servers_per_rack * rack.gpus_per_server
    gpus_per_instance = tp * pp * max(ep, 1)
    if gpus_per_instance > total_gpus:
        return None

    mem_ok, mem_usage, mem_breakdown = fits_memory(
        gpu, model, quant, tp, pp, ep, batch_size
    )
    if not mem_ok:
        return None

    eff = batch_efficiency(batch_size)
    gpu_total_flops = gpu.sustained_flops * quant.compute_scale

    # Compute-bound
    compute_bound = (gpus_per_instance * gpu_total_flops) / model.flops_per_token
    compute_tps = compute_bound * eff

    # HBM-bound
    kv_read = kv_bytes_per_token(model, quant) / (tp * pp)
    kv_read *= 1.1  # read amplification (cache misses, paging)
    hbm_bound = gpu.hbm_bw / kv_read
    hbm_tps = hbm_bound * eff

    # Communication-bound
    network_bytes = {
        "intra": 0.0,
        "inter_server": 0.0,
        "inter_rack": 0.0,
    }

    tp_comm = tensor_parallel_comm_bytes(model, tp, pp)
    if tp_comm > 0.0:
        tp_servers = math.ceil(tp / rack.gpus_per_server)
        tp_racks = math.ceil(tp_servers / rack.servers_per_rack)
        if tp_servers <= 1:
            network_bytes["intra"] += tp_comm
        elif tp_racks <= 1:
            network_bytes["inter_server"] += tp_comm
        else:
            network_bytes["inter_rack"] += tp_comm

    pp_comm = pipeline_comm_bytes(model, pp)
    moe_comm = 0.0

    if pp_comm > 0.0:
        servers_per_stage = math.ceil(tp / rack.gpus_per_server)
        servers_needed = servers_per_stage * pp
        racks_needed = math.ceil(servers_needed / rack.servers_per_rack)
        if racks_needed <= 1:
            # Stage boundaries live inside a rack; assume server-to-server traffic dominates.
            if servers_per_stage <= 1 and rack.gpus_per_server >= tp:
                network_bytes["inter_server"] += pp_comm
            else:
                network_bytes["inter_server"] += pp_comm
        else:
            network_bytes["inter_rack"] += pp_comm

    if model.is_moe and ep > 1:
        moe_comm = moe_dispatch_bytes(model)
        ep_servers = math.ceil(ep / rack.gpus_per_server)
        ep_racks = math.ceil(ep_servers / rack.servers_per_rack)
        if ep_servers <= 1:
            network_bytes["intra"] += moe_comm
        elif ep_racks <= 1:
            network_bytes["inter_server"] += moe_comm
        else:
            network_bytes["inter_rack"] += moe_comm

    # Force inter-rack comm if the instance cannot fit in a single rack
    gpus_per_rack = rack.gpus_per_server * rack.servers_per_rack
    if gpus_per_instance > gpus_per_rack:
        network_bytes["inter_rack"] += tp_comm + pp_comm + moe_comm

    comm_limits: List[float] = []
    if network_bytes["intra"] > 0:
        comm_limits.append((rack.intra_server_bw / network_bytes["intra"]) * eff)
    if network_bytes["inter_server"] > 0:
        comm_limits.append((rack.inter_server_bw / network_bytes["inter_server"]) * eff)
    if network_bytes["inter_rack"] > 0:
        comm_limits.append((rack.inter_rack_bw / network_bytes["inter_rack"]) * eff)

    comm_tps = min(comm_limits) if comm_limits else float("inf")

    instance_tps = min(compute_tps, hbm_tps, comm_tps)
    limiting = "compute"
    if instance_tps == hbm_tps:
        limiting = "hbm"
    if instance_tps == comm_tps:
        limiting = "comm"

    instances_possible = total_gpus // gpus_per_instance
    if instances_possible == 0:
        return None

    total_tps = instance_tps * instances_possible
    tokens_per_gpu = instance_tps / gpus_per_instance
    compute_bound_per_gpu = compute_bound / gpus_per_instance
    hbm_bound_per_gpu = hbm_bound / gpus_per_instance
    comm_bound_raw = min(comm_limits) / eff if comm_limits else float("inf")
    comm_bound_per_gpu = comm_bound_raw / gpus_per_instance

    return {
        "instance_tps": instance_tps,
        "total_tps": total_tps,
        "tokens_per_gpu": tokens_per_gpu,
        "limit": limiting,
        "instances": instances_possible,
        "gpus_per_instance": gpus_per_instance,
        "ep": ep,
        "tp": tp,
        "pp": pp,
        "memory_used": mem_usage,
        "memory_breakdown": mem_breakdown,
        "network_bytes": network_bytes,
        "compute_tps": compute_tps,
        "hbm_tps": hbm_tps,
        "comm_tps": comm_tps,
        "compute_bound": compute_bound,
        "hbm_bound": hbm_bound,
        "comm_bound": comm_bound_raw,
        "compute_bound_per_gpu": compute_bound_per_gpu,
        "hbm_bound_per_gpu": hbm_bound_per_gpu,
        "comm_bound_per_gpu": comm_bound_per_gpu,
    }


def simulate_rack(
    rack: RackPreset,
    model: ModelConfig,
    racks: int,
    batch_sizes: Iterable[int],
    quant_bits: Iterable[int],
    tp_candidates: Iterable[int],
    pp_candidates: Iterable[int],
    ep_candidates: Iterable[int],
) -> Dict[int, Dict[int, Dict[str, float]]]:
    results: Dict[int, Dict[int, Dict[str, float]]] = {}
    for bits in quant_bits:
        quant = QUANT_PRESETS[bits]
        results[bits] = {}
        for batch in batch_sizes:
            best: Optional[Dict[str, float]] = None
            for tp in tp_candidates:
                if model.num_heads % tp != 0:
                    continue
                for pp in pp_candidates:
                    if model.num_layers % pp != 0:
                        continue
                    eps = ep_candidates if model.is_moe else [1]
                    for ep in eps:
                        plan = evaluate_plan(rack, quant, model, batch, tp, pp, ep, racks)
                        if not plan:
                            continue
                        if not best or plan["total_tps"] > best["total_tps"]:
                            best = plan
            if best:
                results[bits][batch] = best
    return results


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


def plot_results(
    rack: RackPreset,
    sim_results: Dict[int, Dict[int, Dict[str, float]]],
    racks: int,
    output_dir: Path,
) -> Optional[Path]:
    if not sim_results:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    plotted = False
    for bits in sorted(sim_results.keys()):
        batches = sorted(sim_results[bits].keys())
        if not batches:
            continue
        totals = [sim_results[bits][batch]["total_tps"] for batch in batches]
        ax.plot(batches, totals, marker="o", label=f"{bits}-bit")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    gpu = GPU_PRESETS[rack.gpu_key]
    total_gpus = racks * rack.servers_per_rack * rack.gpus_per_server

    ax.set_title(f"{rack.name}\n{gpu.name} – {total_gpus} GPUs across {racks} racks")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Total tokens/sec (8 racks)")
    ax.grid(True, alpha=0.2)
    ax.legend()

    output_path = output_dir / f"{slugify(rack.name)}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def print_summary(rack: RackPreset, sim_results: Dict[int, Dict[int, Dict[str, float]]], racks: int) -> None:
    gpu = GPU_PRESETS[rack.gpu_key]
    total_gpus = racks * rack.servers_per_rack * rack.gpus_per_server
    print("=" * 80)
    print(f"{rack.name}  |  GPU: {gpu.name}")
    print(f"Servers/rack: {rack.servers_per_rack}, GPUs/server: {rack.gpus_per_server}, racks: {racks}")
    print(f"Total GPUs: {total_gpus}")
    print(
        f"Interconnects (GB/s): intra={format_gbps(rack.intra_server_bw):.1f}, "
        f"inter-server={format_gbps(rack.inter_server_bw):.1f}, "
        f"inter-rack={format_gbps(rack.inter_rack_bw):.1f}"
    )
    if rack.notes:
        print(f"Notes: {rack.notes}")

    header = (
        "  Batch | Quant | Tok/GPU(avg) | Inst TPS | Tot TPS | #Inst | GPUs/Inst | TPxPPxEP | "
        "Mem/GPU (GB) | Limit | Fabric Load per Inst (GB/s intra/inter/rack) | Bounds tok/GPU(avg) (comp/hbm/comm)"
    )
    print(header)
    print("-" * len(header))

    for bits in sorted(sim_results.keys()):
        batches = sim_results[bits]
        for batch in sorted(batches.keys()):
            plan = batches[batch]
            mem_gb = human_gbytes(plan["memory_used"])
            net_bytes = plan["network_bytes"]
            intra_gbs = format_gbps(net_bytes["intra"] * plan["instance_tps"])
            inter_gbs = format_gbps(net_bytes["inter_server"] * plan["instance_tps"])
            rack_gbs = format_gbps(net_bytes["inter_rack"] * plan["instance_tps"])
            print(
                f"  {batch:5d} | {bits:5d} | {plan['tokens_per_gpu']:13.2f} | "
                f"{plan['instance_tps']:10.2f} | {plan['total_tps']:10.1f} | "
                f"{plan['instances']:5d} | {plan['gpus_per_instance']:9d} | "
                f"{plan['tp']}x{plan['pp']}x{plan['ep']} | {mem_gb:11.1f} | "
                f"{plan['limit']:>6} | {intra_gbs:5.1f}/{inter_gbs:5.1f}/{rack_gbs:5.1f} | "
                f"{format_bound(plan['compute_bound_per_gpu'])}/"
                f"{format_bound(plan['hbm_bound_per_gpu'])}/"
                f"{format_bound(plan['comm_bound_per_gpu'])}"
            )
    print()


def main() -> None:
    racks = 8
    batch_sizes = [1, 2, 4, 8, 10, 12, 16]
    quant_bits = [4, 8, 16]
    tp_candidates = [1, 2, 4, 8, 12, 16, 24, 32]
    pp_candidates = [1, 2, 3, 4, 6, 8]
    ep_candidates = [1, 2, 4, 8]

    print(f"Model: {MODEL.name} ({MODEL.num_params/1e9:.0f}B params)")
    print(f"Layers: {MODEL.num_layers}, hidden size: {MODEL.hidden_size}, heads: {MODEL.num_heads}")
    print(f"Context tokens cached: {MODEL.total_cached_tokens}")
    print(f"Evaluating {racks} racks per preset.")
    print()

    plot_dir = Path("plots")
    generated_plots: List[Tuple[str, Path]] = []

    for rack in RACK_PRESETS:
        sim_results = simulate_rack(
            rack=rack,
            model=MODEL,
            racks=racks,
            batch_sizes=batch_sizes,
            quant_bits=quant_bits,
            tp_candidates=tp_candidates,
            pp_candidates=pp_candidates,
            ep_candidates=ep_candidates,
        )
        print_summary(rack, sim_results, racks)
        plot_path = plot_results(rack, sim_results, racks, plot_dir)
        if plot_path:
            generated_plots.append((rack.name, plot_path))

    if generated_plots:
        print("Saved plots:")
        for name, path in generated_plots:
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
