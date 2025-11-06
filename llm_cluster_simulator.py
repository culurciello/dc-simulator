#!/usr/bin/env python3
"""
Rack-level LLM throughput simulator.

The simulator models sustained decode throughput for Qwen3-235B across
multiple racks while accounting for
- quantized weight storage (4/8/16 bit)
- realistic KV-cache read bandwidth
- multi-level interconnect costs (GPU<->GPU, server<->server, rack<->rack)
- automatic search over tensor/pipeline parallelism plans

Outputs per-rack summaries for 8 racks of each supported hardware preset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from model import ModelConfig, QuantConfig
from presets import GPU_PRESETS, MODEL, QUANT_PRESETS, RACK_PRESETS
from rack import RackPreset
from utils import (
    batch_efficiency,
    CommPattern,
    fits_memory,
    fits_training_memory,
    format_bound,
    format_gbps,
    human_gbytes,
    kv_bytes_per_token,
    moe_dispatch,
    pipeline_comm,
    slugify,
    tensor_parallel_comm,
    train_efficiency,
    training_activation_bytes_per_token,
)

INFERENCE_KV_AMPLIFICATION = 1.1
TRAIN_COMPUTE_MULT = 6.0
TRAIN_HBM_MULT = 1.8
TRAIN_TP_COMM_MULT = 2.0
TRAIN_PP_COMM_MULT = 2.0
TRAIN_MOE_COMM_MULT = 2.0
TRAIN_GRAD_MULT = 2.0
TRAIN_GRAD_MESSAGES = 1.0

DEFAULT_INFERENCE_BATCHES = [1, 2, 4, 8, 10, 12, 16]
DEFAULT_TRAIN_BATCHES = [32, 64, 96, 128, 256, 512]


@dataclass
class LinkLoad:
    bytes_per_token: float = 0.0
    messages_per_token: float = 0.0

    def add_pattern(self, pattern: CommPattern) -> None:
        self.bytes_per_token += pattern.bytes_per_token
        self.messages_per_token += pattern.messages_per_token

    def add_load(self, other: "LinkLoad") -> None:
        self.bytes_per_token += other.bytes_per_token
        self.messages_per_token += other.messages_per_token

    def is_empty(self) -> bool:
        return self.bytes_per_token <= 0.0 and self.messages_per_token <= 0.0


def evaluate_plan(
    rack: RackPreset,
    quant: QuantConfig,
    model: ModelConfig,
    batch_size: int,
    tp: int,
    pp: int,
    ep: int,
    racks: int,
    mode: str,
) -> Optional[Dict[str, float]]:
    if model.num_layers % pp != 0 or model.num_heads % tp != 0:
        return None
    if model.is_moe and ep <= 0:
        return None
    if model.is_moe and model.experts_per_layer % ep != 0:
        return None

    training = mode.lower() == "training"

    gpu = GPU_PRESETS[rack.gpu_key]
    total_gpus = racks * rack.servers_per_rack * rack.gpus_per_server
    gpus_per_instance = tp * pp * max(ep, 1)
    if gpus_per_instance > total_gpus:
        return None

    if training:
        mem_ok, mem_usage, mem_breakdown = fits_training_memory(
            gpu, model, quant, tp, pp, ep, batch_size
        )
    else:
        mem_ok, mem_usage, mem_breakdown = fits_memory(
            gpu, model, quant, tp, pp, ep, batch_size
        )
    if not mem_ok:
        return None

    eff = train_efficiency(batch_size) if training else batch_efficiency(batch_size)
    gpu_total_flops = gpu.sustained_flops * quant.compute_scale

    # Compute-bound
    flops_per_token = model.flops_per_token * (TRAIN_COMPUTE_MULT if training else 1.0)
    compute_bound = (gpus_per_instance * gpu_total_flops) / flops_per_token
    compute_tps = compute_bound * eff

    # HBM-bound
    if training:
        base_bytes = training_activation_bytes_per_token(model, tp, pp)
        hbm_per_token = base_bytes * TRAIN_HBM_MULT
    else:
        hbm_per_token = kv_bytes_per_token(model, quant) / (tp * pp)
        hbm_per_token *= INFERENCE_KV_AMPLIFICATION  # read amplification (cache misses, paging)
    if hbm_per_token <= 0:
        return None
    hbm_bound = gpu.hbm_bw / hbm_per_token
    hbm_tps = hbm_bound * eff

    # Communication-bound
    link_loads = {
        "intra": LinkLoad(),
        "inter_server": LinkLoad(),
        "inter_rack": LinkLoad(),
    }
    link_limits_raw: Dict[str, float] = {}

    tp_comm = tensor_parallel_comm(model, tp, pp).scaled(
        TRAIN_TP_COMM_MULT if training else 1.0
    )
    if tp_comm.bytes_per_token > 0.0 or tp_comm.messages_per_token > 0.0:
        tp_servers = ceil(tp / rack.gpus_per_server)
        tp_racks = ceil(tp_servers / rack.servers_per_rack)
        if tp_servers <= 1:
            link_loads["intra"].add_pattern(tp_comm)
        elif tp_racks <= 1:
            link_loads["inter_server"].add_pattern(tp_comm)
        else:
            link_loads["inter_rack"].add_pattern(tp_comm)

    pp_comm = pipeline_comm(model, pp).scaled(
        TRAIN_PP_COMM_MULT if training else 1.0
    )
    if pp_comm.bytes_per_token > 0.0 or pp_comm.messages_per_token > 0.0:
        servers_per_stage = ceil(tp / rack.gpus_per_server)
        servers_needed = servers_per_stage * pp
        racks_needed = ceil(servers_needed / rack.servers_per_rack)
        if racks_needed <= 1:
            # Stage boundaries live inside a rack; assume server-to-server traffic dominates.
            link_loads["inter_server"].add_pattern(pp_comm)
        else:
            link_loads["inter_rack"].add_pattern(pp_comm)

    moe_comm = CommPattern(0.0, 0.0)
    if model.is_moe and ep > 1:
        moe_comm = moe_dispatch(model).scaled(
            TRAIN_MOE_COMM_MULT if training else 1.0
        )
        ep_servers = ceil(ep / rack.gpus_per_server)
        ep_racks = ceil(ep_servers / rack.servers_per_rack)
        if ep_servers <= 1:
            link_loads["intra"].add_pattern(moe_comm)
        elif ep_racks <= 1:
            link_loads["inter_server"].add_pattern(moe_comm)
        else:
            link_loads["inter_rack"].add_pattern(moe_comm)

    grad_comm_pattern = CommPattern(0.0, 0.0)
    dp_degree = max(1, total_gpus // gpus_per_instance)
    if training and dp_degree > 1:
        tokens_per_step = batch_size * model.context_length
        weights_per_gpu = mem_breakdown.get("weights", 0.0)
        if tokens_per_step > 0 and weights_per_gpu > 0:
            grad_bytes_per_token = (
                weights_per_gpu * TRAIN_GRAD_MULT / tokens_per_step
            )
            grad_comm_pattern = CommPattern(
                bytes_per_token=grad_bytes_per_token,
                messages_per_token=TRAIN_GRAD_MESSAGES,
            )
            if dp_degree <= rack.gpus_per_server:
                link_loads["intra"].add_pattern(grad_comm_pattern)
            elif dp_degree <= rack.servers_per_rack:
                link_loads["inter_server"].add_pattern(grad_comm_pattern)
            else:
                link_loads["inter_rack"].add_pattern(grad_comm_pattern)

    # Force inter-rack comm if the instance cannot fit in a single rack
    gpus_per_rack = rack.gpus_per_server * rack.servers_per_rack
    if gpus_per_instance > gpus_per_rack:
        link_loads["inter_rack"].add_pattern(tp_comm)
        link_loads["inter_rack"].add_pattern(pp_comm)
        link_loads["inter_rack"].add_pattern(moe_comm)

    comm_limits: List[float] = []
    comm_limits_raw: List[float] = []
    for kind, link in (
        ("intra", rack.intra_server),
        ("inter_server", rack.inter_server),
        ("inter_rack", rack.inter_rack),
    ):
        load = link_loads[kind]
        if load.is_empty():
            continue

        time_per_token = 0.0
        if load.bytes_per_token > 0.0 and link.throughput > 0.0:
            time_per_token += load.bytes_per_token / link.throughput
        if load.messages_per_token > 0.0 and link.latency > 0.0:
            time_per_token += load.messages_per_token * link.latency

        if time_per_token <= 0.0:
            continue

        base_limit = 1.0 / time_per_token
        link_limits_raw[kind] = base_limit
        comm_limits_raw.append(base_limit)
        comm_limits.append(base_limit * eff)

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
    comm_bound_raw = min(comm_limits_raw) if comm_limits_raw else float("inf")
    comm_bound_per_gpu = comm_bound_raw / gpus_per_instance

    tokens_per_step = batch_size * model.context_length if training else None
    steps_per_sec = 0.0
    total_steps_per_sec = 0.0
    samples_per_sec = 0.0
    total_samples_per_sec = 0.0
    if training and tokens_per_step and tokens_per_step > 0:
        steps_per_sec = instance_tps / tokens_per_step
        total_steps_per_sec = steps_per_sec * instances_possible
        samples_per_sec = steps_per_sec * batch_size
        total_samples_per_sec = total_steps_per_sec * batch_size

    network_loads = {
        kind: {
            "bytes_per_token": load.bytes_per_token,
            "messages_per_token": load.messages_per_token,
        }
        for kind, load in link_loads.items()
    }

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
        "network_loads": network_loads,
        "link_limits_raw": link_limits_raw,
        "compute_tps": compute_tps,
        "hbm_tps": hbm_tps,
        "comm_tps": comm_tps,
        "compute_bound": compute_bound,
        "hbm_bound": hbm_bound,
        "comm_bound": comm_bound_raw,
        "compute_bound_per_gpu": compute_bound_per_gpu,
        "hbm_bound_per_gpu": hbm_bound_per_gpu,
        "comm_bound_per_gpu": comm_bound_per_gpu,
        "mode": mode,
        "dp_degree": dp_degree,
        "steps_per_sec": steps_per_sec if training else None,
        "total_steps_per_sec": total_steps_per_sec if training else None,
        "samples_per_sec": samples_per_sec if training else None,
        "total_samples_per_sec": total_samples_per_sec if training else None,
        "tokens_per_step": tokens_per_step,
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
    mode: str,
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
                        plan = evaluate_plan(
                            rack, quant, model, batch, tp, pp, ep, racks, mode
                        )
                        if not plan:
                            continue
                        if not best or plan["total_tps"] > best["total_tps"]:
                            best = plan
            if best:
                results[bits][batch] = best
    return results


def plot_results(
    rack: RackPreset,
    sim_results: Dict[int, Dict[int, Dict[str, float]]],
    racks: int,
    output_dir: Path,
    mode: str,
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
        if mode.lower() == "training":
            totals = [
                sim_results[bits][batch].get("total_samples_per_sec", 0.0)
                for batch in batches
            ]
        else:
            totals = [sim_results[bits][batch]["total_tps"] for batch in batches]
        ax.plot(batches, totals, marker="o", label=f"{bits}-bit")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    gpu = GPU_PRESETS[rack.gpu_key]
    total_gpus = racks * rack.servers_per_rack * rack.gpus_per_server

    ax.set_title(f"{rack.name}\n{gpu.name} - {total_gpus} GPUs across {racks} racks")
    ax.set_xlabel("Batch size")
    if mode.lower() == "training":
        ax.set_ylabel(f"Total samples/sec ({racks} racks, training)")
    else:
        ax.set_ylabel(f"Total tokens/sec ({racks} racks)")
    ax.grid(True, alpha=0.2)
    ax.legend()

    output_suffix = "training" if mode.lower() == "training" else "inference"
    output_path = output_dir / f"{slugify(rack.name)}_{output_suffix}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def print_summary(
    rack: RackPreset,
    sim_results: Dict[int, Dict[int, Dict[str, float]]],
    racks: int,
    mode: str,
) -> None:
    gpu = GPU_PRESETS[rack.gpu_key]
    total_gpus = racks * rack.servers_per_rack * rack.gpus_per_server
    print("=" * 80)
    print(f"{rack.name}  |  GPU: {gpu.name}")
    print(
        f"Servers/rack: {rack.servers_per_rack}, GPUs/server: {rack.gpus_per_server}, racks: {racks}"
    )
    print(f"Total GPUs: {total_gpus}")
    print(
        "Interconnects: "
        f"intra={format_gbps(rack.intra_server.throughput):.1f} GB/s @ "
        f"{rack.intra_server.latency * 1e6:.2f} us, "
        f"inter-server={format_gbps(rack.inter_server.throughput):.1f} GB/s @ "
        f"{rack.inter_server.latency * 1e6:.2f} us, "
        f"inter-rack={format_gbps(rack.inter_rack.throughput):.1f} GB/s @ "
        f"{rack.inter_rack.latency * 1e6:.2f} us"
    )
    if rack.notes:
        print(f"Notes: {rack.notes}")

    training = mode.lower() == "training"
    if training:
        header = (
            "  Batch | Quant | Tok/GPU(avg) | Inst TPS | Tot TPS | Inst Samples/s | Tot Samples/s | "
            "#Inst | GPUs/Inst | TPxPPxEP | Mem/GPU (GB) | Limit | "
            "Fabric Load per Inst (GB/s intra/inter/rack) | "
            "Msg Load per Inst (kmsg/s intra/inter/rack) | "
            "Bounds tok/GPU(avg) (comp/hbm/comm)"
        )
    else:
        header = (
            "  Batch | Quant | Tok/GPU(avg) | Inst TPS | Tot TPS | #Inst | GPUs/Inst | TPxPPxEP | "
            "Mem/GPU (GB) | Limit | Fabric Load per Inst (GB/s intra/inter/rack) | "
            "Msg Load per Inst (kmsg/s intra/inter/rack) | "
            "Bounds tok/GPU(avg) (comp/hbm/comm)"
        )
    print(header)
    print("-" * len(header))

    for bits in sorted(sim_results.keys()):
        batches = sim_results[bits]
        for batch in sorted(batches.keys()):
            plan = batches[batch]
            mem_gb = human_gbytes(plan["memory_used"])
            net_loads = plan["network_loads"]
            intra_gbs = format_gbps(
                net_loads["intra"]["bytes_per_token"] * plan["instance_tps"]
            )
            inter_gbs = format_gbps(
                net_loads["inter_server"]["bytes_per_token"] * plan["instance_tps"]
            )
            rack_gbs = format_gbps(
                net_loads["inter_rack"]["bytes_per_token"] * plan["instance_tps"]
            )
            intra_kmsg = (
                net_loads["intra"]["messages_per_token"] * plan["instance_tps"] / 1e3
            )
            inter_kmsg = (
                net_loads["inter_server"]["messages_per_token"] * plan["instance_tps"]
                / 1e3
            )
            rack_kmsg = (
                net_loads["inter_rack"]["messages_per_token"] * plan["instance_tps"]
                / 1e3
            )
            if training:
                inst_samples = plan.get("samples_per_sec") or 0.0
                total_samples = plan.get("total_samples_per_sec") or 0.0
                print(
                    f"  {batch:5d} | {bits:5d} | {plan['tokens_per_gpu']:13.2f} | "
                    f"{plan['instance_tps']:10.2f} | {plan['total_tps']:10.1f} | "
                    f"{inst_samples:13.2f} | {total_samples:12.2f} | "
                    f"{plan['instances']:5d} | {plan['gpus_per_instance']:9d} | "
                    f"{plan['tp']}x{plan['pp']}x{plan['ep']} | {mem_gb:11.1f} | "
                    f"{plan['limit']:>6} | "
                    f"{intra_gbs:5.1f}/{inter_gbs:5.1f}/{rack_gbs:5.1f} | "
                    f"{intra_kmsg:5.1f}/{inter_kmsg:5.1f}/{rack_kmsg:5.1f} | "
                    f"{format_bound(plan['compute_bound_per_gpu'])}/"
                    f"{format_bound(plan['hbm_bound_per_gpu'])}/"
                    f"{format_bound(plan['comm_bound_per_gpu'])}"
                )
            else:
                print(
                    f"  {batch:5d} | {bits:5d} | {plan['tokens_per_gpu']:13.2f} | "
                    f"{plan['instance_tps']:10.2f} | {plan['total_tps']:10.1f} | "
                    f"{plan['instances']:5d} | {plan['gpus_per_instance']:9d} | "
                    f"{plan['tp']}x{plan['pp']}x{plan['ep']} | {mem_gb:11.1f} | "
                    f"{plan['limit']:>6} | "
                    f"{intra_gbs:5.1f}/{inter_gbs:5.1f}/{rack_gbs:5.1f} | "
                    f"{intra_kmsg:5.1f}/{inter_kmsg:5.1f}/{rack_kmsg:5.1f} | "
                    f"{format_bound(plan['compute_bound_per_gpu'])}/"
                    f"{format_bound(plan['hbm_bound_per_gpu'])}/"
                    f"{format_bound(plan['comm_bound_per_gpu'])}"
                )
    print()
def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer value '{chunk}'")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rack-level LLM simulator")
    parser.add_argument(
        "--mode",
        choices=["inference", "training"],
        default="inference",
        help="Select inference (decode) or training mode.",
    )
    parser.add_argument(
        "--racks",
        type=int,
        default=8,
        help="Number of identical racks to simulate per preset.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        help="Comma-separated list of batch sizes to evaluate.",
    )
    parser.add_argument(
        "--quant-bits",
        type=str,
        help="Comma-separated list of quantisation bits (e.g. '4,8,16').",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable plot generation (useful on headless or slow environments).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode.lower()
    racks = max(1, args.racks)

    custom_batches = _parse_int_list(args.batch_sizes)
    batch_sizes = (
        custom_batches
        if custom_batches
        else (DEFAULT_TRAIN_BATCHES if mode == "training" else DEFAULT_INFERENCE_BATCHES)
    )

    custom_bits = _parse_int_list(args.quant_bits)
    quant_bits = custom_bits if custom_bits else [4, 8, 16]

    # tp is the tensor-parallel shard count (how many GPUs split each layer's weights)
    tp_candidates = [1, 2, 4, 8, 12, 16, 24, 32]
    # pp is the pipeline-parallel depth (how many model stages run in sequence on separate GPU groups)
    pp_candidates = [1, 2, 3, 4, 6, 8]
    # ep is the expert-parallel fan-out used when the MoE layers shard their experts across GPUs
    ep_candidates = [1, 2, 4, 8]

    print(f"Mode: {mode}")
    print(f"Model: {MODEL.name} ({MODEL.num_params/1e9:.0f}B params)")
    print(
        f"Layers: {MODEL.num_layers}, hidden size: {MODEL.hidden_size}, heads: {MODEL.num_heads}"
    )
    if mode == "inference":
        print(f"Context tokens cached: {MODEL.total_cached_tokens}")
    else:
        print("Training context length per sample:", MODEL.context_length)
    print(f"Evaluating {racks} racks per preset.")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Quant bits: {quant_bits}")
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
            mode=mode,
        )
        print_summary(rack, sim_results, racks, mode)
        if args.skip_plots:
            continue
        plot_path = plot_results(rack, sim_results, racks, plot_dir, mode)
        if plot_path:
            generated_plots.append((rack.name, plot_path))

    if generated_plots:
        print("Saved plots:")
        for name, path in generated_plots:
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
