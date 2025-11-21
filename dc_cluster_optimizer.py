#!/usr/bin/env python3
"""
Cluster-level optimizer for GB200 inference racks.

Given a target number of active users, desired KV cache per user, and the
model to be served, the optimizer finds the smallest number of GB200 racks
that meet concurrency, throughput, and KV storage requirements.

It reuses the rack/tensor-parallel planner from `llm_cluster_simulator.py`,
then reports a tailored summary plus ±10% sensitivity plots for users, batch
size, and KV cache demand.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_cluster_simulator import (
    DEFAULT_INFERENCE_BATCHES,
    MODEL_SELECTION_ALIASES,
    evaluate_plan,
    simulate_rack,
)
from model import ModelConfig
from presets import MODEL_PRESETS, QUANT_PRESETS, RACK_PRESETS
from rack import RackPreset
from utils import (
    DEFAULT_HBM_RESIDENCY,
    NVME_FALLBACK_BANDWIDTH,
    kv_bytes_per_token,
    human_gbytes,
    slugify,
)
from kv_subsystem import get_kv_system, kv_system_choices
from dc_cluster_visualizer import render_cluster_diagram

INFERENCE_KV_AMPLIFICATION = 1.1

def _parse_model_choice(raw: str) -> str:
    if raw in MODEL_PRESETS:
        return raw
    alias_key = raw.strip().lower()
    if alias_key in MODEL_SELECTION_ALIASES:
        return MODEL_SELECTION_ALIASES[alias_key]
    raise argparse.ArgumentTypeError(
        f"Unknown model '{raw}'. Use 1=qwen3-235b-a22b, 2=deepseek-v3.2-exp-685b, 3=chatgpt5-1p5t."
    )


def _parse_int_list(raw: Optional[str], default: Iterable[int]) -> List[int]:
    if not raw:
        return list(default)
    values: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError as exc:  # pragma: no cover - defensive path
            raise argparse.ArgumentTypeError(
                f"Invalid integer value '{chunk}'"
            ) from exc
    if not values:
        return list(default)
    return sorted(set(values))


def _rack_slug(name: str) -> str:
    return slugify(name)


def _find_rack_for_gpu(gpu_key: str, rack_choice: Optional[str]) -> RackPreset:
    if rack_choice:
        target_slug = _rack_slug(rack_choice)
        target_name = rack_choice.strip().lower()
        for rack in RACK_PRESETS:
            if rack.name.lower() == target_name or _rack_slug(rack.name) == target_slug:
                return rack
        options = ", ".join(rack.name for rack in RACK_PRESETS)
        raise ValueError(f"No rack preset named '{rack_choice}'. Available options: {options}")

    matches = [rack for rack in RACK_PRESETS if rack.gpu_key.lower() == gpu_key.lower()]
    if not matches:
        raise ValueError(f"No rack preset found for GPU '{gpu_key}'.")
    return matches[0]


def _override_tokens(model: ModelConfig, total_cached_tokens: int) -> ModelConfig:
    if total_cached_tokens <= 0:
        return model
    base_context = min(model.context_length, total_cached_tokens)
    remaining = max(total_cached_tokens - base_context, 0)
    return replace(model, context_length=base_context, generation_window=remaining)


def _kv_system_for_rack(rack: RackPreset, override: Optional[str]):
    key = override or getattr(rack, "kv_system_key", None)
    if not key:
        return None
    return get_kv_system(key)


def _compute_kv_profile(
    required_users: int,
    tokens_per_user_rate: float,
    tokens_per_user_cache: int,
    plan: Dict[str, float],
    quant_bytes_per_token: float,
    active_gpus: int,
    kv_system,
) -> Dict[str, float]:
    total_tokens = required_users * tokens_per_user_cache
    total_kv_bytes = total_tokens * quant_bytes_per_token
    active_gpus = max(1, active_gpus)
    hbm_per_gpu = plan["memory_breakdown"].get("kv_cache", 0.0)
    total_hbm_bytes = hbm_per_gpu * active_gpus
    remaining = max(0.0, total_kv_bytes - total_hbm_bytes)

    kv_capacity_per_gpu = kv_system.kv_capacity_bytes if kv_system else 0.0
    total_kv_cache_bytes = min(remaining, kv_capacity_per_gpu * active_gpus)
    total_nvme_bytes = max(0.0, remaining - total_kv_cache_bytes)

    remote_fraction = (
        total_kv_cache_bytes / total_kv_bytes if total_kv_bytes > 0 else 0.0
    )
    nvme_fraction = (
        total_nvme_bytes / total_kv_bytes if total_kv_bytes > 0 else 0.0
    )

    tokens_per_gpu_sec = (required_users * tokens_per_user_rate) / active_gpus
    kv_bytes_per_token_per_gpu = quant_bytes_per_token / (plan["tp"] * plan["pp"])
    per_user_bytes = kv_bytes_per_token_per_gpu * tokens_per_user_cache
    dma_bw_per_gpu = (
        tokens_per_gpu_sec
        * kv_bytes_per_token_per_gpu
        * remote_fraction
        * INFERENCE_KV_AMPLIFICATION
    )
    nvme_bw_per_gpu = (
        tokens_per_gpu_sec
        * kv_bytes_per_token_per_gpu
        * nvme_fraction
        * INFERENCE_KV_AMPLIFICATION
    )

    dma_link_limit = 0.0
    if kv_system:
        dma_link_limit = kv_system.path("gpu", "kv_dram").bandwidth_bytes
    dma_reload_ms = (
        (per_user_bytes * remote_fraction) / dma_link_limit * 1e3
        if dma_link_limit > 0 and remote_fraction > 0.0
        else 0.0
    )
    nvme_reload_ms = (
        (per_user_bytes * nvme_fraction) / NVME_FALLBACK_BANDWIDTH * 1e3
        if nvme_fraction > 0.0
        else 0.0
    )

    return {
        "total_kv_bytes": total_kv_bytes,
        "total_hbm_bytes": total_hbm_bytes,
        "total_cache_bytes": total_kv_cache_bytes,
        "total_nvme_bytes": total_nvme_bytes,
        "remote_fraction": remote_fraction,
        "nvme_fraction": nvme_fraction,
        "dma_bw_per_gpu": dma_bw_per_gpu,
        "nvme_bw_per_gpu": nvme_bw_per_gpu,
        "dma_link_limit": dma_link_limit,
        "per_user_bytes": per_user_bytes,
        "dma_reload_ms": dma_reload_ms,
        "nvme_reload_ms": nvme_reload_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Size GB200 racks for LLM inference workloads."
    )
    parser.add_argument(
        "--active-users",
        type=int,
        required=True,
        help="Concurrent interactive users to serve.",
    )
    parser.add_argument(
        "--model",
        type=_parse_model_choice,
        default="qwen3-235b-a22b",
        help="Model preset or alias (1=qwen, 2=deepseek, 3=chatgpt).",
    )
    parser.add_argument(
        "--tokens-cached",
        type=int,
        help="Total KV tokens (prompt + generation) retained per user.",
    )
    parser.add_argument(
        "--tokens-per-user-rate",
        type=float,
        default=12.0,
        help="Decode tokens/sec requested per active user (default 12 tok/s).",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=8,
        choices=sorted(QUANT_PRESETS.keys()),
        help="Quantisation precision for serving weights.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="GB200",
        help="GPU preset key (defaults to GB200).",
    )
    parser.add_argument(
        "--rack-preset",
        type=str,
        help=(
            "Optional rack preset name/slug (e.g. 'NVIDIA GB200 NVL72'). "
            "Defaults to the first rack that matches --gpu."
        ),
    )
    parser.add_argument(
        "--max-racks",
        type=int,
        default=16,
        help="Upper bound on racks to evaluate.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        help="Comma-separated batch sizes to search (defaults to inference presets).",
    )
    parser.add_argument(
        "--fixed-racks",
        type=int,
        help=(
            "Force the optimizer to evaluate exactly this many racks, even if it undershoots the "
            "requested concurrency / throughput. Useful for studying oversubscribed clusters."
        ),
    )
    parser.add_argument(
        "--prefer-utilization",
        action="store_true",
        help=(
            "Select the rack plan that maximizes GPU compute utilization instead of peak throughput. "
            "Useful when you want to keep clusters smaller and lean on KV offload."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/optimizer"),
        help="Directory for sensitivity plots.",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        help="Average number of new users arriving per minute (Poisson process).",
    )
    parser.add_argument(
        "--avg-session-min",
        type=float,
        help="Average session length in minutes for each user (exponential).",
    )
    parser.add_argument(
        "--simulation-minutes",
        type=int,
        default=240,
        help="Length of the arrival simulation window in minutes (default 240).",
    )
    parser.add_argument(
        "--warmup-minutes",
        type=int,
        default=30,
        help="Discard the initial minutes to remove cold-start artifacts (default 30).",
    )
    parser.add_argument(
        "--arrival-seed",
        type=int,
        help="Random seed for the arrival simulator (optional).",
    )
    parser.add_argument(
        "--kv-system",
        type=str,
        choices=kv_system_choices(),
        help="Override the rack's KV tier (choices: {}).".format(
            ", ".join(kv_system_choices())
        ),
    )
    parser.add_argument(
        "--hbm-residency",
        type=float,
        default=DEFAULT_HBM_RESIDENCY,
        help=(
            "Fraction of users' KV cache resident in HBM at any time (0-1). "
            "Remaining contexts are phased to system memory or NVMe; default 0.50."
        ),
    )
    return parser.parse_args()


def _select_plan(
    rack: RackPreset,
    model: ModelConfig,
    racks: int,
    quant_bits: int,
    batch_sizes: Iterable[int],
    required_users: int,
    required_tps: float,
    prefer_utilization: bool,
    allow_shortfall: bool,
    kv_system,
    hbm_residency: float,
) -> Optional[Tuple[int, Dict[str, float]]]:
    sim_results = simulate_rack(
        rack=rack,
        model=model,
        racks=racks,
        batch_sizes=batch_sizes,
        quant_bits=[quant_bits],
        tp_candidates=[1, 2, 4, 8, 12, 16, 24, 32],
        pp_candidates=[1, 2, 3, 4, 6, 8],
        ep_candidates=[1, 2, 4, 8],
        mode="inference",
        kv_system=kv_system,
        hbm_residency=hbm_residency,
    )
    if quant_bits not in sim_results:
        return None

    best_choice: Optional[Tuple[int, Dict[str, float]]] = None
    best_metric: Optional[float] = None
    for batch in sorted(sim_results[quant_bits].keys()):
        plan = sim_results[quant_bits][batch]
        concurrency_capacity = plan["instances"] * batch
        total_capacity_tps = plan["total_tps"]
        if not allow_shortfall:
            if concurrency_capacity < required_users:
                continue
            if total_capacity_tps < required_tps:
                continue
        else:
            if concurrency_capacity <= 0 or total_capacity_tps <= 0:
                continue
        if prefer_utilization and not allow_shortfall:
            if total_capacity_tps <= 0:
                continue
            compute_util = required_tps / total_capacity_tps
            utilization_score = compute_util
            should_update = (
                best_metric is None
                or utilization_score > best_metric + 1e-9
                or (
                    math.isclose(utilization_score, best_metric, rel_tol=1e-9, abs_tol=1e-9)
                    and total_capacity_tps < best_choice[1]["total_tps"]
                )
            )
            if should_update:
                best_metric = utilization_score
                best_choice = (batch, plan)
        else:
            if not best_choice or total_capacity_tps > best_choice[1]["total_tps"]:
                best_choice = (batch, plan)
                best_metric = total_capacity_tps
    return best_choice


def _kv_capacity_per_gpu(plan: Dict[str, float]) -> float:
    mem = plan.get("memory_breakdown", {})
    limit = mem.get("limit", 0.0)
    weights = mem.get("weights", 0.0)
    overhead = mem.get("overhead", 0.0)
    # Any leftover budget belongs to the KV cache tier.
    return max(limit - weights - overhead, 1.0)


def _network_label(plan: Dict[str, float]) -> str:
    loads = plan.get("network_loads", {})
    inter_bytes = loads.get("inter_server", {}).get("bytes_per_token", 0.0)
    rack_bytes = loads.get("inter_rack", {}).get("bytes_per_token", 0.0)
    if rack_bytes > 1e-9:
        return "Inter-rack InfiniBand / optical spine"
    if inter_bytes > 1e-9:
        return "NVSwitch / NVLink Switch"
    return "Intra-server NVLink"


def _util_stats(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


def _bw_utilization(usage: float, limit: float) -> float:
    if usage <= 0.0:
        return 0.0
    if limit <= 0.0:
        return 1.0
    return min(1.0, usage / limit)


def _format_gib(value_bytes: float) -> str:
    return f"{human_gbytes(value_bytes):.1f} GiB"


def _plot_util_curve(
    x_values: List[float],
    series: Dict[str, List[float]],
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for label, values in series.items():
        if not values:
            continue
        ax.plot(x_values, values, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Utilization (0-1)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _print_summary_table(rows: List[Tuple[str, str]]) -> None:
    col_width = max(len(name) for name, _ in rows) + 2
    print("\nSummary Table")
    print("-" * (col_width + 24))
    for name, value in rows:
        print(f"{name:<{col_width}}{value}")


def simulate_active_users(
    arrival_rate_per_min: float,
    avg_session_minutes: float,
    total_minutes: int,
    warmup_minutes: int,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    total_minutes = max(total_minutes, warmup_minutes + 1)
    rng = np.random.default_rng(seed)
    active_sessions: List[float] = []
    samples: List[float] = []

    for minute in range(total_minutes):
        active_sessions = [end for end in active_sessions if end > minute]
        arrivals = int(rng.poisson(arrival_rate_per_min))
        if arrivals > 0:
            durations = rng.exponential(avg_session_minutes, size=arrivals)
            active_sessions.extend(minute + durations)
        if minute >= warmup_minutes:
            samples.append(len(active_sessions))

    if not samples:
        samples = [len(active_sessions)]

    sample_arr = np.array(samples, dtype=float)
    return {
        "mean": float(np.mean(sample_arr)),
        "p95": float(np.percentile(sample_arr, 95)),
        "max": float(np.max(sample_arr)),
        "timeline": samples,
    }


def main() -> None:
    args = parse_args()

    try:
        rack = _find_rack_for_gpu(args.gpu, args.rack_preset)
    except ValueError as exc:
        raise SystemExit(str(exc))
    model = MODEL_PRESETS[args.model]
    if args.tokens_cached:
        model = _override_tokens(model, args.tokens_cached)
    quant_bits = args.quant_bits
    quant = QUANT_PRESETS[quant_bits]
    kv_system = None
    kv_system_key = args.kv_system or getattr(rack, "kv_system_key", None)
    if kv_system_key:
        try:
            kv_system = get_kv_system(kv_system_key)
        except KeyError as exc:
            raise SystemExit(str(exc))
    hbm_residency = max(0.0, min(args.hbm_residency, 1.0))

    batch_sizes = _parse_int_list(args.batch_sizes, DEFAULT_INFERENCE_BATCHES)
    required_users = max(1, args.active_users)
    tokens_per_user_rate = max(0.1, args.tokens_per_user_rate)
    required_tps = required_users * tokens_per_user_rate
    arrival_stats: Optional[Dict[str, float]] = None

    arrival_args = (args.arrival_rate is not None, args.avg_session_min is not None)
    if any(arrival_args) and not all(arrival_args):
        raise SystemExit(
            "Please provide both --arrival-rate and --avg-session-min to enable the stochastic arrival model."
        )
    if all(arrival_args):
        arrival_stats = simulate_active_users(
            arrival_rate_per_min=args.arrival_rate,
            avg_session_minutes=args.avg_session_min,
            total_minutes=max(args.simulation_minutes, 1),
            warmup_minutes=max(0, args.warmup_minutes),
            seed=args.arrival_seed,
        )
        stochastic_required = max(1, math.ceil(arrival_stats["p95"]))
        required_users = max(required_users, stochastic_required)
        required_tps = required_users * tokens_per_user_rate

    allow_shortfall = args.fixed_racks is not None
    if args.fixed_racks is not None and args.fixed_racks < 1:
        raise SystemExit("--fixed-racks must be >= 1.")
    rack_counts: Iterable[int]
    if args.fixed_racks is not None:
        rack_counts = [args.fixed_racks]
        if args.fixed_racks > args.max_racks:
            args.max_racks = args.fixed_racks
    else:
        rack_counts = range(1, args.max_racks + 1)

    best_selection: Optional[Tuple[int, int, Dict[str, float]]] = None
    for rack_count in rack_counts:
        choice = _select_plan(
            rack=rack,
            model=model,
            racks=rack_count,
            quant_bits=quant_bits,
            batch_sizes=batch_sizes,
            required_users=required_users,
            required_tps=required_tps,
            prefer_utilization=args.prefer_utilization,
            allow_shortfall=allow_shortfall,
            kv_system=kv_system,
            hbm_residency=hbm_residency,
        )
        if choice:
            batch_size, plan = choice
            best_selection = (rack_count, batch_size, plan)
            break

    if not best_selection:
        raise SystemExit(
            "Unable to find a feasible GB200 configuration up to the requested rack limit."
        )

    racks_needed, batch_size, plan = best_selection
    provisioned_gpus = rack.servers_per_rack * rack.gpus_per_server * racks_needed
    instances_available = plan["instances"]
    gpus_per_instance = plan["gpus_per_instance"]
    concurrency_capacity = instances_available * batch_size
    instances_required = min(
        instances_available, max(1, math.ceil(required_users / batch_size))
    )
    gpus_required = max(gpus_per_instance, instances_required * gpus_per_instance)
    throughput_capacity = plan["total_tps"]
    servers_needed = math.ceil(gpus_required / rack.gpus_per_server)
    racks_computed = math.ceil(servers_needed / rack.servers_per_rack)
    per_rack_gpu_capacity = rack.servers_per_rack * rack.gpus_per_server
    active_distribution: List[int] = []
    remaining_gpu = gpus_required
    for _ in range(racks_needed):
        taken = min(per_rack_gpu_capacity, remaining_gpu)
        active_distribution.append(taken)
        remaining_gpu -= taken

    kv_bytes = kv_bytes_per_token(model, quant)
    tokens_per_user_cache = model.total_cached_tokens
    kv_usage_per_gpu = plan["memory_breakdown"].get("kv_cache", 0.0)
    kv_capacity_per_gpu = _kv_capacity_per_gpu(plan)
    kv_usage_total = kv_usage_per_gpu * gpus_required
    kv_capacity_total = kv_capacity_per_gpu * gpus_required
    scenario_scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    storage_util_samples = [
        min(1.0, (kv_usage_total * scale) / kv_capacity_total) for scale in scenario_scales
    ]
    concurrency_shortfall = max(0, required_users - concurrency_capacity)
    throughput_shortfall = max(0.0, required_tps - throughput_capacity)

    kv_profile = _compute_kv_profile(
        required_users=required_users,
        tokens_per_user_rate=tokens_per_user_rate,
        tokens_per_user_cache=tokens_per_user_cache,
        plan=plan,
        quant_bytes_per_token=kv_bytes,
        active_gpus=gpus_required,
        kv_system=kv_system,
    )
    kv_dma_util_samples: List[float] = []
    nvme_util_samples: List[float] = []
    for scale in scenario_scales:
        scaled_users = max(1, int(round(required_users * scale)))
        scaled_instances = min(
            instances_available, max(1, math.ceil(scaled_users / batch_size))
        )
        scaled_gpus = max(gpus_per_instance, scaled_instances * gpus_per_instance)
        scaled_profile = _compute_kv_profile(
            required_users=scaled_users,
            tokens_per_user_rate=tokens_per_user_rate,
            tokens_per_user_cache=tokens_per_user_cache,
            plan=plan,
            quant_bytes_per_token=kv_bytes,
            active_gpus=scaled_gpus,
            kv_system=kv_system,
        )
        kv_dma_util_samples.append(
            _bw_utilization(
                scaled_profile["dma_bw_per_gpu"], scaled_profile["dma_link_limit"]
            )
        )
        nvme_util_samples.append(
            _bw_utilization(
                scaled_profile["nvme_bw_per_gpu"], NVME_FALLBACK_BANDWIDTH
            )
        )

    compute_limit_total = plan["compute_tps"] * instances_available
    network_limit_total = plan["comm_tps"] * instances_available
    compute_utils = [
        min(1.0, (required_tps * scale) / compute_limit_total) for scale in (0.9, 1.0, 1.1)
    ]
    if math.isinf(network_limit_total):
        network_utils = [0.0, 0.0, 0.0]
    else:
        network_utils = [
            min(1.0, (required_tps * scale) / network_limit_total)
            for scale in (0.9, 1.0, 1.1)
        ]

    comp_avg, comp_p95 = _util_stats(compute_utils)
    net_avg, net_p95 = _util_stats(network_utils)
    stor_avg, stor_p95 = _util_stats(storage_util_samples)
    kv_dma_avg, kv_dma_p95 = _util_stats(kv_dma_util_samples)
    nvme_avg, nvme_p95 = _util_stats(nvme_util_samples)

    total_storage_bytes = kv_profile["total_kv_bytes"]
    storage_tb = total_storage_bytes / 1e12
    remote_storage_bytes = kv_profile["total_cache_bytes"] + kv_profile["total_nvme_bytes"]
    remote_storage_tb = remote_storage_bytes / 1e12
    storage_server_capacity = rack.storage_server_capacity_bytes
    storage_capacity_tb = (
        storage_server_capacity / 1e12 if storage_server_capacity > 0 else 0.0
    )
    storage_servers_needed = (
        math.ceil(remote_storage_bytes / storage_server_capacity)
        if storage_server_capacity > 0 and remote_storage_bytes > 0
        else 0
    )
    storage_servers_available = racks_needed * rack.storage_servers_per_rack
    storage_capacity_available_tb = storage_servers_available * storage_capacity_tb
    storage_shortfall = max(0, storage_servers_needed - storage_servers_available)
    batches_per_gpu = batch_size / gpus_per_instance
    network_type = _network_label(plan)

    kv_label = kv_system.label if kv_system else "Host DRAM / NVMe spill"
    user_capacity_text = f"{required_users} ({concurrency_capacity} slots"
    if concurrency_shortfall > 0:
        user_capacity_text += f", shortfall {concurrency_shortfall}"
    user_capacity_text += ")"
    throughput_text = (
        f"{required_tps:.1f} / {throughput_capacity:.1f} tok/s"
    )
    if throughput_shortfall > 0:
        throughput_text += f" (shortfall {throughput_shortfall:.1f})"

    rows: List[Tuple[str, str]] = [
        ("Model", model.name),
        ("Rack preset", rack.name),
        ("Quantisation", f"{quant_bits}-bit"),
        ("Users supported (capacity)", user_capacity_text),
        ("KV system", kv_label),
        (
            "HBM KV residency",
            f"{hbm_residency*100:.0f}% on-GPU (phased users)",
        ),
    ]
    if arrival_stats:
        rows.append(
            (
                "Arrival model (mean/p95/max)",
                f"{arrival_stats['mean']:.1f} / {arrival_stats['p95']:.1f} / {arrival_stats['max']:.0f}",
            )
        )
        rows.append(
            (
                "Arrival inputs",
                f"{args.arrival_rate:.2f} users/min, avg session {args.avg_session_min:.1f} min",
            )
        )
    rows.extend(
        [
            ("Batch size per instance", str(batch_size)),
            (
                "TP x PP x EP",
                f"{plan['tp']} x {plan['pp']} x {plan['ep']}",
            ),
            ("Throughput demand / capacity", throughput_text),
            (
                "Servers needed",
                f"{servers_needed} (provisioned: {racks_needed * rack.servers_per_rack})",
            ),
            ("Racks needed", f"{max(racks_needed, racks_computed)} of type {rack.name}"),
            ("Storage needed (TB)", f"{storage_tb:.2f} TB KV tier"),
            (
                "Storage offloaded (TB)",
                f"{remote_storage_tb:.2f} TB outside GPU HBM",
            ),
            (
                "Storage servers needed",
                f"{storage_servers_needed} (available {storage_servers_available})",
            ),
            (
                "Storage capacity provisioned",
                f"{storage_capacity_available_tb:.1f} TB @ {rack.storage_servers_per_rack} per rack",
            ),
            ("Batches per GPU", f"{batches_per_gpu:.2f}"),
            ("Networking between servers", network_type),
            (
                "Compute util (avg / p95)",
                f"{comp_avg*100:.1f}% / {comp_p95*100:.1f}%",
            ),
            (
                "Network util (avg / p95)",
                f"{net_avg*100:.1f}% / {net_p95*100:.1f}%",
            ),
            (
                "Storage util (avg / p95)",
                f"{stor_avg*100:.1f}% / {stor_p95*100:.1f}%",
            ),
            (
                "KV placement (HBM / cache / NVMe)",
                f"{_format_gib(kv_profile['total_hbm_bytes'])} / "
                f"{_format_gib(kv_profile['total_cache_bytes'])} / "
                f"{_format_gib(kv_profile['total_nvme_bytes'])}",
            ),
        ]
    )
    if kv_system and kv_profile["dma_link_limit"] > 0:
        if kv_profile["dma_reload_ms"] > 0:
            rows.append(
                (
                    "KV reload latency (cache)",
                    f"{kv_profile['dma_reload_ms']:.2f} ms per user swap",
                )
            )
        rows.append(
            (
                "KV DMA util (avg / p95)",
                f"{kv_dma_avg*100:.1f}% / {kv_dma_p95*100:.1f}% "
                f"({kv_profile['dma_link_limit']/1e9:.0f} GB/s link)",
            )
        )
    if kv_profile["total_nvme_bytes"] > 0:
        if kv_profile["nvme_reload_ms"] > 0:
            rows.append(
                (
                    "NVMe reload latency",
                    f"{kv_profile['nvme_reload_ms']:.2f} ms per user swap",
                )
            )
        rows.append(
            (
                "NVMe spill util (avg / p95)",
                f"{nvme_avg*100:.1f}% / {nvme_p95*100:.1f}% "
                f"({NVME_FALLBACK_BANDWIDTH/1e9:.0f} GB/s per GPU)",
            )
        )
    _print_summary_table(rows)
    if concurrency_shortfall > 0:
        print(
            f"WARNING: concurrency shortfall of {concurrency_shortfall} users."
            " Increase rack count or reduce demand to avoid oversubscription."
        )
    if throughput_shortfall > 0:
        print(
            f"WARNING: throughput shortfall of {throughput_shortfall:.1f} tokens/sec "
            "relative to requested workload."
        )
    if storage_shortfall > 0:
        deficit_tb = storage_shortfall * storage_capacity_tb
        print(
            f"WARNING: storage shortfall of {storage_shortfall} servers (~{deficit_tb:.1f} TB)."
            " Add dedicated storage capacity or more racks to stay within limits."
        )

    print("\nPlan details")
    print("-" * 40)
    print(
        f"Provisioned GPUs: {provisioned_gpus} (racks {racks_needed}) | "
        f"Active GPUs: {gpus_required} | "
        f"Instances running: {instances_required}/{instances_available} | "
        f"Tokens/sec capacity: {plan['total_tps']:.1f}"
    )
    comm_bound = plan["comm_tps"]
    comm_text = "inf" if math.isinf(comm_bound) else f"{comm_bound:.1f}"
    print(
        f"Compute bound: {plan['compute_tps']:.1f} tok/s per instance | "
        f"HBM bound: {plan['hbm_tps']:.1f} tok/s | "
        f"Comm bound: {comm_text} tok/s"
    )
    if kv_system and kv_profile["dma_link_limit"] > 0:
        print(
            f"KV DMA link ({kv_system.label}): {_format_gib(kv_profile['total_cache_bytes'])} cached "
            f"| util ≈ {kv_dma_avg*100:.1f}% of {kv_profile['dma_link_limit']/1e9:.0f} GB/s"
        )
        if kv_profile["dma_reload_ms"] > 0:
            print(
                f"  Swap reload latency (cache): {kv_profile['dma_reload_ms']:.2f} ms per user"
            )
    if kv_profile["total_nvme_bytes"] > 0:
        print(
            f"NVMe spill: {_format_gib(kv_profile['total_nvme_bytes'])} "
            f"| util ≈ {nvme_avg*100:.1f}% of {NVME_FALLBACK_BANDWIDTH/1e9:.0f} GB/s per GPU"
        )
        if kv_profile["nvme_reload_ms"] > 0:
            print(
                f"  Swap reload latency (NVMe): {kv_profile['nvme_reload_ms']:.2f} ms per user"
            )
    if arrival_stats:
        print(
            f"Arrival model => mean {arrival_stats['mean']:.1f}, "
            f"P95 {arrival_stats['p95']:.1f}, max {arrival_stats['max']:.0f} concurrent users."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_users = [required_users * s for s in scenario_scales]
    user_series = {}
    if len(x_users) == len(compute_utils):
        user_series["Compute"] = compute_utils
    if len(x_users) == len(network_utils):
        user_series["Network"] = network_utils
    if len(x_users) == len(storage_util_samples):
        user_series["HBM storage"] = storage_util_samples
    if kv_system:
        if len(x_users) == len(kv_dma_util_samples):
            user_series["KV DMA"] = kv_dma_util_samples
    if any(val > 0 for val in nvme_util_samples) and len(x_users) == len(
        nvme_util_samples
    ):
        user_series["NVMe"] = nvme_util_samples
    _plot_util_curve(
        x_values=x_users,
        series=user_series,
        xlabel="Active users",
        title="Utilisation vs active users (±10%)",
        output_path=args.output_dir / "users_utilization.png",
    )

    batch_series = {"Compute": [], "Network": [], "Storage": []}
    batch_values: List[float] = []
    for scale in scenario_scales:
        new_batch = max(1, int(round(batch_size * scale)))
        eval_plan = evaluate_plan(
            rack=rack,
            quant=quant,
            model=model,
            batch_size=new_batch,
            tp=plan["tp"],
            pp=plan["pp"],
            ep=plan["ep"],
            racks=racks_needed,
            mode="inference",
            kv_system=kv_system,
            hbm_residency=hbm_residency,
        )
        if not eval_plan:
            continue
        batch_values.append(new_batch)
        compute_limit = eval_plan["compute_tps"] * eval_plan["instances"]
        network_limit = eval_plan["comm_tps"] * eval_plan["instances"]
        kv_cap = _kv_capacity_per_gpu(eval_plan)
        kv_use = eval_plan["memory_breakdown"].get("kv_cache", 0.0)
        storage_util = min(1.0, kv_use / kv_cap)
        batch_series["Compute"].append(
            min(1.0, required_tps / compute_limit)
        )
        if math.isinf(network_limit):
            batch_series["Network"].append(0.0)
        else:
            batch_series["Network"].append(
                min(1.0, required_tps / network_limit)
            )
        batch_series["Storage"].append(storage_util)

    if batch_values:
        _plot_util_curve(
            x_values=batch_values,
            series=batch_series,
            xlabel="Batch size per instance",
            title="Utilisation vs batch size (±10%)",
            output_path=args.output_dir / "batch_utilization.png",
        )

    token_series = {"Compute": [], "Network": [], "Storage": []}
    token_values: List[float] = []
    for scale in scenario_scales:
        new_total_tokens = max(1, int(round(tokens_per_user_cache * scale)))
        model_override = _override_tokens(model, new_total_tokens)
        eval_plan = evaluate_plan(
            rack=rack,
            quant=quant,
            model=model_override,
            batch_size=batch_size,
            tp=plan["tp"],
            pp=plan["pp"],
            ep=plan["ep"],
            racks=racks_needed,
            mode="inference",
            kv_system=kv_system,
            hbm_residency=hbm_residency,
        )
        if not eval_plan:
            continue
        token_values.append(model_override.total_cached_tokens)
        compute_limit = eval_plan["compute_tps"] * eval_plan["instances"]
        network_limit = eval_plan["comm_tps"] * eval_plan["instances"]
        kv_cap = _kv_capacity_per_gpu(eval_plan)
        kv_use = eval_plan["memory_breakdown"].get("kv_cache", 0.0)
        storage_util = min(1.0, kv_use / kv_cap)
        token_series["Compute"].append(
            min(1.0, required_tps / compute_limit)
        )
        if math.isinf(network_limit):
            token_series["Network"].append(0.0)
        else:
            token_series["Network"].append(
                min(1.0, required_tps / network_limit)
            )
        token_series["Storage"].append(storage_util)

    if token_values:
        _plot_util_curve(
            x_values=token_values,
            series=token_series,
            xlabel="Tokens cached per user",
            title="Utilisation vs KV cache demand (±10%)",
            output_path=args.output_dir / "tokens_utilization.png",
        )

    print("\nPlots saved to:", args.output_dir.resolve())

    diagram_path = args.output_dir / "cluster_diagram.png"

    summary_text = [
        f"Model: {model.name} | Rack: {rack.name}",
        f"Users: {required_users} (capacity {concurrency_capacity})",
        f"Racks: {racks_needed} | GPUs active {gpus_required}/{provisioned_gpus}",
        f"Batch size: {batch_size} | Plan {plan['tp']}x{plan['pp']}x{plan['ep']}",
        f"HBM KV residency: {hbm_residency*100:.0f}% on-GPU",
        (
            f"Storage servers needed {storage_servers_needed} (avail {storage_servers_available})"
            if racks_needed > 0
            else ""
        ),
    ]
    summary_text = [line for line in summary_text if line]
    kv_text = [
        f"KV placement HBM/cache/NVMe: "
        f"{human_gbytes(kv_profile['total_hbm_bytes']):.1f}/"
        f"{human_gbytes(kv_profile['total_cache_bytes']):.1f}/"
        f"{human_gbytes(kv_profile['total_nvme_bytes']):.1f} GiB",
        f"KV system: {kv_label}",
    ]
    per_rack_gpu_servers = [
        min(
            rack.servers_per_rack,
            max(0, math.ceil(gpus / max(1, rack.gpus_per_server))),
        )
        for gpus in active_distribution
    ]
    remaining_storage_servers = min(storage_servers_needed, storage_servers_available)
    storage_distribution: List[int] = []
    for _ in range(racks_needed):
        assign = min(rack.storage_servers_per_rack, remaining_storage_servers)
        storage_distribution.append(assign)
        remaining_storage_servers -= assign

    payload = {
        "rack_name": rack.name,
        "racks_needed": racks_needed,
        "servers_per_rack": rack.servers_per_rack,
        "gpus_per_server": rack.gpus_per_server,
        "active_gpus_per_rack": active_distribution,
        "summary_lines": summary_text,
        "kv_lines": kv_text,
        "control_servers": 2,
        "storage_servers": rack.storage_servers_per_rack,
        "gpu_servers": rack.servers_per_rack,
        "gpu_servers_per_rack": per_rack_gpu_servers,
        "storage_servers_per_rack": storage_distribution,
    }
    render_cluster_diagram(payload, diagram_path)
    print("Diagram saved to:", diagram_path.resolve())


if __name__ == "__main__":
    main()
