#!/usr/bin/env python3
"""
KV-cache offloading bandwidth estimator for the GB200 NVL stack.

The script projects how much host-device bandwidth would be required to
maintain a large cached context when weights run at 4-bit and the KV cache
spills out of HBM. It reuses the shared model/preset definitions from the
cluster simulator so the sizing stays consistent with the rest of the repo.

For each requested context length it reports:
- how many tokens can remain resident in HBM (after weights + overhead)
- how many tokens must be streamed from host memory
- the resulting per-GPU eviction bandwidth (GPU -> host)
- the host read bandwidth under a pessimistic “stream the entire offloaded
  slice every decode step” assumption, plus a gentler amortised view where
  the offloaded slice is fetched once per full generation.

Finally it plots the sustained bandwidth requirement (worst-case and
amortised) versus the cached token count and writes it under plots/.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import random
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from model import ModelConfig, QuantConfig
from presets import GPU_PRESETS, MODEL_PRESETS, QUANT_PRESETS
from kv_subsystem import (
    KVLinkProfile,
    get_kv_system,
    kv_system_choices,
    resolve_kv_system_key,
)
from utils import (
    batch_efficiency,
    DEFAULT_HBM_RESIDENCY,
    fits_memory,
    human_gbytes,
    kv_bytes_per_token,
    NVME_FALLBACK_BANDWIDTH,
)

INFERENCE_KV_AMPLIFICATION = 1.1
SYSTEM_STYLES = {
    "gb200_host": {"color": "tab:orange", "marker": "s", "linestyle": "--"},
    "gb300": {"color": "tab:green", "marker": "^", "linestyle": "-."},
}


@dataclass
class OffloadResult:
    tokens: int
    model_name: str
    tokens_cached: float
    tokens_on_gpu: float
    tokens_offloaded: float
    offload_fraction: float
    available_kv_bytes: float
    total_gpu_memory_bytes: float
    usable_memory_bytes: float
    kv_bytes_per_token_total: float
    kv_bytes_per_token_per_gpu: float
    tokens_per_sec_instance: float
    tokens_per_sec_per_gpu: float
    evict_bw: float
    worst_read_bw: float
    amortized_read_bw: float
    mem_ok: bool
    kv_tier_capacity_bytes: float
    kv_path_bandwidth_bytes: float

    @property
    def total_worst_bw(self) -> float:
        return self.evict_bw + self.worst_read_bw

    @property
    def total_amortized_bw(self) -> float:
        return self.evict_bw + self.amortized_read_bw

    @property
    def offloaded_bytes_per_gpu(self) -> float:
        return self.tokens_offloaded * self.kv_bytes_per_token_per_gpu

    @property
    def kv_tier_utilization(self) -> float:
        if self.kv_tier_capacity_bytes <= 0.0:
            return 0.0
        return min(1.0, self.offloaded_bytes_per_gpu / self.kv_tier_capacity_bytes)



def parse_tokens(arg: str) -> List[int]:
    raw = arg.split(",")
    tokens: List[int] = []
    for item in raw:
        entry = item.strip().lower()
        if not entry:
            continue
        multiplier = 1
        if entry.endswith("k"):
            multiplier = 1_000
            entry = entry[:-1]
        value = int(float(entry) * multiplier)
        tokens.append(value)
    if not tokens:
        raise ValueError("token list must not be empty")
    return tokens


def parse_single_token(arg: str) -> int:
    values = parse_tokens(arg)
    if len(values) != 1:
        raise ValueError("expected a single token value, got multiple entries")
    return values[0]


def parse_kv_system_keys(arg: str) -> List[str]:
    entries = [item.strip().lower() for item in arg.split(",") if item.strip()]
    if not entries:
        return [resolve_kv_system_key("gb200_plain")]
    resolved: List[str] = []
    for entry in entries:
        try:
            resolved.append(resolve_kv_system_key(entry))
        except KeyError as exc:
            raise ValueError(
                f"Unknown KV system '{entry}'. Choices: {', '.join(kv_system_choices())}"
            ) from exc
    return resolved


def path_label(path: KVLinkProfile) -> str:
    return path.notes if path.notes else path.label


def kv_path_keys(kv_system: KVCacheSystem) -> List[str]:
    return [path_label(p) for p in kv_system.kv_data_paths()]


def simulate_offload(
    tokens_list: Iterable[int],
    batch_size: int,
    tp: int,
    pp: int,
    ep: int,
    model: ModelConfig,
    quant: QuantConfig,
    kv_system: KVCacheSystem,
    hbm_residency: float,
) -> List[OffloadResult]:
    gpu = GPU_PRESETS["GB200"]
    gpus_per_instance = tp * pp * max(ep, 1)
    kv_capacity_bytes = kv_system.kv_capacity_bytes
    kv_dma_path = kv_system.path("gpu", "kv_dram")
    residency = max(0.0, min(hbm_residency, 1.0))

    results: List[OffloadResult] = []

    for tokens in tokens_list:
        model_variant = replace(model, context_length=tokens, generation_window=0)
        mem_ok, _, breakdown = fits_memory(
            gpu,
            model_variant,
            quant,
            tp=tp,
            pp=pp,
            ep=ep,
            batch_size=batch_size,
            hbm_residency=residency,
        )

        kv_total = kv_bytes_per_token(model_variant, quant)
        kv_per_gpu = kv_total / (tp * pp)
        tokens_cached = model_variant.total_cached_tokens * batch_size

        limit = breakdown["limit"]
        weights = breakdown["weights"]
        overhead = breakdown["overhead"]
        available_kv_bytes = max(0.0, limit - weights - overhead)
        max_resident_tokens = tokens_cached * residency

        if kv_per_gpu > 0:
            tokens_on_gpu = min(max_resident_tokens, available_kv_bytes / kv_per_gpu)
        else:
            tokens_on_gpu = max_resident_tokens

        tokens_offloaded = max(0.0, tokens_cached - tokens_on_gpu)
        offload_fraction = (
            tokens_offloaded / tokens_cached if tokens_cached > 0 else 0.0
        )

        eff = batch_efficiency(batch_size)
        gpu_total_flops = gpu.sustained_flops * quant.compute_scale
        flops_per_token = model_variant.flops_per_token

        compute_bound = (gpus_per_instance * gpu_total_flops) / flops_per_token
        compute_tps = compute_bound * eff

        kv_hbm = kv_per_gpu * residency * INFERENCE_KV_AMPLIFICATION
        kv_remote = (
            kv_per_gpu * (1.0 - residency) * INFERENCE_KV_AMPLIFICATION
        )
        remote_bw = kv_dma_path.bandwidth_bytes or NVME_FALLBACK_BANDWIDTH

        hbm_time = kv_hbm / gpu.hbm_bw if kv_hbm > 0 else 0.0
        remote_time = kv_remote / remote_bw if kv_remote > 0 else 0.0
        access_time = hbm_time + remote_time
        hbm_bound = 1.0 / access_time if access_time > 0 else float("inf")
        hbm_tps = hbm_bound * eff

        tokens_per_sec_instance = min(compute_tps, hbm_tps)
        tokens_per_sec_per_gpu = (
            tokens_per_sec_instance / gpus_per_instance if gpus_per_instance > 0 else 0.0
        )

        evict_bw = tokens_per_sec_per_gpu * kv_per_gpu
        worst_read_bw = tokens_per_sec_per_gpu * tokens_offloaded * kv_per_gpu
        amortized_read_bw = 0.0
        if tokens_per_sec_instance > 0.0 and tokens_cached > 0.0:
            generation_time = tokens_cached / tokens_per_sec_instance
            amortized_read_bw = (
                tokens_offloaded * kv_per_gpu / generation_time
                if generation_time > 0.0
                else 0.0
            )

        results.append(
            OffloadResult(
                tokens=tokens,
                model_name=model.name,
                tokens_cached=tokens_cached,
                tokens_on_gpu=tokens_on_gpu,
                tokens_offloaded=tokens_offloaded,
                offload_fraction=offload_fraction,
                available_kv_bytes=available_kv_bytes,
                total_gpu_memory_bytes=gpu.max_mem_bytes,
                usable_memory_bytes=limit,
                kv_bytes_per_token_total=kv_total,
                kv_bytes_per_token_per_gpu=kv_per_gpu,
                tokens_per_sec_instance=tokens_per_sec_instance,
                tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
                evict_bw=evict_bw,
                worst_read_bw=worst_read_bw,
                amortized_read_bw=amortized_read_bw,
                mem_ok=mem_ok,
                kv_tier_capacity_bytes=kv_capacity_bytes,
                kv_path_bandwidth_bytes=kv_dma_path.bandwidth_bytes,
            )
        )

    return results


def format_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.0f}k"
    return f"{value:.0f}"


def print_results(
    results: List[OffloadResult],
    batch_size: int,
    tp: int,
    pp: int,
    ep: int,
    show_worst_case_delay: bool,
    model: ModelConfig,
    quant_bits: int,
    kv_system: KVCacheSystem,
    hbm_residency: float,
) -> None:
    gpu = GPU_PRESETS["GB200"]
    params_str = f"{model.num_params/1e9:.1f}B"
    print(f"KV offloading requirements (per GPU unless noted) | KV system: {kv_system.label}")
    print(
        "Batch size: {} | TPxPPxEP: {}x{}x{} | Quant: {}-bit | GPU: {}".format(
            batch_size, tp, pp, ep, quant_bits, gpu.name
        )
    )
    if results:
        total_mem_gib = human_gbytes(results[0].total_gpu_memory_bytes)
        usable_mem_gib = human_gbytes(results[0].usable_memory_bytes)
        available_mem_gib = human_gbytes(results[0].available_kv_bytes)
        print(
            "HBM per GPU: {:.2f} GiB total | {:.2f} GiB usable (post safety margin) | {:.2f} GiB available for KV cache".format(
                total_mem_gib, usable_mem_gib, available_mem_gib
            )
        )
    print(
        f"KV residency in HBM: {hbm_residency*100:.0f}% (phased users; remainder offloaded)"
    )
    print(
        "Model parameters: {} | Decode batch size per GPU: {}".format(
            params_str, batch_size
        )
    )
    print("KV cache fabric description:")
    for line in kv_system.describe():
        print(f"  {line}")
    headers = (
        "Tokens",
        "On-GPU",
        "Offloaded",
        "Offload%",
        "Tok/s (inst)",
        "Tok/s (GPU)",
        "Evict GB/s",
        "Worst-read GB/s",
        "Total worst GB/s",
        "Total amort GB/s",
    )
    print(
        "{:>10} {:>10} {:>11} {:>9} {:>12} {:>12} {:>12} {:>15} {:>16} {:>17}".format(
            *headers
        )
    )
    for res in results:
        tokens_label = format_tokens(res.tokens)
        on_gpu_tokens = format_tokens(res.tokens_on_gpu)
        offloaded_tokens = format_tokens(res.tokens_offloaded)
        offload_pct = res.offload_fraction * 100.0
        evict_gbs = res.evict_bw / 1e9
        worst_read_gbs = res.worst_read_bw / 1e9
        total_worst_gbs = res.total_worst_bw / 1e9
        total_amort_gbs = res.total_amortized_bw / 1e9
        print(
            "{:>10} {:>10} {:>11} {:>8.1f}% {:>12.0f} {:>12.1f} {:>12.2f} {:>15.2f} {:>16.2f} {:>17.2f}".format(
                tokens_label,
                on_gpu_tokens,
                offloaded_tokens,
                offload_pct,
                res.tokens_per_sec_instance,
                res.tokens_per_sec_per_gpu,
                evict_gbs,
                worst_read_gbs,
                total_worst_gbs,
                total_amort_gbs,
            )
        )
        offloaded_bytes = res.tokens_offloaded * res.kv_bytes_per_token_per_gpu
        generation_time = (
            res.tokens_cached / res.tokens_per_sec_instance
            if res.tokens_per_sec_instance > 0.0
            else float("inf")
        )
        link_delay_parts = []
        for path in kv_system.kv_data_paths():
            path_name = path_label(path)
            if offloaded_bytes > 0.0 and path.bandwidth_bytes > 0.0:
                amortized_extra_s = kv_system.transfer_time_seconds(
                    offloaded_bytes, path
                )
                worst_extra_s = (
                    kv_system.transfer_time_seconds(
                        offloaded_bytes * res.tokens_cached, path
                    )
                    if show_worst_case_delay and res.tokens_cached > 0.0
                    else 0.0
                )
            else:
                amortized_extra_s = 0.0
                worst_extra_s = 0.0

            amortized_ms = amortized_extra_s * 1e3
            if generation_time > 0.0 and not math.isinf(generation_time):
                amortized_pct = (amortized_extra_s / generation_time) * 100.0
            else:
                amortized_pct = 0.0

            entry = (
                f"{path_name}: +{amortized_ms:.2f} ms ({amortized_pct:.2f}%) amortized"
            )

            if show_worst_case_delay:
                worst_ms = worst_extra_s * 1e3
                if generation_time > 0.0 and not math.isinf(generation_time):
                    worst_pct = (worst_extra_s / generation_time) * 100.0
                else:
                    worst_pct = 0.0
                entry += f"; worst-case +{worst_ms:.2f} ms ({worst_pct:.2f}%)"

            link_delay_parts.append(entry)
        link_delay_str = " | ".join(link_delay_parts)
        if show_worst_case_delay:
            delay_label = "KV fabric latency (amortized + worst-case)"
        else:
            delay_label = "KV fabric latency (amortized)"
        if res.tokens_offloaded > 0.0:
            print(
                "  -> Fits in HBM? {} | Available KV budget: {:.2f} GiB | KV/token shard: {:.2f} KiB".format(
                    "yes" if res.mem_ok else "no",
                    human_gbytes(res.available_kv_bytes),
                    res.kv_bytes_per_token_per_gpu / 1024.0,
                )
            )
            print(f"     {delay_label}: {link_delay_str}")
        else:
            print(
                "  -> Fully resident in HBM (available KV budget: {:.2f} GiB)".format(
                    human_gbytes(res.available_kv_bytes)
                )
            )
            print(f"     {delay_label}: {link_delay_str}")
        kv_usage_gib = res.offloaded_bytes_per_gpu / (1024**3)
        kv_capacity_tib = res.kv_tier_capacity_bytes / (1024**4)
        kv_usage_pct = res.kv_tier_utilization * 100.0
        dma_gbs = res.kv_path_bandwidth_bytes / 1e9
        print(
            "     KV tier usage: {:.2f} GiB ({:.3f}% of {:.1f} TiB total) | GPU↔KV DMA {:.1f} GB/s".format(
                kv_usage_gib,
                kv_usage_pct,
                kv_capacity_tib,
                dma_gbs,
            )
        )
    print()


def plot_comparison(
    system_runs: List[Tuple[str, List[OffloadResult], KVCacheSystem]],
    base_output: Path,
    batch_size: int,
    show_worst_case_delay: bool,
    model: ModelConfig,
    quant_bits: int,
) -> None:
    if not system_runs:
        return
    compare_path = base_output
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    gpu = GPU_PRESETS["GB200"]
    params_str = f"{model.num_params/1e9:.1f}B"
    fig, (ax_bw, ax_delay) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)

    reference_tokens = [res.tokens / 1_000.0 for res in system_runs[0][1]]
    series_payloads: List[Tuple[str, List[float], List[float], Optional[List[float]]]] = []

    for key, results, kv_system in system_runs:
        label = kv_system.label
        style = SYSTEM_STYLES.get(
            key, {"color": "tab:gray", "marker": "D", "linestyle": "-."}
        )
        tokens_k = [res.tokens / 1_000.0 for res in results]
        if tokens_k != reference_tokens:
            raise ValueError("All KV systems must be evaluated with identical token lists for comparison plots.")
        total_amort = [res.total_amortized_bw / 1e9 for res in results]
        ax_bw.plot(
            tokens_k,
            total_amort,
            label=f"{label} total host traffic",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
        )
        ax_bw.scatter(
            tokens_k,
            total_amort,
            color=style["color"],
            marker=style["marker"],
            s=28,
            alpha=0.85,
        )

        dma_path = kv_system.path("gpu", "kv_dram")
        path_name = path_label(dma_path)
        amortized_ms = []
        worst_ms = []
        for res in results:
            offloaded_bytes = res.tokens_offloaded * res.kv_bytes_per_token_per_gpu
            delay = (
                kv_system.transfer_time_seconds(offloaded_bytes, dma_path) * 1e3
                if offloaded_bytes > 0.0
                else 0.0
            )
            amortized_ms.append(delay)
            if show_worst_case_delay and offloaded_bytes > 0.0 and res.tokens_cached > 0.0:
                worst_delay = kv_system.transfer_time_seconds(
                    offloaded_bytes * res.tokens_cached, dma_path
                )
                worst_ms.append(worst_delay * 1e3)
            elif show_worst_case_delay:
                worst_ms.append(0.0)
        ax_delay.plot(
            tokens_k,
            amortized_ms,
            label=f"{label} {path_name} amortized",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
        )
        ax_delay.scatter(
            tokens_k,
            amortized_ms,
            color=style["color"],
            marker=style["marker"],
            s=30,
            alpha=0.85,
        )
        if show_worst_case_delay:
            ax_delay.plot(
                tokens_k,
                worst_ms,
                linestyle="--",
                marker=None,
                label=f"{label} {path_name} worst-case",
                color=style["color"],
                linewidth=1.2,
                alpha=0.7,
            )
        series_payloads.append(
            (label, total_amort, amortized_ms, worst_ms if show_worst_case_delay else None)
        )

    ax_bw.set_ylabel("Bandwidth per GPU (GB/s)")
    ax_bw.set_title(
        f"KV offload comparison ({gpu.name}) @ {quant_bits}-bit | {params_str} params | batch {batch_size}/GPU"
    )
    ax_bw.grid(True, linestyle="--", alpha=0.3)
    ax_bw.legend()

    ax_delay.set_xlabel("Cached tokens per batch (thousands)")
    ax_delay.set_ylabel("DMA reload latency (ms)")
    ax_delay.grid(True, linestyle="--", alpha=0.3)
    ax_delay.legend()

    fig.tight_layout()
    fig.savefig(compare_path, dpi=160)
    plt.close(fig)
    print(f"Saved comparison plot: {compare_path}")
    write_plot_data_xls(reference_tokens, series_payloads, compare_path)


def write_plot_data_xls(
    tokens_k: List[float],
    series_payloads: List[Tuple[str, List[float], List[float], Optional[List[float]]]],
    plot_path: Path,
) -> None:
    if not series_payloads or not tokens_k:
        return
    xls_path = plot_path.with_suffix(".xls")
    header = ["tokens_k"]
    for label, _, _, worst_series in series_payloads:
        base = label.replace(" ", "_")
        header.append(f"{base}_host_gbps")
        header.append(f"{base}_dma_ms")
        if worst_series is not None:
            header.append(f"{base}_dma_worst_ms")
    xls_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xls_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["# Tab-separated values exported for spreadsheet use (Excel-readable)."]
        )
        writer.writerow(header)
        for idx, token in enumerate(tokens_k):
            row: List[float] = [token]
            for _, host_series, amort_series, worst_series in series_payloads:
                row.append(host_series[idx])
                row.append(amort_series[idx])
                if worst_series is not None:
                    row.append(worst_series[idx])
            writer.writerow(row)
    print(f"Saved comparison data: {xls_path}")


def compute_path_reload_times(
    result: OffloadResult, kv_system: KVCacheSystem
) -> Dict[str, float]:
    times: Dict[str, float] = {}
    for path in kv_system.kv_data_paths():
        key = path_label(path)
        if result.offloaded_bytes_per_gpu > 0.0 and path.bandwidth_bytes > 0.0:
            times[key] = kv_system.transfer_time_seconds(result.offloaded_bytes_per_gpu, path)
        else:
            times[key] = 0.0
    return times


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    fraction = rank - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * fraction


def simulate_random_user_swaps(
    qwen_result: OffloadResult,
    deepseek_result: OffloadResult,
    steps: int,
    seed: int,
    kv_system: KVCacheSystem,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    rng = random.Random(seed)
    models = [qwen_result, deepseek_result]
    reload_tables = {
        qwen_result.model_name: compute_path_reload_times(qwen_result, kv_system),
        deepseek_result.model_name: compute_path_reload_times(deepseek_result, kv_system),
    }
    path_keys = kv_path_keys(kv_system)
    link_delay_log = {key: [] for key in path_keys}
    current = rng.choice(models)
    swap_events = 0

    for _ in range(steps):
        target = rng.choice(models)
        swapped = target is not current
        if swapped:
            swap_events += 1
            target_table = reload_tables[target.model_name]
        for key in path_keys:
            delay = target_table[key] if swapped else 0.0
            link_delay_log[key].append(delay)
        current = target

    summary: Dict[str, Dict[str, float]] = {}
    for link_name, delays in link_delay_log.items():
        avg_ms = (sum(delays) / steps) * 1e3
        p95_ms = percentile(delays, 95.0) * 1e3
        summary[link_name] = {"avg_ms": avg_ms, "p95_ms": p95_ms}

    swap_probability = swap_events / steps if steps > 0 else 0.0
    return summary, swap_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate KV cache offloading bandwidth on GB200 NVL GPUs."
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="8k,16k,24k,32k,48k,52k,64k,96k,128k",
        help="Comma-separated context lengths (use 'k' suffix for thousands).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Decode batch size per GPU (number of concurrent sequences).",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        choices=sorted(QUANT_PRESETS.keys()),
        default=16,
        help="Quantisation precision for weights + KV cache (default: 16).",
    )
    parser.add_argument(
        "--hbm-residency",
        type=float,
        default=DEFAULT_HBM_RESIDENCY,
        help=(
            "Fraction of cached tokens kept resident in HBM at steady state (0-1). "
            "Remaining tokens are phased to system memory/NVMe. Default 0.50."
        ),
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel degree (shards per layer).",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallel stages.",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        help="Expert parallel degree (MoE fan-out).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/gb200_kv_offload.png",
        help="Path to save the bandwidth plot.",
    )
    parser.add_argument(
        "--worst-case-delay",
        action="store_true",
        help="Include worst-case (stream every step) latency overhead alongside amortized estimates.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        default="qwen3-235b-a22b",
        help="Model preset for the primary simulation.",
    )
    parser.add_argument(
        "--random-swap",
        action="store_true",
        help="Simulate random user swaps between Qwen3-235B and DeepSeek-V3.2-Exp.",
    )
    parser.add_argument(
        "--swap-token",
        type=str,
        help="Context length for swap simulation (e.g. '64k'). Defaults to the largest token value passed to --tokens.",
    )
    parser.add_argument(
        "--swap-steps",
        type=int,
        default=2000,
        help="Number of random arrivals to simulate when --random-swap is enabled.",
    )
    parser.add_argument(
        "--swap-seed",
        type=int,
        default=42,
        help="Random seed for the swap simulation.",
    )
    parser.add_argument(
        "--kv-systems",
        type=str,
        default="plain",
        help=f"Comma-separated KV system presets to evaluate (choices: {', '.join(kv_system_choices())}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = parse_tokens(args.tokens)
    model = MODEL_PRESETS[args.model]
    quant = QUANT_PRESETS[args.quant_bits]
    hbm_residency = max(0.0, min(args.hbm_residency, 1.0))
    try:
        kv_system_keys = parse_kv_system_keys(args.kv_systems)
    except ValueError as exc:
        raise SystemExit(str(exc))

    base_output = Path(args.output)
    comparison_payloads: List[
        Tuple[str, List[OffloadResult], KVCacheSystem]
    ] = []

    for idx, system_key in enumerate(kv_system_keys):
        kv_system = get_kv_system(system_key)
        if idx > 0:
            print()
        print(f"=== {kv_system.label} ===")
        results = simulate_offload(
            tokens,
            args.batch_size,
            args.tp,
            args.pp,
            args.ep,
            model,
            quant,
            kv_system,
            hbm_residency,
        )
        print_results(
            results,
            args.batch_size,
            args.tp,
            args.pp,
            args.ep,
            args.worst_case_delay,
            model,
            args.quant_bits,
            kv_system,
            hbm_residency,
        )
        comparison_payloads.append((system_key, results, kv_system))
        if args.random_swap:
            if args.swap_token:
                swap_token_value = parse_single_token(args.swap_token)
            else:
                swap_token_value = max(tokens)
            qwen_model = MODEL_PRESETS["qwen3-235b-a22b"]
            deepseek_model = MODEL_PRESETS["deepseek-v3.2-exp-685b"]
            qwen_res = simulate_offload(
                [swap_token_value],
                args.batch_size,
                args.tp,
                args.pp,
                args.ep,
                qwen_model,
                quant,
                kv_system,
                hbm_residency,
            )[0]
            deepseek_res = simulate_offload(
                [swap_token_value],
                args.batch_size,
                args.tp,
                args.pp,
                args.ep,
                deepseek_model,
                quant,
                kv_system,
                hbm_residency,
            )[0]
            print(
                f"\n[{kv_system.label}] Random user swap simulation @ {format_tokens(swap_token_value)} tokens "
                f"over {args.swap_steps} arrivals (seed={args.swap_seed})"
            )
            for res in (qwen_res, deepseek_res):
                offload_gib = human_gbytes(res.offloaded_bytes_per_gpu)
                print(
                    f"  {res.model_name}: offloaded slice {offload_gib:.2f} GiB/GPU "
                    f"({res.tokens_offloaded:.0f} tokens offloaded)"
                )
                reload_table = compute_path_reload_times(res, kv_system)
                for link_name, delay_s in reload_table.items():
                    print(f"    {link_name}: reload {delay_s*1e3:.2f} ms")
            summary, swap_probability = simulate_random_user_swaps(
                qwen_res, deepseek_res, args.swap_steps, args.swap_seed, kv_system
            )
            print(f"Observed swap probability: {swap_probability*100:.2f}%")
            for link_name, stats in summary.items():
                print(
                    f"  {link_name}: avg +{stats['avg_ms']:.2f} ms | "
                    f"P95 +{stats['p95_ms']:.2f} ms"
                )
    plot_comparison(
        comparison_payloads,
        base_output,
        args.batch_size,
        args.worst_case_delay,
        model,
        args.quant_bits,
    )


if __name__ == "__main__":
    main()
