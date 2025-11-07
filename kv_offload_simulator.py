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
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from presets import GPU_PRESETS, MODEL, QUANT_PRESETS
from utils import batch_efficiency, fits_memory, human_gbytes, kv_bytes_per_token

INFERENCE_KV_AMPLIFICATION = 1.1


@dataclass
class OffloadResult:
    tokens: int
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

    @property
    def total_worst_bw(self) -> float:
        return self.evict_bw + self.worst_read_bw

    @property
    def total_amortized_bw(self) -> float:
        return self.evict_bw + self.amortized_read_bw


@dataclass(frozen=True)
class HostLink:
    name: str
    bandwidth_bytes: float


HOST_LINKS: List[HostLink] = [
    HostLink("Hopper GPU-CPU NVLink", 900e9),  # NVLink C2C from Hopper SuperChip
]


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


def simulate_offload(
    tokens_list: Iterable[int],
    batch_size: int,
    tp: int,
    pp: int,
    ep: int,
) -> List[OffloadResult]:
    quant = QUANT_PRESETS[4]
    gpu = GPU_PRESETS["GB200"]
    gpus_per_instance = tp * pp * max(ep, 1)

    results: List[OffloadResult] = []

    for tokens in tokens_list:
        model_variant = replace(MODEL, context_length=tokens, generation_window=0)
        mem_ok, _, breakdown = fits_memory(
            gpu, model_variant, quant, tp=tp, pp=pp, ep=ep, batch_size=batch_size
        )

        kv_total = kv_bytes_per_token(model_variant, quant)
        kv_per_gpu = kv_total / (tp * pp)
        tokens_cached = model_variant.total_cached_tokens * batch_size

        limit = breakdown["limit"]
        weights = breakdown["weights"]
        overhead = breakdown["overhead"]
        available_kv_bytes = max(0.0, limit - weights - overhead)

        if kv_per_gpu > 0:
            tokens_on_gpu = min(tokens_cached, available_kv_bytes / kv_per_gpu)
        else:
            tokens_on_gpu = tokens_cached

        tokens_offloaded = max(0.0, tokens_cached - tokens_on_gpu)
        offload_fraction = (
            tokens_offloaded / tokens_cached if tokens_cached > 0 else 0.0
        )

        eff = batch_efficiency(batch_size)
        gpu_total_flops = gpu.sustained_flops * quant.compute_scale
        flops_per_token = model_variant.flops_per_token

        compute_bound = (gpus_per_instance * gpu_total_flops) / flops_per_token
        compute_tps = compute_bound * eff

        hbm_per_token = kv_per_gpu * INFERENCE_KV_AMPLIFICATION
        hbm_bound = gpu.hbm_bw / hbm_per_token if hbm_per_token > 0 else float("inf")
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
) -> None:
    gpu = GPU_PRESETS["GB200"]
    params_str = f"{MODEL.num_params/1e9:.1f}B"
    print("KV offloading requirements (per GPU unless noted):")
    print(
        "Batch size: {} | TPxPPxEP: {}x{}x{} | Quant: 4-bit | GPU: {}".format(
            batch_size, tp, pp, ep, gpu.name
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
        "Model parameters: {} | Decode batch size per GPU: {}".format(
            params_str, batch_size
        )
    )
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
        for link in HOST_LINKS:
            if offloaded_bytes > 0.0 and link.bandwidth_bytes > 0.0:
                amortized_extra_s = offloaded_bytes / link.bandwidth_bytes
                worst_extra_s = (
                    (offloaded_bytes * res.tokens_cached) / link.bandwidth_bytes
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
                f"{link.name}: +{amortized_ms:.2f} ms ({amortized_pct:.2f}%) amortized"
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
            delay_label = "Host link induced latency (amortized + worst-case)"
        else:
            delay_label = "Host link induced latency (amortized)"
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
    print()


def plot_results(
    results: List[OffloadResult],
    output_path: Path,
    batch_size: int,
    show_worst_case_delay: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_k = [res.tokens / 1_000.0 for res in results]
    total_worst = [res.total_worst_bw / 1e9 for res in results]
    total_amort = [res.total_amortized_bw / 1e9 for res in results]
    evict_only = [res.evict_bw / 1e9 for res in results]
    gpu = GPU_PRESETS["GB200"]
    params_str = f"{MODEL.num_params/1e9:.1f}B"
    amortized_delay_series = {link.name: [] for link in HOST_LINKS}
    worst_delay_series = {link.name: [] for link in HOST_LINKS} if show_worst_case_delay else None
    swap_reload_series = {link.name: [] for link in HOST_LINKS}

    for res in results:
        offloaded_bytes = res.tokens_offloaded * res.kv_bytes_per_token_per_gpu
        for link in HOST_LINKS:
            if offloaded_bytes > 0.0:
                amortized_extra_s = offloaded_bytes / link.bandwidth_bytes
                amortized_delay_series[link.name].append(amortized_extra_s * 1e3)
                swap_reload_series[link.name].append(amortized_extra_s)
                if show_worst_case_delay and worst_delay_series is not None:
                    worst_extra_s = (
                        (offloaded_bytes * res.tokens_cached) / link.bandwidth_bytes
                        if res.tokens_cached > 0.0
                        else 0.0
                    )
                    worst_delay_series[link.name].append(worst_extra_s * 1e3)
            else:
                amortized_delay_series[link.name].append(0.0)
                swap_reload_series[link.name].append(0.0)
                if show_worst_case_delay and worst_delay_series is not None:
                    worst_delay_series[link.name].append(0.0)

    fig, (ax_bw, ax_delay) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
    ax_bw.plot(
        tokens_k,
        total_amort,
        marker="s",
        label="Host traffic (amortised)",
    )
    ax_bw.plot(
        tokens_k,
        evict_only,
        linestyle="--",
        color="tab:gray",
        label="Eviction writes only",
    )

    ax_bw.set_ylabel("Bandwidth per GPU (GB/s)")
    ax_bw.set_title(
        "{} KV-cache offload @ 4-bit | {} params | batch {}/GPU".format(
            gpu.name, params_str, batch_size
        )
    )
    ax_bw.grid(True, linestyle="--", alpha=0.35)
    ax_bw.legend()

    for link_name, series in amortized_delay_series.items():
        ax_delay.plot(
            tokens_k,
            series,
            marker="o",
            label=f"{link_name} amortized",
        )
    if show_worst_case_delay and worst_delay_series is not None:
        for link_name, series in worst_delay_series.items():
            ax_delay.plot(
                tokens_k,
                series,
                linestyle="--",
                marker=None,
                label=f"{link_name} worst-case",
            )
    ax_delay.set_xlabel("Cached tokens per batch (thousands)")
    ax_delay.set_ylabel("Additional inference time (ms)")
    ax_delay.grid(True, linestyle="--", alpha=0.35)
    ax_delay.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {output_path}")

    swap_path = output_path.with_name(f"{output_path.stem}_swap{output_path.suffix}")
    fig_swap, ax_swap = plt.subplots(figsize=(7.0, 4.0))
    for link_name, series in swap_reload_series.items():
        ax_swap.plot(
            tokens_k,
            [value * 1e3 for value in series],
            marker="o",
            label=f"{link_name} reload",
        )
    ax_swap.set_xlabel("Cached tokens per batch (thousands)")
    ax_swap.set_ylabel("User swap reload time (ms)")
    ax_swap.set_title(
        "{} host reload penalty | {} params | batch {}/GPU".format(
            gpu.name, params_str, batch_size
        )
    )
    ax_swap.grid(True, linestyle="--", alpha=0.35)
    ax_swap.legend()
    fig_swap.tight_layout()
    fig_swap.savefig(swap_path, dpi=160)
    plt.close(fig_swap)
    print(f"Saved swap-delay plot: {swap_path}")


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
        default=1,
        help="Decode batch size per GPU (number of concurrent sequences).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = parse_tokens(args.tokens)
    results = simulate_offload(tokens, args.batch_size, args.tp, args.pp, args.ep)
    print_results(
        results,
        args.batch_size,
        args.tp,
        args.pp,
        args.ep,
        args.worst_case_delay,
    )
    plot_results(
        results,
        Path(args.output),
        args.batch_size,
        args.worst_case_delay,
    )


if __name__ == "__main__":
    main()
