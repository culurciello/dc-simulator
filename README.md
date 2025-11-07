# Data-center AI cluster, rack simulator


![Inference results](plots/amd_mi355x_oam_rack_inference.png)

A lightweight rack-scale throughput estimator for running an LLM at scale (Qwen3-235B) across
multiple data-center compute racks. The Python simulator sweeps tensor/pipeline parallel
plans, quantization (4/8/16 bit), and batch sizes for each hardware preset, then reports the
sustained throughput ceiling and the bottleneck (compute, HBM, or fabric) for both inference
and training workloads.

The current presets cover 8 racks of:
- NVIDIA GB200 NVL72 cabinets
- NVIDIA GB300 NVL72 cabinets (projected)
- AMD MI300X OAM racks
- AMD MI355X OAM racks (projected)

Each run factors in realistic KV-cache traffic, per-link bandwidth ceilings,
and rack-level topology so you can see the optimal number of concurrent model
instances and their tokens/sec.


## Requirements

- Python 3.9+
- `numpy`
- `matplotlib` (only required for the optional plotting step; execution is CLI)

Install dependencies once:

```bash
pip install numpy matplotlib
```

## Run


#### Inference:

```bash
python3 llm_cluster_simulator.py --mode inference --batch-sizes "1,2,4,6,8,10,12,14,16" --quant-bits "4"
```

The script prints a table per rack preset. Columns include the batch size,
quantization, total rack throughput, per-instance throughput, optimal
tensor/pipeline split, memory footprint per GPU, and the load each instance
puts on intra-server, inter-server, and inter-rack fabrics.

#### Training:

Switch to training-mode sizing (forward/backward + optimizer traffic) with:

```bash
python3 llm_cluster_simulator.py --mode training --batch-sizes "8,16,32,64,96,128,192,256" --quant-bits "4"
```

Training summaries add per-instance and aggregate samples/sec (derived from
steps/sec) plus data-parallel gradient load so you can gauge utilisation headroom
on each fabric. Plots in this mode chart total samples processed per second.

![training results](plots/amd_mi355x_oam_rack_training.png)


#### KV-cache offloading:


```
python3 kv_offload_simulator.py
```

See `docs` for more info.


## Customising the model

Key constants live near the top of `llm_cluster_simulator.py`.

- `MODEL`: update layer count, hidden size, or cached tokens to reflect your
  architecture. For Qwen3 MoE tuning, edit `experts_per_layer`, `active_experts`,
  `expert_param_fraction`, and `shared_param_fraction` to match the release notes
  you have on hand.
- `QUANT_PRESETS`: adjust weight size, sustained compute scaling, or KV element
  size for different quantisation schemes.
- `RACK_PRESETS`: add or tweak a rack by editing GPU counts and interconnect
  link profiles (bandwidth in bytes/sec plus per-message latency in seconds).
- `GPU_PRESETS`: change sustained FLOPs or HBM bandwidth to match benchmark
  data.
- `tp_candidates`, `pp_candidates`, `ep_candidates`: space swept during auto-tuning.
  Batch sizes and quant options can be supplied at runtime via
  `--batch-sizes` / `--quant-bits`.

All bandwidth figures are “effective” sustained numbers; swap in values that
match your kernels, NCCL configs, and traffic patterns.

### Parameter provenance

- `sustained_flops`: decode-mode throughput after utilisation losses. Numbers
  are drawn from NVIDIA/AMD architecture whitepapers (e.g. H100/GB200 NVL, MI300X)
  and public tuning notes from TensorRT-LLM and DeepSpeed inference benchmarks.
  They assume 20–35 % efficiency loss from peak FP16/BF16 throughput; replace
  with your own `ncu`/`nsys` measurements for accuracy.
- `hbm_bw`: effective memory bandwidth per GPU, based on vendor specs
  (HBM3/HBM3e) multiplied by a 0.6–0.7 utilisation factor seen in decode-heavy
  workloads. Tune using `nsys` dram_util metrics or vLLM profile dumps.
- `compute_scale` (quant presets): empirical utilisation uplift when kernels
  run in lower precision; defaults reflect typical TensorRT-LLM results for
  FP8/INT8 and QLoRA-style INT4 matmuls. Adjust using benchmark deltas between
  precisions on your stack.
- `kv_bytes_per_elem`: assumes FP16 KV for 16-bit, FP8/INT8 for 8-bit, and
  FP8-with-compression for 4-bit variants. If you use paged attention or
  flash-decoding tweaks, swap in the measured per-element size instead.
- Interconnect links (`RackPreset.intra_server`, `RackPreset.inter_server`,
  `RackPreset.inter_rack`): each link captures sustained bandwidth (bytes/s) and
  per-message latency (seconds). Seed these with NCCL collective benchmarks and
  latency telemetry for your fabric (NVLink/NVSwitch, XGMI/Infinity Fabric,
  InfiniBand/roce). The simulator accounts for both throughput and handshake
  latency when sizing communication ceilings.

### Qwen3 mixture-of-experts assumptions

[Qwen3 architecture](https://magazine.sebastianraschka.com/p/qwen3-from-scratch)


- Default settings assume a top-8-of-64 MoE layout (mirroring Qwen3’s release
  collateral). Adjust `experts_per_layer` and `active_experts` if you have an
  updated architecture card.
- `expert_param_fraction` and `shared_param_fraction` split the 235 B parameters
  into expert FFNs (≈88 %) and shared weights/router/attention (≈12 %). Tweak
  these ratios if your checkpoint skews differently.
- Expert traffic is modelled as an all-to-all exchange of FP16 activations with
  a round-trip multiplier (`moe_dispatch_factor`, default 2×) to capture send +
  gather cost. Increase it if you measure higher overhead due to packing or
  capacity drops.
- Expert-parallel degrees explored in `ep_candidates` multiply the tensor ×
  pipeline mesh. The simulator assumes shared weights are replicated across EP
  groups while expert weights shard, matching how DeepSpeed/TensorRT-LLM deploy
  MoE inference today.


## Interpreting the results

- **Tok/GPU(avg)**: cluster-average tokens/s per active GPU (total rack TPS /
  total active devices); use it to compare device-level efficiency across
  strategies.
- **Inst TPS** / **Total TPS**: sustained tokens/sec for a single model
  instance and across all racks. In training mode the table also lists the
  equivalent **Inst Steps/s** and **Tot Steps/s** to highlight optimiser pace.
- **Limit**: which subsystem capped throughput (`compute`, `hbm`, `comm`).
- **Bounds tok/GPU(avg)**: theoretical per-device ceilings (without utilisation
  scaling) derived from compute, HBM, and communication limits. If the reported
  Tok/GPU(avg) is far below the relevant bound, there may be software headroom.
- **Fabric load**: how much bandwidth (GB/s) each instance consumes on the
  intra-server, inter-server, and inter-rack links. Compare these against your
  real fabrics to gauge headroom.

Because the simulator assumes steady-state decode, prefill/first-token latency
is not modelled. Prefill-heavy workloads typically shift bottlenecks toward
HBM or interconnect; extend the simulator if you need that view.



#### Note 1:
The [published data](https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256?utm_source=publication-search) 2,500 TFLOPs (GB200) and 2,300 TFLOPs (MI300-class) are the peak marketing numbers: they assume
  perfectly-tiled matrix multiplies in FP8/BF16, full SM occupancy, no communication, and all of the power budget devoted to
  math. Two things tank that figure for real LLM decoding:

  - Decode is memory- and latency-bound. Each new token touches the whole KV cache, does small GEMVs, and waits on
    collectives. You rarely exceed 20–30 % FP unit utilisation, so the sustained math rate falls to a few‑dozen TFLOPs even
    on top-tier parts.
  - End-to-end utilisation losses. Scheduling, quant kernels, tensor/pipeline parallel collectives, and micro-batching shave
    off another 10–30 %. The numbers in the simulator (24 TFLOPs for GB200, 11.5 TFLOPs for MI300X, 14.5 TFLOPs for MI355X)
    mirror what vendors and open benchmarks report for decode throughput, not HPC matmul runs.

  If you have better telemetry (e.g. nsys/ncu traces or TensorRT‑LLM benchmarks on your stack), just drop the measured
  sustained FLOP rates into GPU_PRESETS and the simulator will rescale automatically; the defaults are there so the model
  produces realistic tokens/sec out of the box rather than theoretical peaks.

#### Note 2:

The “compute max bound” column is only a first‑order ceiling that we derive from two inputs:

  `compute_bound_per_gpu ≈ (sustained_flops · compute_scale) / flops_per_token`

  where sustained_flops is the decode‑mode FLOPs/s you enter in GPU_PRESETS, compute_scale is the utilisation bump you assign to a quant mode, and flops_per_token
  is computed as 2 × active_params. For Qwen3 we approximate the active parameters as the shared weights plus the slice of experts that fire (expert_param_fraction ×
  active_experts / experts_per_layer). That whole stack is an approximation. Whenever you see the reported tok/GPU(avg) edge above that bound, it simply means one (or
  more) of the inputs is conservative relative to what your kernels are actually doing. Common reasons:

  - Sustained FLOPs are higher than the default – e.g. you have better scheduling, fused kernels, or FP8 pipelines than the 11–24 TFLOPs figures we baked in from
    vendor whitepapers. Plug your measured decode throughput (from ncu/nsys, TensorRT‑LLM, etc.) into GPU_PRESETS.
  - MoE sparsity is stronger than our guess – if fewer experts fire, or the shared/expert fraction in your checkpoint differs from the 12 % / 88 % split we assumed,
    the true FLOPs/token drops, so more tokens/s can fit under the same FLOP budget. Adjust expert_param_fraction, shared_param_fraction, and active_experts.
  - Quantisation uplift – our compute_scale factors (1.12 for 8‑bit, 1.18 for 4‑bit) are averages. If your INT4 kernels deliver a larger boost, the GPU can sustain
    more FLOPs than we account for.

  In short, the ceiling is a modelling aid, not a hard limit enforced by physics. If your observed tok/GPU goes past it, update the sustained FLOP numbers or the MoE
  fractions to match your telemetry; the simulator will rescale accordingly.


### Note 3:

KV-cache offloading is one of the biggest architectural challenges in large-scale LLM inference (especially with multi-hundred-billion-parameter models like Qwen3-235B).

You need to offload (move) part of the KV-cache to CPU memory or another storage tier when GPU memory capacity is insufficient to hold:

- The entire model’s weights (which are read-only), plus
- The active KV-cache for all tokens and all concurrent inference sessions.

Typical scenarios include:

- Long-context inference (e.g., 32 K, 128 K tokens) 
KV-cache grows linearly with sequence length → can exceed 100 GB per context for very large models.


## Extending

- Add prefill modelling by sizing activation footprints and read/write bursts.
- Introduce heterogeneous racks by instantiating multiple `simulate_rack`
  calls with different rack counts.
- Emit CSV/JSON by adapting `print_summary` or plugging the returned dict into
  your own reporting layer.

## Computation flow overview

1. **Preset selection** – For each rack profile we pull the associated GPU
   capabilities, server topology, and interconnect bandwidths. Quantisation
   presets determine weight size, sustained compute scaling, and KV precision.
2. **Search space sweep** – For every batch size and quant setting we iterate
   tensor-parallel (`tp_candidates`), pipeline-parallel (`pp_candidates`), and
   expert-parallel (`ep_candidates`) degrees that evenly divide heads, layers,
   and expert shards respectively.
3. **Feasibility checks** – Each `(tp, pp)` pair must fit within rack GPU
   counts and HBM capacity. Memory usage covers sharded weights, KV cache for
   cached tokens, plus a small buffer to guard against fragmentation.
4. **Throughput estimation** – The simulator computes three ceilings per
   instance: compute (sustained FLOPs, factoring active experts), HBM (KV read
   bandwidth), and communication (tensor/pipeline collectives plus MoE
   dispatch). Batch size influences kernel efficiency via an empirical scaling
   curve.
5. **Instance aggregation** – The minimum of the three ceilings becomes the
   instance tokens/sec. We multiply by the number of instances that fit in
   8 racks to get rack-wide throughput and record which subsystem bottlenecks.
6. **Reporting & plots** – For each rack we print the optimal configuration per
   batch/quant and emit a PNG showing aggregate throughput vs batch so you can
   compare precisions at a glance.
