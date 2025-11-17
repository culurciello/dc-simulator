
## Interpreting the results

- Tok/GPU(avg): cluster-average tokens/s per active GPU (total rack TPS /
  total active devices); use it to compare device-level efficiency across
  strategies.
- Inst TPS / Total TPS: sustained tokens/sec for a single model
  instance and across all racks. In training mode the table also lists the
  equivalent Inst Steps/s and Tot Steps/s to highlight optimiser pace.
- Limit**: which subsystem capped throughput (`compute`, `hbm`, `comm`).
- Bounds tok/GPU(avg): theoretical per-device ceilings (without utilisation
  scaling) derived from compute, HBM, and communication limits. If the reported
  Tok/GPU(avg) is far below the relevant bound, there may be software headroom.
- Fabric load: how much bandwidth (GB/s) each instance consumes on the
  intra-server, inter-server, and inter-rack links. Compare these against your
  real fabrics to gauge headroom.

Because the simulator assumes steady-state decode, prefill/first-token latency
is not modelled. Prefill-heavy workloads typically shift bottlenecks toward
HBM or interconnect; extend the simulator if you need that view.



#### Note 1:
The [published data](https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256?utm_source=publication-search) 2,500 TFLOPs (GB200) and 2,300 TFLOPs (MI300-class) are the peak marketing numbers: they assume perfectly-tiled matrix multiplies in FP8/BF16, full SM occupancy, no communication, and all of the power budget devoted to math. Two things tank that figure for real LLM decoding:

  - Decode is memory- and latency-bound. Each new token touches the whole KV cache, does small GEMVs, and waits on collectives. You rarely exceed 20–30 % FP unit utilisation, so the sustained math rate falls to a few‑dozen TFLOPs even on top-tier parts.
  - End-to-end utilisation losses. Scheduling, quant kernels, tensor/pipeline parallel collectives, and micro-batching shave off another 10–30 %. The numbers in the simulator (24 TFLOPs for GB200, 11.5 TFLOPs for MI300X, 14.5 TFLOPs for MI355X) mirror what vendors and open benchmarks report for decode throughput, not HPC matmul runs.


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


#### Note 4:

Question: why is the network utilization so low? how would it change if we used ethernet 400 gbps everywhere?


Answer: Network util stays tiny because the current optimal plans stick to low tensor/pipeline degrees (1×1×1 or 1×2×2) and relatively small batches, so each instance lives inside a single NVSwitch domain. Most traffic is intra-server NVLink—
pipeline boundaries and MoE dispatch barely touch the inter-rack links. For the example you ran (tokens-cached 0), GPUs run ~27% compute but the comm ceiling is effectively infinite because nothing ever pushes off-box.

If you swapped every fabric link to 400 GbE (≈50 GB/s usable) you’d lower both inter-server and inter-rack throughput by ~18× compared to NVLink/NVSwitch (0.9 TB/s nominal). Utilization numbers would jump proportionally: a plan that currently reports ~1% network util would look more like 18–20%, and multi-stage plans could become comm-bound because the per-token
latency would also rise (Ethernet microseconds vs NVSwitch sub-µs). You’d likely need to cut tensor/pipeline degrees or add racks to keep the same throughput. So while the raw utilization seems low today, it’s a direct result of fast fabrics;
replacing them with 400 GbE would make comm the bottleneck for these workloads.