# Hardware, models and assumptions


## Hardware specifications and capabilities - part I: Claude research

### **AMD MI300X**
- **Peak FP16/BF16:** 1,307.4 TFLOPS
- **Peak FP8:** 2,614.9 TFLOPS
- **Peak FP32:** 163.4 TFLOPS
- **Peak FP64:** 81.7 TFLOPS
- **Sustained Performance:** Achieves only 45-50% of peak in measured workloads (compared to H100/B200 which sustain >90%)

### **AMD MI355X**
- **Peak FP16:** 2.3 petaflops (2,300 TFLOPS)
- **Peak FP8:** 4.6 petaflops (4,600 TFLOPS)
- **Peak FP4:** 9.2 petaflops (9,200 TFLOPS)
- **Peak FP6:** 9.2 petaflops (9,200 TFLOPS)
- **Power Consumption:** 1,400W (for MI355X), 1,000W (for MI350X)

### **NVIDIA GB200 (per GPU)**
- **Peak FP16/BF16:** 2,500 TFLOPS (in GB200 NVL72 configuration)
- **Peak FP8:** ~5,000 TFLOPS (estimated from FP16)
- **Peak FP4:** 20 petaflops per GPU (sparse)
- **Per Superchip (2 GPUs):** 40 petaflops (INT8/FP4)
- **Full NVL72 Rack:** 1.44 exaflops FP4

### **NVIDIA H100**
- **Peak FP64:** 67 TFLOPS (SXM), 51 TFLOPS (PCIe)
- **Peak FP32:** 67 TFLOPS (SXM), 51 TFLOPS (PCIe)
- **Peak TF32:** 989 TFLOPS (SXM), 756 TFLOPS (PCIe)
- **Peak FP16/BF16:** 1,979 TFLOPS (SXM), 1,513 TFLOPS (PCIe)
- **Peak FP8:** 3,958 TFLOPS (SXM), 3,026 TFLOPS (PCIe)
- **Sustained Performance:** Generally achieves >90% of peak for optimized workloads

### **NVIDIA Rubin CPX**
- **Peak FP4:** 30 petaflops (30,000 TFLOPS)
- **Dense FP4:** 20 petaflops
- **Memory:** 128GB GDDR7
- **Power:** Estimated ~800W
- **Purpose:** Specialized for prefill phase/context processing

### **NVIDIA GB300 (Blackwell Ultra)**
- **Peak FP4:** 1.5x more AI compute FLOPS than GB200 (estimated ~15 petaflops per GPU)
- **Full NVL72 Rack:** 1.1 exaflops FP4
- **Memory per GPU:** 288GB HBM3e
- **Power:** 1,400W TDP
- **Performance Improvement:** 50% more FLOPS than B200

### Key Observations

1. **Peak vs Sustained:** AMD MI300X achieves only 45-50% sustained performance, while NVIDIA GPUs typically sustain >90% of peak
2. **Memory bandwidth:** MI300X sustains ~4.3 TB/s (81% utilization)
3. **Power trends:** Datacenter GPUs are increasing from 750W (MI300X) to 1,400W (MI355X, GB300)
4. **Specialization:** Rubin CPX is optimized for compute-heavy prefill, while standard GPUs handle memory-bound decode


## Hardware specifications and capabilities - part II: ChatGPT 5.1 deep research


### First: what “max, sustained FLOPs” really means

There are three different beasts here:

1. Peak FLOPs – marketing number from spec sheet (clock × units × ops).
2. Max-achievable FLOPs (MAF) – a sustained value on a dense GEMM microbenchmark under realistic thermals and non-cheating data. AMD publishes this for MI300X/MI325X.([rocm.blogs.amd.com][1])
3. Model FLOPs Utilization (MFU) – fraction of theoretical FLOPs actually used when training a big model (e.g. GPT-3, Llama-3, etc.) across a big cluster. SemiAnalysis and Meta publish these for H100 & friends.([SemiAnalysis][2])

You asked for “max, sustained FLOPs”; I’ll treat that as “best-known sustained FLOPs under heavy load,” which maps to (2) when available and otherwise to “good MFU × peak FLOPs” for large-model workloads.

---

### AMD MI300X

Peak (per AMD spec / server vendors)

* Peak FP16/BF16: ≈ 1.31 PFLOPS dense
* Peak FP8: ≈ 2.61 PFLOPS dense([GIGABYTE][3])

Max-Achievable FLOPs (MAF, AMD’s own measurement)([rocm.blogs.amd.com][1])

* FP16: 654 TFLOPS
* BF16: 708 TFLOPS
* FP8: 1,273 TFLOPS

Notes:

* These are sustained GEMM numbers with realistic random data, warmed-up thermals, etc., not a one-shot spike.
* For BF16, AMD notes that 708 TFLOPS corresponds to about 94% of the “calculated” BF16 peak at the measured clock, which is why MAF looks lower than the marketing peak but very close to “true” hardware capability.([rocm.blogs.amd.com][1])

If you want one-line “max sustained” numbers for MI300X, the honest answer is:

* Roughly 0.65–0.7 PFLOPS sustained BF16/FP16 on ideal GEMM
* Roughly 1.3 PFLOPS sustained FP8 on ideal GEMM

Real LLM training MFU will be lower (lots of non-GEMM work, comms, etc.), but AMD hasn’t published cluster MFU the way NVIDIA/Meta have for H100 yet.

---

### AMD MI355X (your “355X”)

MI355X is part of the MI350 family and is basically the bigger, newer sibling of MI300X.

Peak (from AMD / server vendors)([GIGABYTE][3])

* Peak FP8: 5.0 PFLOPS dense (10 PFLOPS sparse)
* Peak BF16/FP16: 2.5 PFLOPS dense (5 PFLOPS sparse)

Max sustained: no official MAF table from AMD yet as of late 2025.

However, Modular published a detailed bring-up blog where their Mojo matmul kernel on MI355X hits:

* ≈ 1,610,514 GFLOP/s on a tuned matmul kernel – about 1.61 PFLOPS sustained GEMM on MI355X.([modular.com][4])

That’s one specific kernel, one precision (effectively FP16/BF16) and not the whole stack, but it gives a feel: about two-thirds of the nominal 2.5 PFLOPS BF16 peak on a well-tuned matmul.

So a reasonable “max sustained GEMM” ballpark for MI355X right now is:

* ≈ 1.6–1.8 PFLOPS BF16/FP16 sustained on GEMM, depending on library and tuning.

Again, real LLM training MFU will be lower and workload-dependent.

---

### NVIDIA H100

Official peak (H100 SXM, per NVIDIA)([NVIDIA][5])

* FP16/BF16 Tensor: 1.98 PFLOPS
* FP8 Tensor: 3.96 PFLOPS

We don’t have an official “MAF table” from NVIDIA, but we do have good MFU data:

* SemiAnalysis shows BF16 MFU for GPT-3-style training on large H100 clusters improving from ~34% to ~54% over 2024 purely from software stack tuning.([SemiAnalysis][2])
* For Llama-3 405B, MFU in Meta’s large training run is around 41% BF16 on 16k H100s; SemiAnalysis confirms ~40–54% MFU across different scales.([SemiAnalysis][2])

Translate to rough “max sustained per GPU”:

* BF16: 1.98 PFLOPS × ~0.5 MFU ≈ 1.0 PFLOPS sustained in a well-tuned big LLM training run.
* FP8: 3.96 PFLOPS × ~0.4 MFU ≈ 1.5–1.6 PFLOPS sustained FP8 for similar workloads.

Those are not lab microbenchmarks; they’re “end-to-end training the huge model” numbers. In practice, the best operators are living in that ~40–55% of peak range on Hopper.

---

### NVIDIA GB200 (Grace Blackwell, B200 + Grace)

GB200 is the full Grace-Blackwell “superchip”; the AI part is the B200 GPU.

Peak compute:

* A single B200 GPU delivers around 20 PFLOPS dense FP4 (NVFP4) on fifth-gen Tensor Cores, per multiple architectural writeups.([Introl][6])
* Compared to H100, that’s roughly 5× the FP8 / FP16-class peak in terms of ops/second at low precision.

Performance data:

* NVIDIA’s own GB200 NVL72 tech blog shows 4× faster training vs H100 at equal GPU count on large LLMs and up to 30× more real-time MoE inference tokens/s than H100 in a specific 1.8T parameter model scenario.([NVIDIA Developer][7])
* SemiAnalysis’ GB200 vs H100 piece focuses on capex/opex and notes that large-scale frontier training on GB200 is still early and reliability-limited; there isn’t yet a clean public MFU table the way we have for H100.([SemiAnalysis][2])

So we can’t honestly say “GB200 max sustained FLOPs is X PFLOPS” yet. The best you can do is:

* Theoretical per B200: ~20 PFLOPS FP4 dense.
* If it ultimately reaches similar MFU as H100 (say 40–50%), you’re looking at something like 8–10 PFLOPS FP4 per GPU in big-model training as a plausible upper bound – but that’s a projection, not a measured public number.

Right now the public data is mostly relative (“4× faster than H100 on this LLM”) rather than absolute sustained FLOPs.

---

### NVIDIA GB300 (Blackwell Ultra / B300)

GB300 NVL72 uses Blackwell Ultra B300 GPUs.

Peak:

* NVIDIA’s own Blackwell Ultra writeups and Tom’s Hardware analysis:
  – Base Blackwell (B200): ~10 PFLOPS NVFP4 per GPU
  – Blackwell Ultra (B300) boosts NVFP4 TFLOPs by 1.5×, i.e. ~15 PFLOPS FP4 per GPU, and some sources/paraphrases round that to 20→30 PFLOPS when comparing older estimates.([NVIDIA Developer][8])

Cluster-level numbers:

* Microsoft’s first “supercomputer-scale” GB300 NVL72 Azure cluster: 4,608 GB300 GPUs delivering 92.1 exaFLOPS of FP4 inference performance (so ≈20 PFLOPS FP4 per GPU at peak, which is consistent with the higher end of those spec estimates).([Tom's Hardware][9])

Again, nobody is publishing a clean, per-GPU MFU/MAF table yet. If we take the ~20 PFLOPS FP4 peak per GPU from the Azure cluster math and assume a similar 40–50% MFU for big LLM training once the stack matures, you’d expect:

* Roughly 8–10 PFLOPS sustained FP4 per GB300 GPU on large-model training, in the same hand-wavy sense as the GB200 estimate.

For now, most public data is: “GB300 racks deliver roughly 2× the FP4 throughput of GB200 racks on MLPerf-style benchmarks” rather than “here’s a per-GPU sustained FLOPs number.”([HPCwire][10])

---

### NVIDIA Rubin CPX

Rubin CPX is a specialized “massive context inference” accelerator announced after Blackwell.

Peak:

* NVIDIA and SemiAnalysis both describe Rubin CPX as providing:
  – 30 PFLOPS sparse FP4, or 20 PFLOPS dense FP4 per CPX accelerator.([NVIDIA Newsroom][11])

System numbers:

* A full Vera Rubin NVL144 CPX rack: almost 8 exaFLOPS NVFP4 for inference, 100 TB high-speed memory, 1.7 PB/s bandwidth, and about 7.5× the AI performance of a GB300 NVL72 system for long-context workloads.([Futurum][12])

But again: no public per-GPU MFU table. Given its focus (inference, huge contexts), we’re likely to see workload-specific sustained numbers rather than a generic “max GEMM FLOPs” metric.

If you want a rough ceiling:

* Theoretical: 20 PFLOPS FP4 dense per CPX.
* Very hand-wavy “good MFU” sustained: something on the order of 8–10 PFLOPS FP4 for the right attention-heavy inference kernels – but that’s extrapolating from the same 40–50% MFU we see on H100/Blackwell for training, not a published benchmark.

---

### Putting rough numbers side by side

All numbers below are per GPU / accelerator, in petaFLOPS (PFLOPS).
“Ballpark sustained” is “peak × ~0.5” except where we have better data.

| GPU / accel         | Precision focus | Peak FLOPs (dense)      | Best public sustained-ish number | How solid is it?                                        |
| ------------------- | --------------- | ----------------------- | -------------------------------- | ------------------------------------------------------- |
| AMD MI300X          | BF16/FP16       | ≈1.31 PFLOPS            | 0.708 PFLOPS BF16 MAF            | Official AMD MAF([rocm.blogs.amd.com][1])               |
| AMD MI300X          | FP8             | ≈2.61 PFLOPS            | 1.273 PFLOPS FP8 MAF             | Official AMD MAF([rocm.blogs.amd.com][1])               |
| AMD MI355X          | BF16/FP16       | ≈2.5 PFLOPS             | ≈1.6 PFLOPS matmul (Modular)     | Third-party kernel microbench([modular.com][4])         |
| NVIDIA H100 (SXM)   | BF16            | 1.98 PFLOPS             | ≈1.0 PFLOPS (≈50% MFU)           | Inferred from GPT-3/Llama MFU([SemiAnalysis][2])        |
| NVIDIA H100 (SXM)   | FP8             | 3.96 PFLOPS             | ≈1.5–1.6 PFLOPS (≈40% MFU)       | Same as above                                           |
| NVIDIA GB200 / B200 | FP4             | ≈20 PFLOPS              | ≈8–10 PFLOPS (projected)         | Projection; no public MAF/MFU yet([Introl][6])          |
| NVIDIA GB300 / B300 | FP4             | ≈20 PFLOPS (Azure math) | ≈8–10 PFLOPS (projected)         | From cluster peak÷GPU; MFU unknown([Tom's Hardware][9]) |
| NVIDIA Rubin CPX    | FP4             | 20 PFLOPS               | ≈8–10 PFLOPS (very speculative)  | Only peak published so far([NVIDIA Newsroom][11])       |

---

### Why this is all annoyingly fuzzy

A couple of meta-points you probably already know, but they matter here:

* Vendors are starting to publish more realistic metrics (AMD’s MAF, NVIDIA’s MFU in Llama-3 training, etc.), but only AMD has done the “here’s a GEMM max-achievable FLOPs table” so far.
* NVIDIA leans more on MLPerf and big-cluster LLM training/inference demos, which are easier to market but don’t give a clean per-GPU sustained FLOPs number.
* MFU depends on model architecture, sequence length, parallelism strategy, comms topology, kernel fusion, framework, and 100 tiny details of the stack. So there is no single “true” sustained FLOPs per GPU; just upper-envelope curves for particular workloads.

If you’re trying to do sizing / TCO math (e.g. “how many GPUs to train X-B LLM to Y tokens”), the best mental model right now is:

* Take vendor peak FLOPs at the precision you care about.
* Assume ~40–55% MFU for well-tuned large-model training on a mature stack (H100 today, GB200/GB300 later).
* For MI300X/MI355X, you can aim for “MAF” as your GEMM ceiling and then apply a similar MFU haircut for full models.

That keeps you in the realm of “physics-respecting numbers” instead of hopeful marketing.

* [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/microsoft-deploys-worlds-first-supercomputer-scale-gb300-nvl72-azure-cluster-4-608-gb300-gpus-linked-together-to-form-a-single-unified-accelerator-capable-of-1-44-pflops-of-inference?utm_source=chatgpt.com)
* [The Verge](https://www.theverge.com/news/631835/nvidia-blackwell-ultra-ai-chip-gb300?utm_source=chatgpt.com)

[1]: https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html "Measuring Max-Achievable FLOPs – Part 2 — ROCm Blogs"
[2]: https://newsletter.semianalysis.com/p/h100-vs-gb200-nvl72-training-benchmarks "H100 vs GB200 NVL72 Training Benchmarks - Power, TCO, and Reliability Analysis, Software Improvement Over Time"
[3]: https://www.gigabyte.com/Solutions/amd-instinct "AMD Instinct | Solution - GIGABYTE Global"
[4]: https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days "Modular: Achieving State-of-the-Art Performance on AMD MI355 — in Just 14 Days"
[5]: https://www.nvidia.com/en-us/data-center/h100/?utm_source=chatgpt.com "H100 GPU"
[6]: https://introl.com/blog/fp4-inference-efficiency-nvidia-2025?utm_source=chatgpt.com "NVIDIA's FP4 Inference Delivers 50x Efficiency - Introl"
[7]: https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/ "NVIDIA GB200 NVL72 Delivers Trillion-Parameter LLM Training and Real-Time Inference | NVIDIA Technical Blog"
[8]: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/?utm_source=chatgpt.com "Inside NVIDIA Blackwell Ultra: The Chip Powering the AI ..."
[9]: https://www.tomshardware.com/tech-industry/artificial-intelligence/microsoft-deploys-worlds-first-supercomputer-scale-gb300-nvl72-azure-cluster-4-608-gb300-gpus-linked-together-to-form-a-single-unified-accelerator-capable-of-1-44-pflops-of-inference?utm_source=chatgpt.com "Microsoft deploys world's first 'supercomputer-scale' GB300 NVL72 Azure cluster - 4,608 GB300 GPUs linked together to form a single, unified accelerator capable of 92.1 exaFLOPS of FP4 inference"
[10]: https://www.hpcwire.com/aiwire/2025/11/14/nvidia-showcases-blackwell-ultra-performance-on-mlperf-benchmark/?utm_source=chatgpt.com "Nvidia Showcases Blackwell Ultra Performance on MLPerf ..."
[11]: https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference?utm_source=chatgpt.com "NVIDIA Unveils Rubin CPX: A New Class of GPU ..."
[12]: https://futurumgroup.com/insights/nvidias-new-rubin-cpx-targets-future-of-large-scale-inference/?utm_source=chatgpt.com "NVIDIA Rubin CPX Targets Future of Large-Scale Inference"





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

All bandwidth figures are “effective” sustained numbers; swap in values that match your kernels, NCCL configs, and traffic patterns.


## Parameter provenance

- `sustained_flops`: decode-mode throughput after utilisation losses. Numbers are drawn from NVIDIA/AMD architecture whitepapers (e.g. H100/GB200 NVL, MI300X) and public tuning notes from TensorRT-LLM and DeepSpeed inference benchmarks. They assume 20–35 % efficiency loss from peak FP16/BF16 throughput; replace with your own `ncu`/`nsys` measurements for accuracy.
- `hbm_bw`: effective memory bandwidth per GPU, based on vendor specs (HBM3/HBM3e) multiplied by a 0.6–0.7 utilisation factor seen in decode-heavy workloads. Tune using `nsys` dram_util metrics or vLLM profile dumps.
- `compute_scale` (quant presets): empirical utilisation uplift when kernels run in lower precision; defaults reflect typical TensorRT-LLM results for FP8/INT8 and QLoRA-style INT4 matmuls. Adjust using benchmark deltas between precisions on your stack.
- `kv_bytes_per_elem`: assumes FP16 KV for 16-bit, FP8/INT8 for 8-bit, and FP8-with-compression for 4-bit variants. If you use paged attention or flash-decoding tweaks, swap in the measured per-element size instead.
- Interconnect links (`RackPreset.intra_server`, `RackPreset.inter_server`, `RackPreset.inter_rack`): each link captures sustained bandwidth (bytes/s) and per-message latency (seconds). Seed these with NCCL collective benchmarks and latency telemetry for your fabric (NVLink/NVSwitch, XGMI/Infinity Fabric, InfiniBand/roce). The simulator accounts for both throughput and handshake latency when sizing communication ceilings.
  
See also `presets.py` for model details.

## Qwen3 mixture-of-experts assumptions

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



## KV cache systems

- The default preset is the three-component server:
  - **Grace CPU complex** with 2 TiB DRAM feeding the Switch over aggregated PCIe 5 (≈512 GB/s)
  - **Hopper/GB300 GPU** connected via NVLink C2C (≈900 GB/s)
  - **Dedicated 512 TiB KV DRAM sled** wired to the switch through a ~2 TB/s link and serviced by the switch’s KV DMA engine
  All eviction/reload traffic is modeled over the GPU↔KV DMA lane exposed by the switch, while CPU↔GPU/DRAM paths cover orchestration spill. Every CLI run prints the fabric description plus how much of the KV tier the workload consumes so it’s easy to size latency and bandwidth headroom without diving into the docs.

- Pass `--kv-systems plain` to model a stock GB300 Vera Rubin server that offloads directly to host DRAM (≈2 TiB over 512 GB/s PCIe/NVLink). The offload tier in this mode is limited to the CPU’s memory bandwidth and capacity.

- Provide a comma-separated list like `--kv-systems plain` to emit side-by-side tables in the CLI while the simulator writes a **single** comparison plot (to the `--output` path) that overlays host-traffic and reload-latency curves for every selected fabric.

- Set `--quant-bits 4|8|16` to match the precision you are serving with (default 4). The simulator applies the corresponding weight footprint, KV element size, and compute scaling when sizing memory and bandwidth.

- Every run exports the plotted data as a tab-separated `.xls` (same basename as `--output`) so you can slice the bandwidth/latency numbers directly in Excel/Sheets.