# models and assumptions


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