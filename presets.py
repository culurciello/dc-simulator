from typing import Dict, List

from GPU import GPUConfig
from model import ModelConfig, QuantConfig
from rack import RackPreset

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
        notes="NVL cabinet with 9 servers x 8 GB200 GPUs.",
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

