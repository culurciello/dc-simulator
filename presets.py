from typing import Dict, List

from GPU import GPUConfig
from model import ModelConfig, QuantConfig
from rack import LinkProfile, RackPreset


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
        intra_server=LinkProfile(throughput=1.8e12, latency=1.2e-6),
        inter_server=LinkProfile(throughput=0.9e12, latency=1.8e-6),
        inter_rack=LinkProfile(throughput=0.45e12, latency=4.5e-6),
        notes="NVL cabinet with 9 servers x 8 GB200 GPUs (host DRAM offload).",
        kv_system_key="gb200_host",
    ),
    RackPreset(
        name="NVIDIA GB300 NVL72",
        gpu_key="GB300",
        gpus_per_server=8,
        servers_per_rack=9,
        intra_server=LinkProfile(throughput=2.1e12, latency=1.0e-6),
        inter_server=LinkProfile(throughput=0.9e12, latency=1.8e-6),
        inter_rack=LinkProfile(throughput=0.45e12, latency=4.5e-6),
        notes="Future Blackwell-class NVL72 cabinet (estimates).",
    ),
    RackPreset(
        name="AMD MI300X OAM Rack",
        gpu_key="MI300X",
        gpus_per_server=8,
        servers_per_rack=8,
        intra_server=LinkProfile(throughput=0.95e12, latency=1.8e-6),
        inter_server=LinkProfile(throughput=0.45e12, latency=3.0e-6),
        inter_rack=LinkProfile(throughput=0.25e12, latency=6.0e-6),
        notes="4U OAM trays with Infinity Fabric interconnect.",
    ),
    RackPreset(
        name="AMD MI355X OAM Rack",
        gpu_key="MI355X",
        gpus_per_server=8,
        servers_per_rack=8,
        intra_server=LinkProfile(throughput=1.2e12, latency=1.5e-6),
        inter_server=LinkProfile(throughput=0.55e12, latency=2.5e-6),
        inter_rack=LinkProfile(throughput=0.3e12, latency=5.5e-6),
        notes="Projected MI355X fabric improvements.",
    ),
]


QUANT_PRESETS: Dict[int, QuantConfig] = {
    16: QuantConfig(bits=16, bytes_per_param=2.0, compute_scale=1.0, kv_bytes_per_elem=2.0),
    8: QuantConfig(bits=8, bytes_per_param=1.0, compute_scale=1.12, kv_bytes_per_elem=1.0),
    4: QuantConfig(bits=4, bytes_per_param=0.5, compute_scale=1.18, kv_bytes_per_elem=0.75),
}


MODEL_PRESETS: Dict[str, ModelConfig] = {
    "qwen3-235b-a22b": ModelConfig(
        name="Qwen3-235B-A22B",
        num_params=235e9,
        num_layers=94,
        hidden_size=1536,
        num_heads=64,
        context_length=4096,
        generation_window=1024,
        experts_per_layer=128,
        active_experts=8,
        expert_param_fraction=0.88,
        shared_param_fraction=0.12,
        moe_dispatch_factor=2.0,
    ),
    "deepseek-v3.2-exp-685b": ModelConfig(
        name="DeepSeek-V3.2-Exp",
        num_params=685e9,
        num_layers=128,
        hidden_size=2048,
        num_heads=128,
        context_length=8192,
        generation_window=2048,
        experts_per_layer=160,
        active_experts=12,
        expert_param_fraction=0.92,
        shared_param_fraction=0.08,
        moe_dispatch_factor=2.2,
    ),
    "chatgpt5-1p5t": ModelConfig(
        name="ChatGPT5-1P5T",
        num_params=1.5e12,
        num_layers=160,
        hidden_size=3072,
        num_heads=192,
        context_length=131072,
        generation_window=32768,
        experts_per_layer=256,
        active_experts=24,
        expert_param_fraction=0.94,
        shared_param_fraction=0.06,
        moe_dispatch_factor=2.4,
    ),
}


MODEL = MODEL_PRESETS["qwen3-235b-a22b"]
