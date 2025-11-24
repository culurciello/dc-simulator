from typing import Dict, List

from GPU import GPUConfig
from kv_subsystem import KVCacheSystem, KVComponent, KVSwitch
from model import ModelConfig, QuantConfig
from rack import LinkProfile, RackPreset


GPU_PRESETS: Dict[str, GPUConfig] = {
    "MI300X": GPUConfig(
        "AMD MI300X",
        sustained_flops=11.5e12,
        hbm_bw=1.9e12,
        max_mem_bytes=192e9,
    ),
    "H100": GPUConfig(
        "NVIDIA H100",
        sustained_flops=18e12,
        hbm_bw=3.3e12,
        max_mem_bytes=80e9,
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
        name="NVIDIA H100 NVL72",
        gpu_key="H100",
        gpus_per_server=8,
        servers_per_rack=9,
        intra_server=LinkProfile(throughput=0.9e12, latency=1.6e-6),
        inter_server=LinkProfile(throughput=0.6e12, latency=2.2e-6),
        inter_rack=LinkProfile(throughput=0.3e12, latency=4.8e-6),
        notes="H100 NVL72-style rack (host DRAM offload).",
        kv_system_key="h100_plain",
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
    RackPreset(
        name="NVIDIA Vera Rubin",
        gpu_key="GB300",
        gpus_per_server=12,
        servers_per_rack=8,
        intra_server=LinkProfile(throughput=2.5e12, latency=0.9e-6),
        inter_server=LinkProfile(throughput=1.4e12, latency=1.5e-6),
        inter_rack=LinkProfile(throughput=0.75e12, latency=3.5e-6),
        notes="Speculative GB300-class Rubin inference pod (NVLink 6 fabric + denser sleds).",
        kv_system_key="gb300_host",
        storage_servers_per_rack=3,
        storage_server_capacity_bytes=160e12,
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


def _build_gb200_plain_kv_system() -> KVCacheSystem:
    cpu = KVComponent(
        name="Grace CPU",
        role="cpu",
        memory_bytes=0.0,
        link_bandwidth=512e9,
        link_latency=500e-9,
    )
    gpu = KVComponent(
        name="Hopper/GB200 GPU",
        role="gpu",
        memory_bytes=192e9,
        link_bandwidth=900e9,
        link_latency=350e-9,
    )
    kv_dram = KVComponent(
        name="Host DRAM (CPU-attached)",
        role="kv_dram",
        memory_bytes=512e9,
        link_bandwidth=512e9,
        link_latency=500e-9,
    )
    switch = KVSwitch(
        name="Grace Hopper NVLink/PCIe fabric",
        fabric_bandwidth=512e9,
        base_latency=250e-9,
        kv_dma_bandwidth=546e9,
    )
    return KVCacheSystem(
        cpu=cpu,
        gpu=gpu,
        kv_dram=kv_dram,
        switch=switch,
        label="Plain GB200 host-offload",
    )


def _build_gb300_plain_kv_system() -> KVCacheSystem:
    cpu = KVComponent(
        name="Vera CPU",
        role="cpu",
        memory_bytes=0.0,
        link_bandwidth=800e9,
        link_latency=400e-9,
    )
    gpu = KVComponent(
        name="Rubin/GB300 GPU",
        role="gpu",
        memory_bytes=228e9,
        link_bandwidth=1.4e12,
        link_latency=250e-9,
    )
    kv_dram = KVComponent(
        name="Vera Host DRAM",
        role="kv_dram",
        memory_bytes=1.0e12,
        link_bandwidth=1.0e12,
        link_latency=450e-9,
    )
    switch = KVSwitch(
        name="NVLink 6 fabric",
        fabric_bandwidth=1.2e12,
        base_latency=200e-9,
        kv_dma_bandwidth=1.2e12,
    )
    return KVCacheSystem(
        cpu=cpu,
        gpu=gpu,
        kv_dram=kv_dram,
        switch=switch,
        label="Vera-Rubin GB300 host-offload",
    )


def _build_h100_plain_kv_system() -> KVCacheSystem:
    cpu = KVComponent(
        name="Grace CPU",
        role="cpu",
        memory_bytes=0.0,
        link_bandwidth=512e9,
        link_latency=500e-9,
    )
    gpu = KVComponent(
        name="Hopper H100 GPU",
        role="gpu",
        memory_bytes=80e9,
        link_bandwidth=900e9,
        link_latency=350e-9,
    )
    kv_dram = KVComponent(
        name="Host DRAM (CPU-attached)",
        role="kv_dram",
        memory_bytes=512e9,
        link_bandwidth=512e9,
        link_latency=500e-9,
    )
    switch = KVSwitch(
        name="Grace Hopper NVLink/PCIe fabric",
        fabric_bandwidth=512e9,
        base_latency=250e-9,
        kv_dma_bandwidth=546e9,
        kv_issue_cycles=200,  # GPU + CPU load/stores to shuttle KV
    )
    return KVCacheSystem(
        cpu=cpu,
        gpu=gpu,
        kv_dram=kv_dram,
        switch=switch,
        label="Plain H100 host-offload",
        gpu_kv_mode="cpu_bounce",
    )


KV_SYSTEM_PRESETS: Dict[str, KVCacheSystem] = {
    "h100_plain": _build_h100_plain_kv_system(),
    "gb200_plain": _build_gb200_plain_kv_system(),
    "gb300_plain": _build_gb300_plain_kv_system(),
}
