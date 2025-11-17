from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LinkProfile:
    throughput: float  # bytes/s of usable bandwidth
    latency: float     # seconds of per-message latency


@dataclass(frozen=True)
class RackPreset:
    name: str
    gpu_key: str
    gpus_per_server: int
    servers_per_rack: int
    intra_server: LinkProfile   # intra-box fabric (NVLink/XGMI)
    inter_server: LinkProfile   # within-rack fabric (NVSwitch/IF link)
    inter_rack: LinkProfile     # rack-to-rack optics/IB
    notes: str = ""
    kv_system_key: Optional[str] = None
    storage_servers_per_rack: int = 2
    storage_server_capacity_bytes: float = 122e12
