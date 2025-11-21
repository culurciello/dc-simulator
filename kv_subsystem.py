from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class KVComponent:
    name: str
    role: str
    memory_bytes: float
    link_bandwidth: float  # bytes/s to the switch
    link_latency: float    # seconds one-way


@dataclass(frozen=True)
class KVSwitch:
    name: str
    fabric_bandwidth: float  # bytes/s sustained across the crossbar
    base_latency: float      # seconds added by the traversal
    kv_dma_bandwidth: float  # bytes/s dedicated GPU↔KV DMA engine


@dataclass(frozen=True)
class KVLinkProfile:
    src: str
    dst: str
    bandwidth_bytes: float
    latency_seconds: float
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.src}↔{self.dst}"


class KVCacheSystem:
    """
    Models the KV tier topology (Grace CPU + GB200 GPU + KV DRAM sled + switch).
    """
    def __init__(
        self,
        cpu: KVComponent,
        gpu: KVComponent,
        kv_dram: KVComponent,
        switch: KVSwitch,
        label: str,
    ):
        self._cpu = cpu
        self._gpu = gpu
        self._kv_dram = kv_dram
        self._switch = switch
        self._label = label
        self._components: Dict[str, KVComponent] = {
            cpu.role.lower(): cpu,
            gpu.role.lower(): gpu,
            kv_dram.role.lower(): kv_dram,
        }

    @property
    def label(self) -> str:
        return self._label

    @property
    def kv_capacity_bytes(self) -> float:
        return self._kv_dram.memory_bytes

    def _component(self, role: str) -> KVComponent:
        key = role.lower()
        if key not in self._components:
            raise KeyError(f"Unknown component role: {role}")
        return self._components[key]

    def _is_dma_pair(self, role_a: str, role_b: str) -> bool:
        roles = {role_a.lower(), role_b.lower()}
        return roles == {"gpu", "kv_dram"}

    def _path_bandwidth(self, src: KVComponent, dst: KVComponent) -> float:
        limit = min(src.link_bandwidth, dst.link_bandwidth, self._switch.fabric_bandwidth)
        if self._is_dma_pair(src.role, dst.role):
            limit = min(limit, self._switch.kv_dma_bandwidth)
        return limit

    def _path_latency(self, src: KVComponent, dst: KVComponent) -> float:
        return src.link_latency + self._switch.base_latency + dst.link_latency

    def path(self, src_role: str, dst_role: str, notes: str = "") -> KVLinkProfile:
        src = self._component(src_role)
        dst = self._component(dst_role)
        bandwidth = self._path_bandwidth(src, dst)
        latency = self._path_latency(src, dst)
        if not notes and self._is_dma_pair(src.role, dst.role):
            notes = "KV DMA direct path"
        return KVLinkProfile(
            src=src.name,
            dst=dst.name,
            bandwidth_bytes=bandwidth,
            latency_seconds=latency,
            notes=notes,
        )

    def kv_data_paths(self) -> List[KVLinkProfile]:
        return [
            self.path("gpu", "kv_dram", notes="GPU eviction + reload"),
            self.path("gpu", "cpu", notes="control / fallback DMA"),
            self.path("cpu", "kv_dram", notes="CPU orchestration when needed"),
        ]

    def transfer_time_seconds(self, bytes_to_move: float, path: KVLinkProfile) -> float:
        if bytes_to_move <= 0.0 or path.bandwidth_bytes <= 0.0:
            return 0.0
        return bytes_to_move / path.bandwidth_bytes

    def describe(self) -> List[str]:
        rows = []
        rows.append(
            f"{self._switch.name}: fabric {self._switch.fabric_bandwidth/1e12:.2f} TB/s, "
            f"base latency {self._switch.base_latency*1e9:.0f} ns"
        )
        rows.append(f"  KV DMA engine: {self._switch.kv_dma_bandwidth/1e12:.2f} TB/s dedicated")
        for comp in (self._cpu, self._gpu, self._kv_dram):
            mem_tb = comp.memory_bytes / (1024**4)
            rows.append(
                f"  {comp.name} ({comp.role}) ->KVSwitch: {comp.link_bandwidth/1e9:.0f} GB/s, "
                f"lat {comp.link_latency*1e9:.0f} ns, capacity {mem_tb:.1f} TiB"
            )
        return rows

def _kv_registry() -> Dict[str, KVCacheSystem]:
    from presets import KV_SYSTEM_PRESETS

    return KV_SYSTEM_PRESETS


def resolve_kv_system_key(key: str) -> str:
    presets = _kv_registry()
    lookup = key.lower()
    if lookup not in presets:
        raise KeyError(f"Unknown KV system '{key}'. Known keys: {', '.join(sorted(presets))}")
    return lookup

def get_kv_system(key: str) -> KVCacheSystem:
    presets = _kv_registry()
    return presets[resolve_kv_system_key(key)]


def kv_system_choices() -> List[str]:
    presets = _kv_registry()
    return sorted(presets.keys())


__all__ = [
    "KVComponent",
    "KVSwitch",
    "KVLinkProfile",
    "KVCacheSystem",
    "resolve_kv_system_key",
    "get_kv_system",
    "kv_system_choices",
]
