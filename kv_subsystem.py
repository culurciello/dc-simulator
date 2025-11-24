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
    kv_issue_cycles: int = 0  # control-plane setup cycles for KV moves
    control_clock_hz: float = 1.5e9  # clock used to translate cycles→seconds


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
        gpu_kv_mode: str = "dma",  # "dma" or "cpu_bounce"
    ):
        valid_modes = {"dma", "cpu_bounce"}
        if gpu_kv_mode not in valid_modes:
            raise ValueError(f"Unknown GPU↔KV mode '{gpu_kv_mode}'. Expected one of {sorted(valid_modes)}.")
        self._cpu = cpu
        self._gpu = gpu
        self._kv_dram = kv_dram
        self._switch = switch
        self._label = label
        self._gpu_kv_mode = gpu_kv_mode
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

    def _kv_issue_latency_seconds(self) -> float:
        if self._switch.kv_issue_cycles <= 0 or self._switch.control_clock_hz <= 0:
            return 0.0
        return self._switch.kv_issue_cycles / self._switch.control_clock_hz

    def _is_dma_pair(self, role_a: str, role_b: str) -> bool:
        roles = {role_a.lower(), role_b.lower()}
        return roles == {"gpu", "kv_dram"}

    def _cpu_bounce_bandwidth(self) -> float:
        # Sequential DDR→CPU and CPU→GPU copies when the CPU has to orchestrate movement.
        to_cpu = min(
            self._gpu.link_bandwidth,
            self._cpu.link_bandwidth,
            self._switch.fabric_bandwidth,
        )
        to_kv = min(
            self._cpu.link_bandwidth,
            self._kv_dram.link_bandwidth,
            self._switch.fabric_bandwidth,
        )
        if self._switch.kv_dma_bandwidth > 0:
            to_cpu = min(to_cpu, self._switch.kv_dma_bandwidth)
            to_kv = min(to_kv, self._switch.kv_dma_bandwidth)
        if to_cpu <= 0.0 or to_kv <= 0.0:
            return 0.0
        # Effective bandwidth for two back-to-back copies.
        return 1.0 / ((1.0 / to_cpu) + (1.0 / to_kv))

    def _gpu_kv_bandwidth(self) -> float:
        if self._gpu_kv_mode == "cpu_bounce":
            return self._cpu_bounce_bandwidth()
        return min(
            self._gpu.link_bandwidth,
            self._kv_dram.link_bandwidth,
            self._switch.fabric_bandwidth,
            self._switch.kv_dma_bandwidth,
        )

    def _path_bandwidth(self, src: KVComponent, dst: KVComponent) -> float:
        return min(src.link_bandwidth, dst.link_bandwidth, self._switch.fabric_bandwidth)

    def _cpu_bounce_latency(self) -> float:
        gpu_to_cpu = self._gpu.link_latency + self._switch.base_latency + self._cpu.link_latency
        cpu_to_kv = self._cpu.link_latency + self._switch.base_latency + self._kv_dram.link_latency
        return gpu_to_cpu + cpu_to_kv

    def _gpu_kv_latency(self) -> float:
        base_latency = (
            self._cpu_bounce_latency()
            if self._gpu_kv_mode == "cpu_bounce"
            else self._gpu.link_latency + self._switch.base_latency + self._kv_dram.link_latency
        )
        return base_latency + self._kv_issue_latency_seconds()

    def _path_latency(self, src: KVComponent, dst: KVComponent) -> float:
        return src.link_latency + self._switch.base_latency + dst.link_latency

    def _kv_path_note(self) -> str:
        if self._gpu_kv_mode == "cpu_bounce":
            return "GPU eviction + reload via CPU bounce (load/store mediated)"
        return "GPU eviction + reload (KV DMA request)"

    def path(self, src_role: str, dst_role: str, notes: str = "") -> KVLinkProfile:
        src = self._component(src_role)
        dst = self._component(dst_role)
        if self._is_dma_pair(src.role, dst.role):
            bandwidth = self._gpu_kv_bandwidth()
            latency = self._gpu_kv_latency()
            if not notes:
                notes = self._kv_path_note()
        else:
            bandwidth = self._path_bandwidth(src, dst)
            latency = self._path_latency(src, dst)
        return KVLinkProfile(
            src=src.name,
            dst=dst.name,
            bandwidth_bytes=bandwidth,
            latency_seconds=latency,
            notes=notes,
        )

    def kv_data_paths(self) -> List[KVLinkProfile]:
        return [
            self.path("gpu", "kv_dram"),
            self.path("gpu", "cpu", notes="control / fallback DMA"),
            self.path("cpu", "kv_dram", notes="CPU orchestration when needed"),
        ]

    def transfer_time_seconds(self, bytes_to_move: float, path: KVLinkProfile) -> float:
        if bytes_to_move <= 0.0 or path.bandwidth_bytes <= 0.0:
            return path.latency_seconds
        return bytes_to_move / path.bandwidth_bytes + path.latency_seconds

    def describe(self) -> List[str]:
        rows = []
        rows.append(
            f"{self._switch.name}: fabric {self._switch.fabric_bandwidth/1e12:.2f} TB/s, "
            f"base latency {self._switch.base_latency*1e9:.0f} ns"
        )
        rows.append(f"  KV DMA engine: {self._switch.kv_dma_bandwidth/1e12:.2f} TB/s dedicated")
        if self._switch.kv_issue_cycles > 0 and self._switch.control_clock_hz > 0:
            issue_ns = self._kv_issue_latency_seconds() * 1e9
            rows.append(
                f"  KV issue overhead: {self._switch.kv_issue_cycles} cycles "
                f"({issue_ns:.1f} ns @ {self._switch.control_clock_hz/1e9:.2f} GHz)"
            )
        mode_desc = (
            "CPU-mediated copies (DDR→CPU→GPU)"
            if self._gpu_kv_mode == "cpu_bounce"
            else "direct GPU↔KV DMA"
        )
        rows.append(f"  GPU↔KV flow: {mode_desc}")
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
