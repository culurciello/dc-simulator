from dataclasses import dataclass


@dataclass(frozen=True)
class GPUConfig:
    name: str
    sustained_flops: float  # FLOPs/s (decode-mode sustained)
    hbm_bw: float           # bytes/s of effective HBM bandwidth
    max_mem_bytes: float    # bytes of HBM capacity

