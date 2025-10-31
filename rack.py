from dataclasses import dataclass


@dataclass(frozen=True)
class RackPreset:
    name: str
    gpu_key: str
    gpus_per_server: int
    servers_per_rack: int
    intra_server_bw: float      # bytes/s (e.g., NVLink/XGMI)
    inter_server_bw: float      # bytes/s (within rack, NVSwitch/IF link)
    inter_rack_bw: float        # bytes/s (rack-to-rack optics/IB)
    notes: str = ""

