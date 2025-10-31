from dataclasses import dataclass


@dataclass(frozen=True)
class QuantConfig:
    bits: int
    bytes_per_param: float
    compute_scale: float
    kv_bytes_per_elem: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_params: float
    num_layers: int
    hidden_size: int
    num_heads: int
    context_length: int       # prompt + cached tokens
    generation_window: int    # planned decode horizon per request
    experts_per_layer: int = 0
    active_experts: int = 0
    expert_param_fraction: float = 0.0
    shared_param_fraction: float = 1.0
    moe_dispatch_factor: float = 2.0  # out + back traffic multiplier per token

    @property
    def flops_per_token(self) -> float:
        return 2.0 * self.active_params

    @property
    def total_cached_tokens(self) -> int:
        return self.context_length + self.generation_window

    @property
    def is_moe(self) -> bool:
        return (
            self.experts_per_layer > 0
            and self.active_experts > 0
            and self.expert_param_fraction > 0
        )

    @property
    def active_params(self) -> float:
        if not self.is_moe:
            return self.num_params
        shared = self.num_params * self.shared_param_fraction
        expert_pool = self.num_params * self.expert_param_fraction
        expert_active = expert_pool * (self.active_experts / self.experts_per_layer)
        return shared + expert_active

