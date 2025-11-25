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
    kv_heads: int = 0  # optional override for KV heads (GQA/MQA/MLA); defaults to num_heads
    experts_per_layer: int = 0
    active_experts: int = 0
    expert_param_fraction: float = 0.0
    shared_param_fraction: float = 1.0
    moe_dispatch_factor: float = 2.0  # out + back traffic multiplier per token

    @property
    def flops_per_token(self) -> float:
        # Base matmul cost: 2 * active params (same as typical LLM FLOPs/token).
        # Add a lightweight attention term that scales with sequence length so long
        # prompts show the expected O(seq) growth; negligible for short contexts.
        seq_len = max(self.total_cached_tokens, 1)
        attn_flops = 2.0 * self.num_layers * self.hidden_size * seq_len
        return 2.0 * self.active_params + attn_flops

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
