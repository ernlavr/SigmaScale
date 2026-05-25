import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional

import train_utils.svd_smoother as svd_smoother
import utils.low_rank_utils as lru
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen3ForCausalLM, Qwen3Config


# ---------------------------------------------------------------------------
# Custom HuggingFace config & model for SVD-compressed Llama
# ---------------------------------------------------------------------------

class SVDLlamaConfig(LlamaConfig):
    """
    Extends LlamaConfig with per-layer SVD rank information.

    ``svd_layers_config`` maps each replaced module's dot-separated path
    (as returned by ``model.named_modules()``) to the truncation rank used,
    e.g. ``{"model.layers.0.self_attn.q_proj": 64, ...}``.
    """
    model_type = "svd_llama"

    def __init__(self, svd_layers_config: Optional[Dict[str, int]] = None, **kwargs):
        self.svd_layers_config = svd_layers_config or {}
        super().__init__(**kwargs)

class SVDLlamaForCausalLM(LlamaForCausalLM):
    """
    LlamaForCausalLM variant that reconstructs SVDLinear modules from the
    config on instantiation.  Used as the target class when calling
    ``SVDLlamaForCausalLM.from_pretrained(saved_dir)``.
    """
    config_class = SVDLlamaConfig

    def __init__(self, config: SVDLlamaConfig):
        super().__init__(config)
        
        if not config.svd_layers_config:
            return

        # Build a flat {path: module} map once — O(N) instead of O(N * depth)
        module_map = {name: mod for name, mod in self.named_modules()}

        for module_path, rank in config.svd_layers_config.items():
            parts = module_path.rsplit(".", 1)
            parent_path, attr = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
            parent = module_map[parent_path] if parent_path else self
            
            original = getattr(parent, attr)
            out_f, in_f = original.weight.shape
            has_bias = original.bias is not None
            dtype = original.weight.dtype
            device = original.weight.device  # 'meta' during from_pretrained

            L = torch.zeros(out_f, rank, dtype=dtype, device=device)
            R = torch.zeros(rank, in_f, dtype=dtype, device=device)
            bias_placeholder = (
                torch.zeros(out_f, dtype=dtype, device=device) if has_bias else None
            )
            setattr(parent, attr, SVDLinear(L, R, bias_placeholder))
        
class SVDQwenConfig(Qwen3Config):
    """
    Extends Qwen3Config with per-layer SVD rank information.

    ``svd_layers_config`` maps each replaced module's dot-separated path
    (as returned by ``model.named_modules()``) to the truncation rank used,
    e.g. ``{"model.layers.0.self_attn.q_proj": 64, ...}``.
    """
    model_type = "svd_qwen"

    def __init__(self, svd_layers_config: Optional[Dict[str, int]] = None, **kwargs):
        self.svd_layers_config = svd_layers_config or {}
        super().__init__(**kwargs)
        
class SVDQwenForCausalLM(Qwen3ForCausalLM):
    """
    QwenForCausalLM variant that reconstructs SVDLinear modules from the
    config on instantiation.  Used as the target class when calling
    ``SVDQwenForCausalLM.from_pretrained(saved_dir)``.
    """
    config_class = SVDQwenConfig

    def __init__(self, config: SVDQwenConfig):
        super().__init__(config)
        
        if not config.svd_layers_config:
            return

        # Build a flat {path: module} map once — O(N) instead of O(N * depth)
        module_map = {name: mod for name, mod in self.named_modules()}

        for module_path, rank in config.svd_layers_config.items():
            parts = module_path.rsplit(".", 1)
            parent_path, attr = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
            parent = module_map[parent_path] if parent_path else self
            
            original = getattr(parent, attr)
            out_f, in_f = original.weight.shape
            has_bias = original.bias is not None
            dtype = original.weight.dtype
            device = original.weight.device  # 'meta' during from_pretrained

            L = torch.zeros(out_f, rank, dtype=dtype, device=device)
            R = torch.zeros(rank, in_f, dtype=dtype, device=device)
            bias_placeholder = (
                torch.zeros(out_f, dtype=dtype, device=device) if has_bias else None
            )
            setattr(parent, attr, SVDLinear(L, R, bias_placeholder))


# Register so that AutoConfig / AutoModelForCausalLM work transparently.
AutoConfig.register("svd_llama", SVDLlamaConfig)
AutoConfig.register("svd_qwen", SVDQwenConfig)
AutoModelForCausalLM.register(SVDLlamaConfig, SVDLlamaForCausalLM)
AutoModelForCausalLM.register(SVDQwenConfig, SVDQwenForCausalLM)


# ---------------------------------------------------------------------------
# SVDLinear  (merged from the previous SVDLinear + SVDLinear_Smoothed)
# ---------------------------------------------------------------------------

class SVDLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using a low-rank factorisation
    W ≈ A @ B, where A has shape (out_features, rank) and B has shape
    (rank, in_features).

    The forward pass executes two small matmuls instead of one large one:
        y = ALinear(BLinear(x)) + bias

    Construction
    ------------
    * ``SVDLinear(L, R, bias)``                      — from pre-computed factors
    * ``SVDLinear.from_svd(U, S, V, bias, ...)``     — from SVD components
    * ``SVDLinear.from_linear(linear, ratio, ...)``  — compress an nn.Linear
    * ``SVDLinear.from_linear_with_trained_smoothing(...)`` — gradient-trained
    """

    def __init__(
        self,
        L: torch.Tensor,
        R: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        out_features, rank = L.shape
        in_features = R.shape[1]
        assert R.shape[0] == rank, f"L/R rank mismatch: L has rank {rank}, R has rank {R.shape[0]}"

        # ALinear : (rank,) → (out_features,)
        self.ALinear = nn.Linear(rank, out_features, bias=bias is not None)
        if L.device.type != "meta":
            self.ALinear.weight.data = L.contiguous()
            if bias is not None:
                self.ALinear.bias.data = bias.contiguous()

        # BLinear : (in_features,) → (rank,)
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        if R.device.type != "meta":
            self.BLinear.weight.data = R.contiguous()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct the full weight matrix from the factors."""
        return self.ALinear.weight @ self.BLinear.weight

    @classmethod
    def from_svd(
        cls,
        U: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        sigma_fuse: str = "UV",
    ) -> "SVDLinear":
        """
        Build from the output of torch.svd_lowrank:
            U : (out_features, rank)
            S : (rank,)
            V : (in_features, rank)
        The full approximation is  U diag(S) V^T.
        """
        if sigma_fuse == "UV":
            L = U.mul(S.sqrt()).contiguous()
            R = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            L = U.mul(S).contiguous()
            R = V.t().contiguous()
        elif sigma_fuse == "V":
            L = U.contiguous()
            R = V.t().mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError(f"Unknown sigma_fuse mode: {sigma_fuse!r}")
        return cls(L, R, bias)

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware: bool = False,
        ic_split: int = 1,
        oc_split: int = 1,
        alpha: float = 1.0,
        sigma_fuse: str = "UV",
        rank_align: int = 1,
    ) -> "SVDLinear":
        """Compress an nn.Linear to target parameter ratio via truncated SVD."""
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        rank = max(1, int(np.ceil(rank / rank_align) * rank_align))

        w = linear.weight.data.float()
        if act_aware:
            scaling = torch.ones(linear.in_features, device=w.device, dtype=w.dtype)
            if hasattr(linear, "scaling_diag_matrix"):
                scaling = scaling * linear.scaling_diag_matrix ** alpha
            if hasattr(linear, "fisher_info"):
                scaling = scaling * linear.fisher_info ** alpha
            scaling = scaling + 1e-6
            w = w * scaling.view(1, -1)

        try:
            U, S, V = torch.svd_lowrank(w, q=rank)
        except Exception:
            print(f"SVD failed for {linear}, returning original linear")
            return linear  # type: ignore[return-value]

        if act_aware:
            V = V / scaling.view(-1, 1)

        if (S != S).any() or (U != U).any() or (V != V).any():
            print("NaN in SVD output, returning original linear")
            return linear  # type: ignore[return-value]

        bias = linear.bias.data if linear.bias is not None else None
        return SVDLinear.from_svd(U, S, V, bias, sigma_fuse).to(linear.weight.dtype)

    @staticmethod
    def from_linear_with_trained_smoothing(
        linear: nn.Linear,
        param_ratio: float,
        rank_align: int = 1,
        calib_data: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        args=None,
    ) -> "SVDLinear":
        """Gradient-train L and R to minimise reconstruction loss on calib_data."""
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        rank = compressed_params // (linear.in_features + linear.out_features)
        rank = max(1, int(np.ceil(rank / rank_align) * rank_align))

        num_epochs = getattr(args, "num_epochs", 100)
        
        if args.train_svd_scalers_sequentially:
            print(f"Training SVD scalers sequentially for layer with param ratio {param_ratio}...")
            L, R, _loss_history = svd_smoother.train_svd_scalers_sequentially(
                linear.weight.data, calib_data, rank, args, layername=layer_name,
                num_epochs=num_epochs, lr=1e-4,
            )
        else:
            print(f"Training SVD scalers jointly for layer with param ratio {param_ratio}...")
            L, R, _loss_history = svd_smoother.train_svd_scalers_simultaneously(
                linear.weight.data, calib_data, rank, args, layername=layer_name,
                num_epochs=num_epochs, lr=1e-4,
            )
            
        bias = linear.bias.data if linear.bias is not None else None
        return SVDLinear(L, R, bias).to(linear.weight.dtype)

    # ------------------------------------------------------------------
    # Forward / utilities
    # ------------------------------------------------------------------

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.ALinear(self.BLinear(inp))

    def to_linear(self) -> nn.Linear:
        """
        Return a standard nn.Linear whose weight equals ALinear.weight @ BLinear.weight.
        Useful for exporting a merged (non-compressed) checkpoint.
        """
        A = self.ALinear.weight.data   # (out_features, rank)
        B = self.BLinear.weight.data   # (rank, in_features)
        dtype, device = A.dtype, A.device

        full_weight = (A.float() @ B.float()).to(dtype)
        has_bias = self.ALinear.bias is not None
        linear = nn.Linear(
            full_weight.shape[1], full_weight.shape[0],
            bias=has_bias, device=device, dtype=dtype,
        )
        linear.weight.data.copy_(full_weight)
        if has_bias:
            linear.bias.data.copy_(self.ALinear.bias.data)
        return linear

    @property
    def truncation_rank(self) -> int:
        return self.BLinear.weight.shape[0]

    def extra_repr(self) -> str:
        in_f = self.BLinear.weight.shape[1]
        out_f = self.ALinear.weight.shape[0]
        return (
            f"in_features={in_f}, out_features={out_f}, "
            f"rank={self.truncation_rank}, bias={self.ALinear.bias is not None}"
        )


# ---------------------------------------------------------------------------
# GradSVDLinear  (unchanged — re-computes SVD every forward for gradient flow)
# ---------------------------------------------------------------------------

class GradSVDLinear(nn.Module):
    """Applies scaled SVD every forward pass to allow gradient through the rank."""

    def __init__(self, weight, scale, bias, rank) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(scale)
        self.bias = bias
        self.rank = rank

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware: bool = False,
        ic_split: int = 1,
        oc_split: int = 1,
        alpha: float = 1.0,
        sigma_fuse: str = "UV",
    ):
        if param_ratio >= 1:
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        w = linear.weight.data.float()
        scaling = torch.ones(linear.in_features, device=w.device, dtype=w.dtype)
        if act_aware:
            if hasattr(linear, "scaling_diag_matrix"):
                scaling = scaling * linear.scaling_diag_matrix ** alpha
            if hasattr(linear, "fisher_info"):
                scaling = scaling * linear.fisher_info ** alpha
            scaling = scaling + 1e-6
        bias = linear.bias.data if linear.bias is not None else None
        return GradSVDLinear(w, scaling, bias, rank)

    def forward(self, inp):
        w = self.weight * self.scale.view(1, -1)
        U, S, V = torch.svd_lowrank(w, q=self.rank)
        new_w = U.mul(S).mm(V.t())
        return F.linear(inp, new_w, self.bias)


# ---------------------------------------------------------------------------
# LowRankLinear  (parametric, for training from scratch)
# ---------------------------------------------------------------------------

class LowRankLinear(nn.Module):
    """
    Trainable low-rank factorisation: W ≈ L @ R
        L : (in_features, rank)  — nn.Parameter
        R : (rank, out_features) — nn.Parameter

    Forward:  y = x @ L @ R + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.L = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        self.R = nn.Parameter(torch.empty(rank, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.L, std=math.sqrt(2.0 / (self.in_features + self.rank)))
        nn.init.normal_(self.R, std=math.sqrt(2.0 / (self.rank + self.out_features)))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, self.L), self.R) + (
            self.bias if self.bias is not None else 0
        )

    def to_linear(self) -> nn.Linear:
        """Return a merged nn.Linear equivalent. W = (L @ R).T"""
        dtype, device = self.L.dtype, self.L.device
        full_weight = (self.L.data.float() @ self.R.data.float()).T.to(dtype)
        has_bias = self.bias is not None
        linear = nn.Linear(
            full_weight.shape[1], full_weight.shape[0],
            bias=has_bias, device=device, dtype=dtype,
        )
        linear.weight.data.copy_(full_weight)
        if has_bias:
            linear.bias.data.copy_(self.bias.data)
        return linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )
