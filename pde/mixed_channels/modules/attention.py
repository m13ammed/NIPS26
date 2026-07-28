from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func

from pde.mixed_channels.modules.embedding import (
    PosEmbMLPSwinv2D,
    VisionRotaryEmbeddingFast,
)
from pde.mixed_channels.utils import apply_conv, build_conv


class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, dim, window_size):
        """
        Args:
            dim: feature size dimension.
            window_size: window size.
        """
        super().__init__()

        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)

        self.to_global_feature = to_global_feature
        self.window_size = window_size

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        x = self.to_global_feature(x)

        B, C, H, W = x.shape

        pad_right = (self.window_size - W % self.window_size) % self.window_size
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(
            x, (0, pad_right, 0, pad_bottom, 0, 0, 0, 0), mode="constant", value=0
        )

        x = torch.nn.functional.avg_pool2d(
            x,
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size),
            divisor_override=(self.window_size - pad_right)
            * (self.window_size - pad_bottom),
            padding=0,
        )

        x = x.permute(0, 2, 3, 1)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: str | None = None,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        norm_eps: float = 1e-6,
        use_flash_attn: bool = True,
        use_v1_res: bool = False,
        use_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert qk_norm in (
            None,
            "logit",
            "layer",
        ), "qk_norm should be None, 'logit' or 'layer'"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_flash_attn = use_flash_attn
        self.qk_norm = qk_norm
        self.use_v1_res = use_v1_res
        self.use_conv = use_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Defaults for all modes
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.scale = self.head_dim**-0.5

        if self.qk_norm == "logit":  # matches WindowAttention2DTime
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )
            self.scale = 1.0

        elif self.qk_norm == "layer":
            self.q_norm = norm_layer(self.head_dim, eps=norm_eps)
            self.k_norm = norm_layer(self.head_dim, eps=norm_eps)
            self.scale = self.head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_v1_res:
            self.v1_lambda = nn.Parameter(torch.tensor(0.5))
        self.v_last = None

        if use_conv:  # Canon B
            self.conv_q = build_conv(dim, conv_size=conv_size, conv_bias=conv_bias)
            self.conv_k = build_conv(dim, conv_size=conv_size, conv_bias=conv_bias)
            self.conv_v = build_conv(dim, conv_size=conv_size, conv_bias=conv_bias)
            self.conv_act = nn.SiLU()

    def _apply_logit_norm_and_scale(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k shapes:
        #   varlen: [T, H, D]
        #   batched flash: [B, N, H, D]
        #   fallback: [B, H, N, D]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()

        if q.dim() == 3:
            # [T, H, D] * [1, H, 1]
            q = q * logit_scale.view(1, self.num_heads, 1)
        elif q.dim() == 4:
            if q.shape[2] == self.num_heads:
                # [B, N, H, D] * [1,1,H,1]
                q = q * logit_scale.view(1, 1, self.num_heads, 1)
            elif q.shape[1] == self.num_heads:
                # [B, H, N, D] * [1,H,1,1]
                q = q * logit_scale.view(1, self.num_heads, 1, 1)
            else:
                raise ValueError(f"Unexpected q shape for logit mode: {tuple(q.shape)}")
        else:
            raise ValueError(f"Unsupported q dim for logit mode: {q.dim()}")

        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        rope_ids: Optional[torch.Tensor] = None,
        rope: Optional[VisionRotaryEmbeddingFast] = None,
        v1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.dim() in (
            2,
            3,
        ), f"Expected x to have shape [T, C] or [B, N, C], got {tuple(x.shape)}"
        assert rope_ids is None or rope_ids.dim() == 2, (
            f"Expected rope_ids to have shape [B, N], got {tuple(rope_ids.shape)}"
        )
        if rope_ids is not None:
            assert rope is not None, (
                "rope module must be provided if rope_ids are given"
            )
        if rope is not None:
            assert rope_ids is not None, (
                "rope_ids must be provided if rope module is given"
            )
        if self.use_v1_res:
            assert v1 is not None, "v1 tensor must be provided if use_v1_res is True"

        if x.dim() == 3:
            B, N, C = x.shape
        else:
            B, N, C = 1, x.shape[0], x.shape[1]

        qkv = self.qkv(x).reshape(B, 3, N, C)

        if cu_seqlens is not None and max_seqlen is not None:
            # Flash attention varlen path (packed).
            assert x.dim() == 2, (
                f"FlashAttention varlen path expects packed tokens of shape [total_tokens, C]. "
                f"Got {tuple(x.shape)}."
            )

            qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)  # [T, 3, H, D]
            q, k, v = qkv.unbind(1)  # [T, H, D]

            if self.qk_norm == "logit":
                q, k = self._apply_logit_norm_and_scale(q, k)
            elif self.qk_norm == "layer":
                q, k = self.q_norm(q), self.k_norm(k)

            cu_seqlens = cu_seqlens.to(torch.int32)

            # flash-attn only supports fp16/bf16.
            if q.dtype not in (torch.float16, torch.bfloat16):
                if q.is_cuda and torch.cuda.is_bf16_supported():
                    cast_dtype = torch.bfloat16
                else:
                    cast_dtype = torch.float16
                q = q.to(dtype=cast_dtype)
                k = k.to(dtype=cast_dtype)
                v = v.to(dtype=cast_dtype)

            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,  # 1.0 in logit mode, head_dim**-0.5 otherwise
                causal=False,
            )
            if out.dtype != x.dtype:
                out = out.to(dtype=x.dtype)
            out = out.reshape(-1, C)  # [T, C]
        else:
            # Standard attention path (batched [B, N, C])
            if x.dim() != 3:
                x = x.unsqueeze(0)
                B, N, C = x.shape

            if self.use_flash_attn and x.is_cuda:
                q, k, v = qkv.unbind(1)  # each is [B, N, C]
                if self.use_conv:
                    q = apply_conv(q, self.conv_q, self.conv_act)
                    k = apply_conv(k, self.conv_k, self.conv_act)
                    v = apply_conv(v, self.conv_v, self.conv_act)

                q = q.reshape(B, self.num_heads, N, self.head_dim)
                k = k.reshape(B, self.num_heads, N, self.head_dim)
                v = v.reshape(B, self.num_heads, N, self.head_dim)
                self.v_last = v

                if self.qk_norm == "logit":
                    q, k = self._apply_logit_norm_and_scale(q, k)
                elif self.qk_norm == "layer":
                    q, k = self.q_norm(q), self.k_norm(k)

                if rope is not None and rope_ids is not None:
                    q = rope(q, rope_ids)
                    k = rope(k, rope_ids)

                if self.use_v1_res and v1 is not None:
                    v = self.v1_lambda * v1 + (1 - self.v1_lambda) * v

                q = q.reshape(B, N, self.num_heads, self.head_dim)
                k = k.reshape(B, N, self.num_heads, self.head_dim)
                v = v.reshape(B, N, self.num_heads, self.head_dim)

                orig_dtype = q.dtype
                if q.dtype not in (torch.float16, torch.bfloat16):
                    if torch.cuda.is_bf16_supported():
                        cast_dtype = torch.bfloat16
                    else:
                        cast_dtype = torch.float16
                    q = q.to(dtype=cast_dtype)
                    k = k.to(dtype=cast_dtype)
                    v = v.to(dtype=cast_dtype)

                out = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,  # 1.0 in logit mode
                    causal=False,
                )
                if out.dtype != orig_dtype:
                    out = out.to(dtype=orig_dtype)
                out = out.reshape(B, N, C)
            else:
                # Fallback to scaled_dot_product_attention
                q, k, v = qkv.unbind(1)  # each is [B, N, C]
                if self.use_conv:
                    q = apply_conv(q, self.conv_q, self.conv_act)
                    k = apply_conv(k, self.conv_k, self.conv_act)
                    v = apply_conv(v, self.conv_v, self.conv_act)

                q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

                self.v_last = v
                if self.use_v1_res and v1 is not None:
                    v = self.v1_lambda * v1 + (1 - self.v1_lambda) * v

                if self.qk_norm == "logit":
                    # Manual attention to avoid implicit 1/sqrt(d) scaling in scaled_dot_product_attention
                    q, k = self._apply_logit_norm_and_scale(q, k)
                    if rope is not None and rope_ids is not None:
                        q = rope(q, rope_ids)
                        k = rope(k, rope_ids)
                    attn = q @ k.transpose(-2, -1)  # [B, H, N, N]
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                else:
                    if self.qk_norm == "layer":
                        q, k = self.q_norm(q), self.k_norm(k)
                    if rope is not None and rope_ids is not None:
                        q = rope(q, rope_ids)
                        k = rope(k, rope_ids)
                    out = (
                        F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            dropout_p=self.attn_drop.p if self.training else 0.0,
                        )
                        .transpose(1, 2)
                        .reshape(B, N, C)
                    )

            out = out.squeeze(0) if out.shape[0] == 1 else out

        x = self.proj(out)
        x = self.proj_drop(x)
        return x


# @torch.compile
class WindowAttention2DTime(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        resolution: int = 0,
        use_output_proj: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.resolution = resolution

        if use_output_proj:
            self.proj = nn.Linear(dim, dim, bias=True)
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()

        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=num_heads,
        )

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

    def _get_pos_bias(self, q):
        _, num_heads, N, _ = q.shape
        dummy = torch.zeros(1, num_heads, N, N).to(q.dtype).to(q.device)
        self.pos_emb_funct(dummy, self.resolution**2)
        return self.pos_emb_funct.pos_emb  # (1, H, N, N)

    def forward(self, x, attn_mask=None):
        B, N, C = (
            x.shape
        )  # B: batch size*num_windows, N: window_size*window_size, C: channels (dim)
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # [3, B, N, C]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.reshape(B, self.num_heads, N, C // self.num_heads)
        k = k.reshape(B, self.num_heads, N, C // self.num_heads)
        v = v.reshape(B, self.num_heads, N, C // self.num_heads)

        q = F.normalize(q, dim=-1) * torch.clamp(self.logit_scale, max=4.6052).exp()
        k = F.normalize(k, dim=-1)

        bias = self._get_pos_bias(q)

        if attn_mask is not None:
            bias += attn_mask

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            scale=1.0,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
