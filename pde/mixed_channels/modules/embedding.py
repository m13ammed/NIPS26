import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
from typing import Optional
from einops import rearrange, repeat
from pde.mixed_channels.utils import (
    broadcat,
    precompute_freqs_cis_2d,
    apply_rotary_emb,
    rotate_half,
)


class VisionRotaryEmbeddingFast(nn.Module):
    """
    Fast EVA-02 2D RoPE with broadcasting, extended to support:
      • per-token rope_ids (routing / unsorted subsets)
      • leading CLS tokens (unrotated)
    Accepts q/k shaped (B, Hh, N, D_rot) where N may be HW or HW+extra.
    """

    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        base = torch.einsum("..., f -> ... f", t, freqs)  # (S, dim//2)
        base = repeat(base, "... n -> ... (n r)", r=2)  # (S, dim)
        freqs_2d = broadcat((base[:, None, :], base[None, :, :]), dim=-1)  # (S,S,2*dim)

        freqs_cos = freqs_2d.cos().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)
        freqs_sin = freqs_2d.sin().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        self.grid_size = ft_seq_len  # H == W
        self.rot_dim = freqs_2d.shape[-1]  # 2*dim

    def _gather_cos_sin(self, rope_ids, N, device, dtype):
        cos_table = self.freqs_cos.to(dtype=dtype, device=device)
        sin_table = self.freqs_sin.to(dtype=dtype, device=device)

        if rope_ids is None:
            assert N == cos_table.shape[0], (
                f"When rope_ids is None, expected N == HW ({cos_table.shape[0]}), got N={N}"
            )
            return cos_table.view(1, 1, N, -1), sin_table.view(1, 1, N, -1)

        rope_ids = rope_ids.to(device=device, dtype=torch.long)

        if rope_ids.dim() == 1:
            cos = cos_table.index_select(0, rope_ids).view(1, 1, N, -1)
            sin = sin_table.index_select(0, rope_ids).view(1, 1, N, -1)
            return cos, sin

        if rope_ids.dim() == 2:
            cos = cos_table[rope_ids].unsqueeze(1)  # (B,1,N,D)
            sin = sin_table[rope_ids].unsqueeze(1)  # (B,1,N,D)
            return cos, sin

        raise ValueError(
            f"rope_ids must be None, (N,), or (B,N); got {tuple(rope_ids.shape)}"
        )

    def forward(self, t: torch.Tensor, rope_ids: Optional[torch.Tensor] = None):
        """
        t: (B, Hh, N, D_rot), D_rot == self.rot_dim
        rope_ids: None | (N,) | (B,N), indexing flattened HW positions for the
                  rotated portion. If t includes CLS, supply rope_ids for the
                  spatial tail only or for the full N; both are accepted.
        """
        B, Hh, N, D = t.shape
        assert D == self.rot_dim, f"Head dim {D} must equal RoPE dim {self.rot_dim}"

        HW = self.freqs_cos.shape[0]

        # Determine how many leading tokens to leave unrotated (CLS or others).
        if rope_ids is None:
            # No ids -> sequence must be either [HW] or [extra + HW] in default grid order.
            if N == HW:
                extra = 0
            else:
                assert N >= HW, f"N={N} shorter than HW={HW}"
                extra = N - HW
        else:
            # ids given for either full N or just the spatial tail.
            ids_len = rope_ids.shape[-1]
            if ids_len == N:
                extra = max(0, N - HW)  # assume any surplus is leading CLS tokens
            else:
                # ids describe only the spatial tail
                extra = N - ids_len
                assert extra >= 0, "rope_ids longer than sequence length"
                # quick sanity: spatial tail size should match HW when not routing
                # but allow routing to keep arbitrary N_tail
            # Ensure the tail length we will rotate is positive
        assert extra <= N, "extra leading tokens exceeds sequence length"

        if extra > 0:
            t_lead = t[:, :, :extra, :]  # unrotated CLS or similar
            t_tail = t[:, :, extra:, :]  # rotate these
            ids_tail = (
                None
                if rope_ids is None
                else (
                    rope_ids
                    if rope_ids.shape[-1] == t_tail.shape[-2]
                    else rope_ids[..., extra:]
                )
            )
        else:
            t_lead = None
            t_tail = t
            ids_tail = rope_ids

        if t_tail.shape[-2] == 0:
            # Only CLS present
            return t

        cos, sin = self._gather_cos_sin(ids_tail, t_tail.shape[-2], t.device, t.dtype)
        rot = t_tail * cos + rotate_half(t_tail) * sin

        return rot if t_lead is None else torch.cat([t_lead, rot], dim=-2)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)  # Match MLP dtype (e.g., bfloat16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class SimplePatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, patch_size=4, bias=True):
        super(SimplePatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x):
        x = self.proj(x)

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_c: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
        overlap_size: int = 1,
        bias: bool = False,
    ):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c,
            embed_dim,
            kernel_size=(patch_size + 2 * overlap_size, patch_size + 2 * overlap_size),
            stride=patch_size,
            padding=overlap_size,
            bias=bias,
        )

        self.patch_size = patch_size

    def forward(self, x, periodic_x: bool = False, periodic_y: bool = False):

        # x shape = (B, C, H, W)
        if periodic_x:
            x1 = x[:, :, :, : self.patch_size]
            x2 = x[:, :, :, -self.patch_size :]
            x = torch.cat((x2, x, x1), dim=-1)

        if periodic_y:
            x1 = x[:, :, : self.patch_size, :]
            x2 = x[:, :, -self.patch_size :, :]
            x = torch.cat((x2, x, x1), dim=-2)

        x = self.proj(x)

        if periodic_x:
            x = x[:, :, :, 1:-1]

        if periodic_y:
            x = x[:, :, 1:-1, :]

        return x


class PosEmbMLPSwinv2D(nn.Module):
    def __init__(
        self,
        window_size: list[int],
        pretrained_window_size: list[int],
        num_heads: int,
        no_log=False,
    ):
        super().__init__()

        self.window_size = [int(ws) for ws in window_size]
        self.num_heads = num_heads

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )

        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
            )

        self.register_buffer(
            "relative_coords_table", relative_coords_table, persistent=False
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).int()

        self.register_buffer(
            "relative_position_index", relative_position_index, persistent=False
        )

        self.pos_emb = None

    def forward(self, input_tensor, local_window_size):

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        n_global_feature = input_tensor.shape[2] - local_window_size

        relative_position_bias = torch.nn.functional.pad(
            relative_position_bias, (n_global_feature, 0, n_global_feature, 0)
        ).contiguous()

        self.pos_emb = relative_position_bias.unsqueeze(0)

        input_tensor += self.pos_emb
        return input_tensor


class CarrierTokenAttention2DTimestep(nn.Module):
    def __init__(
        self, dim, num_heads, bias=False, posemb_type="rope2d", attn_type="v2", **kwargs
    ):

        super(CarrierTokenAttention2DTimestep, self).__init__()
        if kwargs != dict():  # is not empty
            print(f"Kwargs: {kwargs}")

        self.dim = dim
        self.heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)

        # v2
        self.attn_type = attn_type
        if attn_type == "v2":
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        self.dh = 1
        self.dw = 1

        # posemb
        self.posemb_type = posemb_type

        # posemb type
        if self.posemb_type == "rope2d":
            self.freqs_cis = None

    def forward(self, x):

        b, n, c = x.size()
        x = torch.unsqueeze(x, 1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if self.posemb_type == "rope2d":
            if self.freqs_cis is None or self.freqs_cis.shape[0] != n:
                self.freqs_cis = precompute_freqs_cis_2d(self.dim // self.heads, n).to(
                    x.device
                )

            # q, k input shape: B N H Hc
            q, k = map(
                lambda t: rearrange(t, "b p n (h d) -> (b p) n h d", h=self.heads),
                qkv[:-1],
            )

            v = rearrange(qkv[2], "b p n (h d) -> b p h n d", h=self.heads)

            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)

            q = rearrange(q, "(b p) n h d -> b p h n d", b=b)
            k = rearrange(k, "(b p) n h d -> b p h n d", b=b)

        else:
            q, k, v = map(
                lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=self.heads), qkv
            )

        if self.attn_type is None:  # v1 attention
            attn = q @ k.transpose(-2, -1)
            attn = attn * self.scale

        elif self.attn_type == "v2":  # v2 attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = rearrange(x, "b p h n d -> b p n (h d)")

        x = x[:, 0]

        return x


class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self, dim, rank=2, conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(
                nn.Linear(self.rank, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, dim, bias=False),
            )
        else:
            self.cpb_mlp = nn.Sequential(
                nn.Conv1d(self.rank, 512, 1, bias=True),
                nn.ReLU(),
                nn.Conv1d(512, dim, 1, bias=False),
            )
        self.grid_exists = False
        self.pos_emb = None
        self.conv = conv

    def forward(self, input_tensor):

        if self.rank == 1:
            seq_length = (
                input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
            )

            relative_coords_h = torch.arange(
                0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
            )
            relative_coords_h -= seq_length // 2
            relative_coords_h /= seq_length // 2
            relative_coords_table = relative_coords_h
            self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))

        else:
            height = input_tensor.shape[1]
            width = input_tensor.shape[2]

            relative_coords_h = torch.arange(
                0, height, device=input_tensor.device, dtype=input_tensor.dtype
            )
            relative_coords_w = torch.arange(
                0, width, device=input_tensor.device, dtype=input_tensor.dtype
            )
            relative_coords_table = (
                torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
                .contiguous()
                .unsqueeze(0)
            )
            relative_coords_table[:, 0] -= height // 2
            relative_coords_table[:, 1] -= width // 2
            relative_coords_table[:, 0] /= max(
                (height // 2), 1.0
            )  # special case for 1x1
            relative_coords_table[:, 1] /= max(
                (width // 2), 1.0
            )  # special case for 1x1
            if not self.conv:
                # self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1,2))
                self.pos_emb = self.cpb_mlp(relative_coords_table.permute(0, 2, 3, 1))
            else:
                self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))

        input_tensor = input_tensor + self.pos_emb
        return input_tensor
