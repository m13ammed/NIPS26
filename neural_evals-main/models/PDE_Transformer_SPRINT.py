import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Optional, Tuple

from timm.models.layers import DropPath


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        bias=True,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

    def forward(
        self, emb: Optional[torch.Tensor] = None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        emb = self.linear(self.silu(emb))
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = emb.chunk(
            6, dim=1
        )
        return msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x


class GatedMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        adjust_hidden=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Typically we reduce the hidden_features by 2/3 to maintain param count
        if adjust_hidden:
            hidden_features = int(hidden_features * 2 / 3)
            hidden_features = 256 * ((hidden_features + 256 - 1) // 256)

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x shape: [B, H, W, C] (if channels last)

        # SwiGLU: (gate * silu(up)) -> down
        # Or typically: (silu(gate) * up) -> down.
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act(gate) * up
        x = self.drop(x)
        x = self.down_proj(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of IPT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv2d(
            hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(
            -1
        )
        x = self.out_proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat, keep_dim=False):
        super(Downsample, self).__init__()

        if keep_dim:
            n_feat_out = n_feat // 4
        else:
            n_feat_out = n_feat // 2

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat_out, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, keep_dim=False, use_activation=False):
        super(Upsample, self).__init__()

        if keep_dim:
            n_feat_out = n_feat * 4
        else:
            n_feat_out = n_feat * 2

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat_out, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
            nn.SiLU() if use_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.body(x)


def precompute_freqs_cis_2d(
    dim: int, end: int, theta: float = 10000.0, scale=1.0, use_cls=False
):
    H = int(end**0.5)
    # assert  H * H == end
    flat_patch_pos = torch.arange(0 if not use_cls else -1, end)  # N = end
    x_pos = flat_patch_pos % H  # N
    y_pos = flat_patch_pos // H  # N
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
    )  # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float()  # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float()  # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat(
        [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
    )  # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(end if not use_cls else end + 1, -1)
    # we need to think how to implement this for multi heads.
    # freqs_cis = torch.cat([x_cis, y_cis], dim=-1) # N, Hc/2
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x: B N H Hc/2
    # freqs_cis:  N, H*Hc/2 or  N Hc/2
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [
            1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)
        ]  # 1, N, 1, Hc/2
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)]  # 1, N, H, Hc/2
        # B, N, Hc/2
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    # xq : B N Head Ch_per_Head
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape

    input_feature = input_feature.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        num_channels,
    )
    windows = (
        input_feature.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, num_channels)
    )
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(
        -1,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        num_channels,
    )
    windows = (
        windows.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, height, width, num_channels)
    )
    return windows


def calc_entropy_map_gpu(x, kernel_size=9, num_bins=32):
    """
    GPU-based local entropy calculation using convolutions.
    x: (C, H, W) tensor on GPU
    kernel_size: size of local window (similar to disk radius * 2 + 1)
    num_bins: number of histogram bins (fewer = faster, 32 is usually enough)
    Returns: (H, W) entropy map on same device
    """
    device = x.device
    dtype = x.dtype
    C, H, W = x.shape

    # Normalize each channel to [0, 1]
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)

    # Box filter kernel for counting
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device, dtype=dtype)
    kernel = kernel / (kernel_size * kernel_size)

    # Bin edges
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=device, dtype=dtype)

    entropy_total = torch.zeros(H, W, device=device, dtype=dtype)

    for c in range(C):
        x_c = x_norm[c]  # (H, W)

        # Create binary masks for each bin: (num_bins, H, W)
        bin_masks = (x_c.unsqueeze(0) >= bin_edges[:-1].view(-1, 1, 1)) & (
            x_c.unsqueeze(0) < bin_edges[1:].view(-1, 1, 1)
        )

        # Handle edge case: last bin should include the maximum value
        bin_masks[-1] = bin_masks[-1] | (x_c == 1.0)

        # Convert to float after boolean operations
        bin_masks = bin_masks.float()

        # Apply box filter to get local probabilities
        # Shape: (num_bins, 1, H, W) -> conv -> (num_bins, 1, H, W)
        bin_masks = bin_masks.unsqueeze(1)  # (num_bins, 1, H, W)
        probs = F.conv2d(bin_masks, kernel, padding=pad)  # (num_bins, 1, H, W)
        probs = probs.squeeze(1)  # (num_bins, H, W)

        # Compute entropy: -sum(p * log2(p)), avoiding log(0)
        probs_safe = probs.clamp(min=1e-10)
        channel_entropy = -torch.sum(probs * torch.log2(probs_safe), dim=0)  # (H, W)
        entropy_total = entropy_total + channel_entropy

    return entropy_total / C


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

    def forward(self, x, attn_mask=None):

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

        # attn shape: [B, P, H, N, N]. Apply mask only when it matches N.
        if attn_mask is not None and attn_mask.shape[-2:] == (n, n):
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            elif attn_mask.ndim == 4:
                attn_mask = attn_mask.unsqueeze(1)

            if attn_mask.ndim == 5:
                attn = attn + attn_mask.to(dtype=attn.dtype, device=attn.device)

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
            num_windows = attn_mask.shape[0]
            mask_n = attn_mask.shape[-1]
            if N != mask_n:
                n_extra = N - mask_n
                if n_extra > 0:
                    # Prepend zero-mask entries for carrier/global tokens.
                    attn_mask = F.pad(attn_mask, (n_extra, 0, n_extra, 0), value=0.0)
                else:
                    attn_mask = attn_mask[:, :N, :N]

            bias = bias.view(1, 1, self.num_heads, N, N)
            bias = bias + attn_mask.view(1, num_windows, 1, N, N)
            bias = bias.expand(B // num_windows, -1, -1, -1, -1).reshape(
                B, self.num_heads, N, N
            )
        else:
            bias = bias.expand(B, -1, -1, -1)

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


class PDELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        periodic=False,
        carrier_token_active: bool = True,
        norm_type: str = "layer",
        mlp_ratio: float = 4.0,
        use_gated_mlp: bool = False,
        adjust_hidden: bool = True,
        drop_path: float = 0.0,
        use_output_proj: bool = False,
        sprint_fusion_type: str = "linear",
        sprint_drop_mode: str = "random",
        use_sprint: bool = False,
        tokens_per_block_sprint: int = 1,
        block_size_sprint: int = 2,
    ):
        super().__init__()

        self.dim = dim
        self.periodic = periodic
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.carrier_token_active = carrier_token_active
        self.norm_type = norm_type
        if norm_type == "layer":
            self.norm = nn.LayerNorm
        elif norm_type == "rms":
            self.norm = nn.RMSNorm
        self.sprint_fusion_type = sprint_fusion_type
        self.sprint_drop_mode = sprint_drop_mode
        self.use_sprint = use_sprint
        self.tokens_per_block_sprint = tokens_per_block_sprint
        self.block_size_sprint = block_size_sprint

        blocks = []
        for i in range(depth):
            block = PDEBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                use_gated_mlp=use_gated_mlp,
                adjust_hidden=adjust_hidden,
                carrier_token_active=carrier_token_active,
                norm_layer=self.norm,
                drop_path=drop_path,
                use_output_proj=use_output_proj,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        if self.carrier_token_active:
            self.global_tokenizer = TokenInitializer(dim, window_size)

        if self.use_sprint:
            self.sprint_mask_token = nn.Parameter(torch.zeros(1, dim))
            nn.init.trunc_normal_(self.sprint_mask_token, std=0.02)
            self.sprint_spread_conv = nn.Conv2d(
                dim,
                dim,
                kernel_size=2,
                padding=1,
                groups=dim,
                bias=False,
            )
            nn.init.constant_(self.sprint_spread_conv.weight, 1.0 / 4.0)
            if sprint_fusion_type == "linear":
                self.sprint_fusion_proj = nn.Linear(2 * dim, dim, bias=True)
            elif sprint_fusion_type == "gated":
                self.sprint_fusion_gate = nn.Sequential(
                    nn.Linear(2 * dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, dim),
                    nn.Sigmoid(),
                )
                self.sprint_fusion_proj_f = nn.Linear(dim, dim, bias=True)
            elif sprint_fusion_type == "mlp":
                self.sprint_fusion_proj = nn.Sequential(
                    nn.Linear(2 * dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, dim),
                )

            if self.sprint_drop_mode == "learned":
                self.importance_scorer = nn.Sequential(
                    nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(dim // 4, 1, kernel_size=1),  # (B, 1, H, W)
                )

            self.output_activation = nn.SiLU()

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def get_attn_mask(self, shift_size, height, width, dtype, device):

        if height < self.window_size or width < self.window_size:
            return None

        if self.shift_size > 0 and not self.periodic:
            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def _drop_tokens_sprint_blockwise(self, x, entropy_map=None, temperature=1.0):
        """
        Drop tokens for SPRINT using block-wise selection.
        Within each block, select multiple tokens to keep, optionally weighted by entropy.

        x: (N, C, H, W) tensor of spatial inputs
        entropy_map: (N, H, W) optional tensor of entropy values to weight selection (higher = more likely to keep)
        temperature: float, controls how much entropy influences selection (higher = more uniform, lower = more entropy-driven)

        Returns: downsampled tensor (N, C, H_out, W_out) where H_out = H//block_size * sqrt(tokens_per_block)
                 and kept indices (B, H_blocks, W_blocks, tokens_per_block)
        """
        B, C, H, W = x.shape
        assert H % self.block_size_sprint == 0 and W % self.block_size_sprint == 0, (
            f"H={H} and W={W} must be divisible by block_size={self.block_size_sprint}"
        )

        H_blocks, W_blocks = H // self.block_size_sprint, W // self.block_size_sprint
        tokens_in_block = self.block_size_sprint * self.block_size_sprint

        assert self.tokens_per_block_sprint <= tokens_in_block, (
            f"tokens_per_block ({self.tokens_per_block_sprint}) must be <= block_size^2 ({tokens_in_block})"
        )

        if self.sprint_drop_mode == "bilinear":
            # Simple bilinear downsample as a baseline (no selection, just resizing)
            x_downsampled = nn.functional.interpolate(
                x, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            # kept ids are top left corner of each block for now
            H_blocks = H // 2
            W_blocks = W // 2

            # Generate block row and column indices
            h_indices = torch.arange(H_blocks, device=x.device) * 2 * W
            w_indices = torch.arange(W_blocks, device=x.device) * 2

            # Combine to get all top-left corner indices
            ids_keep_global = (
                (h_indices.unsqueeze(1) + w_indices.unsqueeze(0))
                .repeat(B, 1, 1)
                .unsqueeze(-1)
            )  # Shape: (B, H_blocks, W_blocks, 1)
            return x_downsampled, ids_keep_global

        x = x.reshape(
            B, C, H_blocks, self.block_size_sprint, W_blocks, self.block_size_sprint
        )  # Reshape to expose blocks
        x = x.permute(
            0, 1, 2, 4, 3, 5
        )  # (B, C, H_blocks, W_blocks, block_size, block_size)
        x = x.reshape(
            B, C, H_blocks, W_blocks, tokens_in_block
        )  # flatten: (B, C, H_out, W_out, block_size^2)

        if entropy_map is not None:
            # Reshape entropy map similarly
            entropy_map = entropy_map.reshape(
                B, H_blocks, self.block_size_sprint, W_blocks, self.block_size_sprint
            )
            entropy_map = entropy_map.permute(0, 1, 3, 2, 4)
            entropy_map = entropy_map.reshape(B, H_blocks, W_blocks, tokens_in_block)

            # Convert entropy to probabilities using softmax with temperature
            # Higher entropy -> higher probability of being selected
            logits = entropy_map / temperature
            probs = torch.softmax(
                logits, dim=-1
            )  # (B, H_blocks, W_blocks, block_size^2)

            # Sample based on probabilities
            # Use Gumbel-max trick for differentiable sampling
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(probs) + 1e-10) + 1e-10
            )
            perturbed = torch.log(probs + 1e-10) + gumbel_noise
            _, ids_keep = torch.topk(
                perturbed, k=self.tokens_per_block_sprint, dim=-1
            )  # (B, H_blocks, W_blocks, k)
        else:
            # Randomly select one token from each block (uniform)
            noise = torch.rand(B, H_blocks, W_blocks, tokens_in_block, device=x.device)
            _, ids_keep = torch.topk(
                noise, k=self.tokens_per_block_sprint, dim=-1
            )  # (B, H_blocks, W_blocks, k)

        # Sort to maintain spatial order
        ids_keep = ids_keep.sort(dim=-1).values

        # Gather the selected tokens
        ids_keep_expanded = ids_keep.unsqueeze(1).expand(
            -1, C, -1, -1, -1
        )  # (B, C, H_blocks, W_blocks, k)
        x_out = torch.gather(
            x, dim=-1, index=ids_keep_expanded
        )  # (B, C, H_blocks, W_blocks, k)

        # Reshape to rectangular grid
        k_sqrt = int(self.tokens_per_block_sprint**0.5)
        assert k_sqrt * k_sqrt == self.tokens_per_block_sprint, (
            f"tokens_per_block must be a perfect square, got {self.tokens_per_block_sprint}"
        )

        x_out = x_out.reshape(B, C, H_blocks, W_blocks, k_sqrt, k_sqrt)
        x_out = x_out.permute(
            0, 1, 2, 4, 3, 5
        )  # (B, C, H_blocks, k_sqrt, W_blocks, k_sqrt)
        x_out = x_out.reshape(B, C, H_blocks * k_sqrt, W_blocks * k_sqrt)

        # Convert local block indices to global indices for reference
        h_block = torch.arange(H_blocks, device=x.device).view(1, H_blocks, 1, 1)
        w_block = torch.arange(W_blocks, device=x.device).view(1, 1, W_blocks, 1)
        h_offset = ids_keep // self.block_size_sprint
        w_offset = ids_keep % self.block_size_sprint
        h_global = (
            h_block * self.block_size_sprint + h_offset
        )  # (B, H_blocks, W_blocks, k)
        w_global = (
            w_block * self.block_size_sprint + w_offset
        )  # (B, H_blocks, W_blocks, k)
        ids_keep_global = h_global * W + w_global  # (B, H_blocks, W_blocks, k)

        return x_out, ids_keep_global

    def _calc_entropy_map(self, x):
        """
        Compute local entropy map on GPU.
        x: (C, H, W) tensor
        Returns: (H, W) entropy map
        """
        return calc_entropy_map_gpu(x, kernel_size=3, num_bins=32)

    def _pad_tokens_sprint(self, x_sparse, ids_keep_global, H_orig, W_orig):
        """
        Pad blockwise dropped tokens back to original image size.
        x_sparse: (B, C, H_out, W_out) tensor (output from _drop_tokens_sprint_blockwise)
        ids_keep_global: (B, H_out, W_out) tensor of global indices
        H_orig, W_orig: original image size
        mask_token: value to fill dropped positions (default: 0.0 for black)
        Returns: (B, C, H_orig, W_orig) tensor
        """
        B, C, H_out, W_out = x_sparse.shape
        x_full_flat = (
            self.sprint_mask_token.unsqueeze(-1)
            .expand((B, C, H_orig * W_orig))
            .clone()
            .to(device=x_sparse.device, dtype=x_sparse.dtype)
        )
        x_sparse_flat = x_sparse.reshape(B, C, H_out * W_out)
        flat_indices = ids_keep_global.reshape(B, H_out * W_out)
        x_full_flat.scatter_(
            2, flat_indices.unsqueeze(1).expand(-1, C, -1), x_sparse_flat
        )
        return x_full_flat.reshape(B, C, H_orig, W_orig).contiguous()

    def _fuse_sprint(self, f_dense, g_full):
        """
        Fuse dense (evolved) and full (original) features per layer.
        f_dense: (N, H, W, C) tensor of spread evolved features
        g_full: (N, H, W, C) tensor of lifted original features
        layer_idx: index of the layer for selecting the correct fusion projection
        """
        if self.sprint_fusion_type == "gated":
            # 1. Calculate the gate based on both inputs
            concat = torch.cat([f_dense, g_full], dim=-1)
            gate = self.sprint_fusion_gate(concat)

            # 2. Process ONLY the new features (f_dense)
            # We remove the projection/activation on g_full to preserve the "highway"
            update = self.output_activation(self.sprint_fusion_proj_f(f_dense))

            # 3. Residual connection: Old + (Gate * New)
            h = g_full + (gate * update)
        elif self.sprint_fusion_type == "linear" or self.sprint_fusion_type == "mlp":
            h = torch.cat([f_dense, g_full], dim=-1)
            h = self.sprint_fusion_proj(h)
        else:
            raise ValueError(f"Unknown fusion_type: {self.sprint_fusion_type}")
        return h

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):

        # drop tokens for SPRINT
        hidden_states_orig = hidden_states
        H_curr, W_curr = hidden_states.shape[2], hidden_states.shape[3]
        if self.use_sprint:
            importance_map = None
            if self.sprint_drop_mode == "entropy":
                entropy_maps = []
                for b in range(hidden_states.shape[0]):
                    emap = self._calc_entropy_map(hidden_states[b])
                    entropy_maps.append(emap)
                importance_map = torch.stack(entropy_maps)
            elif self.sprint_drop_mode == "var":
                importance_map = hidden_states.var(dim=1)
            elif self.sprint_drop_mode == "l2":
                importance_map = torch.norm(hidden_states, p=2, dim=1)
            elif self.sprint_drop_mode == "learned":
                importance_map = self.importance_scorer(hidden_states).squeeze(1)
                # multiply with importance map to include it in gradient flow
                hidden_states = hidden_states * (
                    1.0 + importance_map.unsqueeze(1)
                )  # 1+importance_map to retain base signal

            hidden_states, ids_keep = self._drop_tokens_sprint_blockwise(
                hidden_states, temperature=0.25, entropy_map=importance_map
            )

        B, C, H, W = hidden_states.shape

        # precompute attention mask
        attn_mask_precomputed = self.get_attn_mask(
            self.window_size // 2, H, W, hidden_states.dtype, hidden_states.device
        )

        for n, block in enumerate(self.blocks):
            shift_size = 0 if n % 2 == 0 else self.window_size // 2

            # channels last
            hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))

            if shift_size > 0:
                attn_mask = attn_mask_precomputed
                shifted_hidden_states = torch.roll(
                    hidden_states, shifts=(-shift_size, -shift_size), dims=(1, 2)
                )
            else:
                attn_mask = None
                shifted_hidden_states = hidden_states

            shifted_hidden_states, pad_values = self.maybe_pad(
                shifted_hidden_states, H, W
            )
            _, height_pad, width_pad, _ = shifted_hidden_states.shape

            if self.carrier_token_active:
                ct = self.global_tokenizer(hidden_states)
            else:
                ct = None

            hidden_states = window_partition(shifted_hidden_states, self.window_size)

            hidden_states, ct = block(hidden_states, ct, emb=cond, attn_mask=attn_mask)

            hidden_states = window_reverse(
                hidden_states, self.window_size, height_pad, width_pad
            )

            if height_pad > 0 or width_pad > 0:
                hidden_states = hidden_states[:, :H, :W, :].contiguous()

            if shift_size > 0:
                hidden_states = torch.roll(
                    hidden_states, shifts=(shift_size, shift_size), dims=(1, 2)
                )

            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))

        # pad back dropped tokens for SPRINT
        if self.use_sprint:
            hidden_states = self._pad_tokens_sprint(
                hidden_states, ids_keep, H_curr, W_curr
            )
            hidden_states = self.sprint_spread_conv(hidden_states)[
                :, :, :H_curr, :W_curr
            ]
            hidden_states = (
                self._fuse_sprint(
                    f_dense=torch.permute(hidden_states, (0, 2, 3, 1)),
                    g_full=torch.permute(hidden_states_orig, (0, 2, 3, 1)),
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        return hidden_states


class PDEBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        use_gated_mlp=False,
        adjust_hidden=True,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        window_size=7,
        last=False,
        do_propagation=False,
        carrier_token_active=True,
        use_output_proj=False,
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            window_size: window size for sliding window attention.
            last: bool argument to indicate if this is the last block in the PDE.
            do_propagation: bool argument to indicate if this block is used for propagation.
            carrier_token_active: bool argument to indicate if carrier tokens are used.
        """
        self.norm1 = norm_layer(dim)
        self.carrier_token_active = carrier_token_active
        self.cr_window = 1
        self.window_size = window_size
        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.last = last
        self.do_propagation = do_propagation
        self.attn = WindowAttention2DTime(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
            use_output_proj=use_output_proj,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = (
            Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            if not use_gated_mlp
            else GatedMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                drop=drop,
                adjust_hidden=adjust_hidden,
            )
        )

        self.adain_2 = AdaLayerNormZero(dim)

        if self.carrier_token_active:
            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = CarrierTokenAttention2DTimestep(
                dim=dim,
                num_heads=num_heads,
            )

            self.hat_mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.hat_drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )
            self.hat_pos_embed = PosEmbMLPSwinv1D(
                dim,
                rank=2,
            )

            self.adain_1 = AdaLayerNormZero(dim)
            self.upsampler = nn.Upsample(size=window_size, mode="nearest")

    def forward(
        self,
        x,
        carrier_tokens,
        emb: Optional[torch.LongTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        B, H, W, N = x.shape
        ct = carrier_tokens

        x = x.view(B, H * W, N)

        Bc = emb.shape[0]

        if self.carrier_token_active:
            Bc, Hc, Wc, Nc = ct.shape

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)
            ct = ct.reshape(Bc, Hc * Wc, Nc)

            ######## DiT block with MSA, MLP, and AdaIN ########
            msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = (
                self.adain_1(emb=emb)
            )
            ct_msa = self.hat_norm1(ct)
            ct_msa = ct_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]

            # attention plus mlp

            ct_msa = self.hat_attn(ct_msa, attn_mask=attn_mask)

            ct_msa = ct_msa * (1 + msa_gate[:, None])
            ct = ct + self.hat_drop_path(ct_msa)

            ct_mlp = self.hat_norm2(ct)
            ct_mlp = ct_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
            ct_mlp = self.hat_mlp(ct_mlp)
            ct_mlp = ct_mlp * (1 + mlp_gate[:, None])

            ct = ct + self.hat_drop_path(ct_mlp)
            ct = ct.reshape(x.shape[0], -1, N)

            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)

        ########### DiT block with MSA, MLP, and AdaIN ############
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain_2(
            emb=emb
        )

        num_windows_total = int(B // Bc)

        msa_shift = msa_shift.repeat_interleave(num_windows_total, dim=0)
        msa_scale = msa_scale.repeat_interleave(num_windows_total, dim=0)
        msa_gate = msa_gate.repeat_interleave(num_windows_total, dim=0)
        mlp_shift = mlp_shift.repeat_interleave(num_windows_total, dim=0)
        mlp_scale = mlp_scale.repeat_interleave(num_windows_total, dim=0)
        mlp_gate = mlp_gate.repeat_interleave(num_windows_total, dim=0)

        x_msa = self.norm1(x)

        x_msa = x_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]

        x_msa = self.attn(x_msa, attn_mask=attn_mask)
        x_msa = x_msa * (1 + msa_gate[:, None])

        x = x + self.drop_path(x_msa)

        x_mlp = self.norm2(x)
        x_mlp = x_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate[:, None])
        x = x + self.drop_path(x_mlp)

        ##########################################################

        if self.carrier_token_active:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split(
                [
                    x.shape[1] - self.window_size * self.window_size,
                    self.window_size * self.window_size,
                ],
                dim=1,
            )

            ct = ctr.reshape(Bc, Hc * Wc, Nc)  # reshape carrier tokens.

            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(
                    B, N, self.cr_window, self.cr_window
                )
                x = x + self.gamma1 * self.upsampler(
                    ctr_image_space.to(dtype=torch.float32)
                ).flatten(2).transpose(1, 2).to(dtype=x.dtype)

        return x, ct


class PDEModel(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

    def __init__(
        self,
        args,
        in_channels: int = 4,
        out_channels: int = 4,
        window_size: int = 8,
        patch_size: Optional[int] = 4,
        hidden_size: int = 96,
        max_hidden_size: int = 2048,
        depth=[2, 4, 4, 6, 4, 4, 2],
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_gated_mlp: bool = False,
        adjust_hidden: bool = True,
        norm_type: str = "layer",
        output_activation: Optional[str] = None,
        use_upsample_activation: bool = False,
        use_latent_residual: bool = False,
        class_dropout_prob: float = 0.1,
        num_classes=1,
        periodic=True,
        carrier_token_active: bool = False,
        path_drop_prob: float = 0.05,
        sprint_fusion_type: str = "linear",
        sprint_drop_mode: str = "random",
        use_output_proj: bool = False,
        block_size_sprint: int = 2,
        tokens_per_block_sprint: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.args = args

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."
        assert sprint_fusion_type in [
            "linear",
            "gated",
            "mlp",
        ], f"Invalid sprint_fusion_type: {sprint_fusion_type}."
        assert sprint_drop_mode in [
            "random",
            "entropy",
            "var",
            "l2",
            "learned",
            "bilinear",
        ], f"Invalid sprint_drop_mode: {sprint_drop_mode}."
        assert output_activation in [
            "gelu",
            "silu",
            None,
        ], f"Invalid output_activation: {output_activation}."
        assert norm_type in ["layer", "rms"], f"Invalid norm_type: {norm_type}."

        self.num_encoder_layers = len(depth) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.periodic = periodic
        self.norm_type = norm_type
        self.use_carrier_tokens = carrier_token_active
        self.max_hidden_size = max_hidden_size
        self.use_latent_residual = use_latent_residual

        # SPRINT properties
        self.use_entropy = True
        self.path_drop_prob = path_drop_prob
        self.sprint_fusion_type = sprint_fusion_type
        self.sprint_drop_mode = sprint_drop_mode

        assert self.max_hidden_size >= hidden_size, (
            f"max_hidden_size {max_hidden_size} must be greater than or equal to hidden_size {hidden_size}."
        )

        dit_stage_args = {
            "drop_path": 0.0,
            "periodic": periodic,
            "carrier_token_active": carrier_token_active,
            "norm_type": norm_type,
            "mlp_ratio": mlp_ratio,
            "use_gated_mlp": use_gated_mlp,
            "adjust_hidden": adjust_hidden,
            "use_output_proj": use_output_proj,
            "sprint_fusion_type": sprint_fusion_type,
            "sprint_drop_mode": sprint_drop_mode,
            "block_size_sprint": block_size_sprint,
            "tokens_per_block_sprint": tokens_per_block_sprint,
        }

        if patch_size is not None:
            self.x_embedder = SimplePatchEmbed(
                in_channels, hidden_size, patch_size, bias=True
            )
            self.patch_size = patch_size
        else:
            self.x_embedder = OverlapPatchEmbed(in_channels, hidden_size, bias=True)
            self.patch_size = 1

        # timestep and label embedders
        for i in range(self.num_encoder_layers + 1):
            hidden_size_layer = min(hidden_size * 2**i, max_hidden_size)
            self.__setattr__(f"t_embedder_{i}", TimestepEmbedder(hidden_size_layer))
            self.__setattr__(
                f"y_embedder_{i}",
                LabelEmbedder(num_classes, hidden_size_layer, class_dropout_prob),
            )

        # encoder
        for i in range(self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2**i, max_hidden_size)
            self.__setattr__(
                f"encoder_level_{i}",
                PDELayer(
                    dim=hidden_size_layer,
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=depth[i],
                    use_sprint=True,
                    **dit_stage_args,
                ),
            )

            if hidden_size_layer == max_hidden_size:
                keep_dim = True
            else:
                keep_dim = False
            self.__setattr__(
                f"down{i}_{i + 1}", Downsample(hidden_size_layer, keep_dim=keep_dim)
            )

        # latent
        hidden_size_latent = min(
            hidden_size * 2**self.num_encoder_layers, max_hidden_size
        )
        self.latent = PDELayer(
            dim=hidden_size_latent,
            num_heads=num_heads,
            window_size=window_size,
            depth=depth[self.num_encoder_layers],
            **dit_stage_args,
        )

        hidden_size_layer0 = min(hidden_size * 2, max_hidden_size)
        if hidden_size_layer0 >= max_hidden_size:
            keep_dim = True
        else:
            keep_dim = False

        # double hidden size for last decoder layer 0
        self.__setattr__(
            "up1_0",
            Upsample(
                hidden_size_layer0,
                keep_dim=keep_dim,
                use_activation=use_upsample_activation,
            ),
        )
        self.__setattr__(
            "reduce_chan_level0",
            nn.Conv2d(
                2 * min(hidden_size, max_hidden_size),
                hidden_size_layer0,
                kernel_size=1,
                bias=True,
            ),
        )
        self.__setattr__(
            "decoder_level_0",
            PDELayer(
                dim=hidden_size_layer0,
                num_heads=num_heads,
                window_size=window_size,
                depth=depth[self.num_encoder_layers + 1],
                use_sprint=True,
                **dit_stage_args,
            ),
        )

        # decoder layers 1 - num_encoder_layers
        for i in range(1, self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2**i, max_hidden_size)
            if 2 * hidden_size_layer >= max_hidden_size:
                keep_dim = True
                hidden_size_upsample = max_hidden_size
            else:
                keep_dim = False
                hidden_size_upsample = 2 * hidden_size_layer
            self.__setattr__(
                f"up{i + 1}_{i}",
                Upsample(
                    hidden_size_upsample,
                    keep_dim=keep_dim,
                    use_activation=use_upsample_activation,
                ),
            )
            self.__setattr__(
                f"reduce_chan_level{i}",
                nn.Conv2d(
                    hidden_size_layer * 2, hidden_size_layer, kernel_size=1, bias=True
                ),
            )
            # we want that the last decoder layer uses SPRINT, so depending on whether num_encoder_layers is even or odd,
            # we alternate the use of SPRINT in the decoder layers accordingly
            self.__setattr__(
                f"decoder_level_{i}",
                PDELayer(
                    dim=hidden_size_layer,
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=depth[self.num_encoder_layers + i + 1],
                    use_sprint=True,
                    **dit_stage_args,
                ),
            )

        hidden_size_out = min(2 * hidden_size, max_hidden_size)

        if output_activation == "gelu":
            self.output_activation = nn.GELU()
        elif output_activation == "silu":
            self.output_activation = nn.SiLU()
        else:
            self.output_activation = nn.Identity()

        self.output = nn.Conv2d(
            hidden_size_out,
            hidden_size_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.final_layer = FinalLayer(
            hidden_size_out, self.out_channels * self.patch_size * self.patch_size
        )

        self.initialize_weights()

    def initialize_weights(self):

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for i in range(self.num_encoder_layers):
            # Initialize label embedding table:
            nn.init.normal_(
                self.__getattr__(f"y_embedder_{i}").embedding_table.weight, std=0.02
            )

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[2].weight, std=0.02)

        blocks = [
            self.__getattr__(f"encoder_level_{i}")
            for i in range(self.num_encoder_layers)
        ]
        blocks += [self.latent]
        blocks += [
            self.__getattr__(f"decoder_level_{i}")
            for i in range(self.num_encoder_layers)
        ]

        for block in blocks:
            for blc in block.blocks:
                nn.init.constant_(blc.adain_2.linear.weight, 0)
                nn.init.constant_(blc.adain_2.linear.bias, 0)

                if self.use_carrier_tokens:
                    nn.init.constant_(blc.adain_1.linear.weight, 0)
                    nn.init.constant_(blc.adain_1.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def forward(self, x, fx, T=None, geo=None):
        """
        Forward pass of PDE transformer.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        T: (N, ) tensor of diffusion timesteps
        """
        if self.args.geotype != "structured_2D":
            raise ValueError(f"Not supported geo: {self.args.geotype}")

        x = self.x_embedder(x)  # (B, hidden_size, H//ps, W//ps)

        if T is None:
            T = torch.Tensor([0]).to(x.device)

        if len(T.shape) == 0:
            T = T.unsqueeze(0)
            T = T.repeat(x.shape[0])
            T = T.to(x.device)

        # timestep scaling (from 0 - 1 to 0 - 1000)
        T = T * 1000.0

        # if y is None:
        y = torch.ones(x.shape[0], dtype=torch.long, device=x.device) * self.num_classes

        emb_list = []
        for i in range(self.num_encoder_layers + 1):
            t_emb = self.__getattr__(f"t_embedder_{i}")(T)
            y_emb = self.__getattr__(f"y_embedder_{i}")(y, self.training)
            c = t_emb + y_emb
            emb_list.append(c)

        residuals_list = []
        for i, c in enumerate(emb_list[:-1]):
            # encoder
            out_enc_level = self.__getattr__(f"encoder_level_{i}")(x, c)
            residuals_list.append(out_enc_level)
            x = self.__getattr__(f"down{i}_{i + 1}")(out_enc_level)

        if self.use_latent_residual:
            x_pre = x
            x = self.latent(x, emb_list[-1])
            x = x + x_pre
        else:
            x = self.latent(x, emb_list[-1])

        for i, (residual, emb) in enumerate(
            zip(residuals_list[1:][::-1], emb_list[1:-1][::-1])
        ):
            # decoder
            x = self.__getattr__(
                f"up{self.num_encoder_layers - i}_{self.num_encoder_layers - i - 1}"
            )(x)
            x = torch.cat([x, residual], 1)
            x = self.__getattr__(f"reduce_chan_level{self.num_encoder_layers - i - 1}")(
                x
            )
            x = self.__getattr__(f"decoder_level_{self.num_encoder_layers - i - 1}")(
                x, emb
            )

        x = self.__getattr__(f"up1_0")(x)
        x = torch.cat([x, residuals_list[0]], 1)
        x = self.__getattr__(f"reduce_chan_level0")(x)
        x = self.__getattr__(f"decoder_level_0")(x, emb_list[1])

        x = self.output_activation(x)
        x = self.output(x)
        x = self.final_layer(x, emb_list[1])  # (B, out_channels * ps^2, H_feat, W_feat)

        # unpatchify
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(
            shape=x.shape[:3] + (self.patch_size, self.patch_size, self.out_channels)
        )

        height = x.shape[1]
        width = x.shape[2]

        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.patch_size,
                width * self.patch_size,
            )
        )
        return x


class Model(nn.Module):
    """Adapter that wraps PDEModel for the Neural-Solver-Library interface.

    Handles:
      - Concatenation of spatial coordinates (x) and input features (fx)
      - Reshaping flat [B, N, C] inputs to [B, C, H, W] grids
      - Padding to U-Net-compatible dimensions
      - Cropping output back to original grid size
    """

    def __init__(self, args):
        super().__init__()
        self.__name__ = "PDE_Transformer_SPRINT"
        self.args = args

        if args.geotype != "structured_2D":
            raise ValueError(
                f"PDE_Transformer_SPRINT only supports structured_2D, got {args.geotype}"
            )

        self.H_orig, self.W_orig = args.shapelist

        in_channels = args.space_dim + args.fun_dim

        # Model hyperparameters (with defaults)
        patch_size = getattr(args, "pdet_patch_size", 2)
        window_size = getattr(args, "pdet_window_size", 4)
        depth_str = getattr(args, "pdet_depth", "2,4,6,4,2")
        depth = [int(d) for d in depth_str.split(",")]
        max_hidden_size = getattr(args, "pdet_max_hidden", 512)
        periodic = bool(getattr(args, "pdet_periodic", 0))
        sprint_drop_mode = getattr(args, "pdet_sprint_drop_mode", "random")
        sprint_fusion_type = getattr(args, "pdet_sprint_fusion_type", "linear")
        use_upsample_act = bool(getattr(args, "pdet_use_upsample_act", 0))
        use_gated_mlp = bool(getattr(args, "pdet_use_gated_mlp", 0))
        carrier_tokens = bool(getattr(args, "pdet_carrier_tokens", 0))
        output_activation = getattr(args, "pdet_output_act", None)
        pretrained_in = getattr(args, "pdet_pretrained_in_channels", None)
        pretrained_out = getattr(args, "pdet_pretrained_out_channels", None)
        use_pretrained = pretrained_in is not None and pretrained_out is not None

        num_encoder_layers = len(depth) // 2

        # Pad so feature maps stay divisible by window_size at every level,
        # including after SPRINT halving (block_size_sprint=2, tokens_per_block=1).
        alignment = patch_size * window_size * (2**num_encoder_layers)
        self.H_pad = math.ceil(self.H_orig / alignment) * alignment
        self.W_pad = math.ceil(self.W_orig / alignment) * alignment

        if use_pretrained:
            self.input_conv = nn.Conv2d(
                in_channels,
                int(pretrained_in),
                kernel_size=1,
                bias=True,
            )
            self.output_conv = nn.Conv2d(
                int(pretrained_out),
                args.out_dim,
                kernel_size=1,
                bias=True,
            )
            in_channels = int(pretrained_in)
            out_channels = int(pretrained_out)
        else:
            self.input_conv = nn.Identity()
            self.output_conv = nn.Identity()
            in_channels = in_channels
            out_channels = args.out_dim

        self.pde_model = PDEModel(
            args=args,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=args.n_hidden,
            max_hidden_size=max_hidden_size,
            num_heads=args.n_heads,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            mlp_ratio=float(args.mlp_ratio),
            periodic=periodic,
            num_classes=1000,
            class_dropout_prob=0.1,
            input_size=max(self.H_pad, self.W_pad) // patch_size,
            sprint_drop_mode=sprint_drop_mode,
            sprint_fusion_type=sprint_fusion_type,
            use_upsample_activation=use_upsample_act,
            use_gated_mlp=use_gated_mlp,
            carrier_token_active=carrier_tokens,
            output_activation=output_activation,
        )

    def forward(self, x, fx=None, T=None, geo=None):
        B, N, _ = x.shape
        H, W = self.H_orig, self.W_orig

        # Concatenate spatial coordinates and input features
        if fx is not None:
            inp = torch.cat([x, fx], dim=-1)
        else:
            inp = x

        # Reshape flat points to 2D grid: [B, C, H, W]
        inp = inp.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Pad to U-Net-compatible dimensions
        pad_h = self.H_pad - H
        pad_w = self.W_pad - W
        if pad_h > 0 or pad_w > 0:
            inp = F.pad(inp, (0, pad_w, 0, pad_h))

        # Handle timestep shape: [B, 1] -> [B]
        T_input = None
        if T is not None:
            T_input = T.reshape(-1)

        inp = self.input_conv(inp)
        out = self.pde_model(inp, None, T=T_input)
        out = self.output_conv(out)

        # Crop back to original spatial dimensions
        out = out[:, :, :H, :W]

        # Reshape to [B, N, out_dim]
        out = out.permute(0, 2, 3, 1).reshape(B, N, -1)

        return out
