import torch
import torch.nn as nn
from typing import List, Optional
from einops import rearrange
import torch.nn.functional as F


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


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


def repeat_cond_per_token(cond: torch.Tensor, seqlens: List[int]) -> torch.Tensor:
    """Repeat conditioning from [B, C] to [sum_L, C] based on sequence lengths."""
    idx = torch.arange(len(seqlens), device=cond.device).repeat_interleave(
        torch.tensor(seqlens, device=cond.device)
    )
    return cond[idx]


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


def compute_global_rope_ids(H, W, window_size, shift_size=0, device="cuda"):
    """
    Compute global position indices for windowed attention with optional shifting.

    Args:
        H, W: feature map height and width
        window_size: window size
        shift_size: shift amount for shifted window attention
        device: target device

    Returns:
        rope_ids: [num_windows * window_size², window_size²] tensor of global position indices
    """
    # Create global position grid
    h_pos = torch.arange(H, device=device)
    w_pos = torch.arange(W, device=device)
    pos_h, pos_w = torch.meshgrid(h_pos, w_pos, indexing="ij")
    global_pos = pos_h * W + pos_w  # [H, W] flattened position indices

    # Apply cyclic shift if needed
    if shift_size > 0:
        global_pos = torch.roll(
            global_pos, shifts=(-shift_size, -shift_size), dims=(0, 1)
        )

    # Partition into windows
    global_pos = global_pos.view(
        H // window_size, window_size, W // window_size, window_size
    )
    global_pos = global_pos.permute(0, 2, 1, 3).contiguous()  # [nH, nW, ws, ws]
    global_pos = global_pos.view(-1, window_size * window_size)  # [num_windows, ws²]

    return global_pos


def build_conv(conv_dim: int, conv_size: int, conv_bias: bool = False) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels=conv_dim,
        out_channels=conv_dim,
        kernel_size=conv_size,
        groups=conv_dim,
        padding=conv_size // 2,
        bias=conv_bias,
    )


def apply_conv(
    x: torch.Tensor,
    conv: nn.Conv1d,
    act: Optional[nn.Module] = None,
    residual: bool = True,
) -> torch.Tensor:
    """Apply 1D depthwise convolution with appropriate padding to maintain sequence length."""
    # x: B, N, C
    x_conv = x.transpose(1, 2)  # B, C, N
    x_conv = conv(x_conv)
    if conv.kernel_size[0] % 2 == 0:
        x_conv = x_conv[:, :, :-1]  # Remove extra padding for even kernel sizes
    x_conv = x_conv.transpose(1, 2)  # B, N, C
    if act is not None:
        x_conv = act(x_conv)

    return x + x_conv if residual else x_conv


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
