import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from timm.layers import trunc_normal_
except ImportError:  # pragma: no cover - timm changed its API over time
    from timm.models.layers import trunc_normal_

try:  # optional, only needed when flash attention is enabled
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
except Exception:  # pragma: no cover - allow running without flash-attn installed
    flash_attn_func = None

from layers.Embedding import timestep_embedding


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU,
}

ALLOWED_CHUNKING_MODES = {'linear', 'balltree'}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def _get_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError(f'RoPE requires an even head dimension, got {head_dim}')
    half_dim = head_dim // 2
    freq_seq = torch.arange(0, half_dim, device=device, dtype=torch.float32)
    inv_freq = base ** (-freq_seq / half_dim)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', positions, inv_freq)
    cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).reshape(seq_len, head_dim)
    sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).reshape(seq_len, head_dim)
    cos = cos.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = _rotate_half(q)
    k_rot = _rotate_half(k)
    q_out = (q * cos) + (q_rot * sin)
    k_out = (k * cos) + (k_rot * sin)
    return q_out, k_out


def _pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1):
    length = x.size(dim)
    pad_len = (multiple - (length % multiple)) % multiple
    if pad_len == 0:
        return x, 0
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = x.new_zeros(pad_shape)
    x_padded = torch.cat([x, pad_tensor], dim=dim)
    return x_padded, pad_len


def chunk_points_balltree(points: torch.Tensor, num_chunks: int):
    """Compute chunk indices using balltree-erwin partitioning."""
    try:
        from balltree import partition_balltree  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "balltree-erwin is required for balltree chunking. Install with `pip install balltree-erwin`."
        ) from exc

    if points.dim() != 2:
        raise ValueError(f"Expected 2D tensor for points, got shape {tuple(points.shape)}")

    device = points.device
    num_points = points.size(0)
    if num_points % num_chunks != 0:
        raise ValueError("Number of points must be divisible by num_chunks after padding.")

    chunk_size = num_points // num_chunks
    batch_idx = torch.zeros(num_points, dtype=torch.long, device=device)
    target_level = max(0, math.ceil(math.log2(num_chunks)))
    partition_indices = partition_balltree(points, batch_idx, target_level).long()
    if partition_indices.numel() < num_points:
        raise RuntimeError("balltree partition returned insufficient indices")

    indices = []
    start = 0
    for _ in range(num_chunks):
        indices.append(partition_indices[start:start + chunk_size])
        start += chunk_size
    return indices


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f'Unsupported activation {act}')
        activation = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
            for _ in range(n_layers)
        ])
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            residual = x
            x = layer(x)
            x = x + residual if self.res else x
        return self.linear_post(x)


class ChunkedGlobalPoolAttention(nn.Module):
    """Chunked self-attention with optional RoPE or flash attention."""

    def __init__(self, dim, heads=8, V=16, Q=1, dropout=0.1, pool='mean', use_rope=False,
                 rope_base=10000.0, use_flash_attn=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.V = V
        self.Q = Q
        self.pool = pool
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.use_flash_attn = bool(use_flash_attn and flash_attn_func is not None)

        if pool == 'linear':
            self.pool_proj = nn.Linear(dim, Q * dim)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        if self.use_rope or self.use_flash_attn:
            if dim % heads != 0:
                raise ValueError(f"embed_dim ({dim}) must be divisible by heads ({heads})")
            self.head_dim = dim // heads

        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, features, prev_supernodes=None):
        B, N, D = features.shape
        x, pad_len = _pad_to_multiple(features, self.V, dim=1)
        N_pad = x.size(1)
        seq_len = N_pad // self.V
        chunks = x.view(B, self.V, seq_len, D)

        if self.pool == 'mean':
            if self.Q == 1:
                pooled = chunks.mean(dim=2, keepdim=True)
            else:
                seq_len = chunks.size(2)
                k = min(self.Q, seq_len)
                norms = chunks.norm(dim=-1)
                order = torch.argsort(norms, dim=2, descending=True)
                running_sum = chunks.sum(dim=2)
                counts = torch.full((B, self.V), seq_len, device=chunks.device, dtype=chunks.dtype)
                means = []
                for q in range(k):
                    means.append((running_sum / counts.unsqueeze(-1)).unsqueeze(2))
                    if q == k - 1:
                        break
                    idx = order[:, :, q].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)
                    selected = torch.gather(chunks, 2, idx).squeeze(2)
                    running_sum = running_sum - selected
                    counts = counts - 1
                pooled = torch.cat(means, dim=2)
                if k < self.Q:
                    pad = torch.zeros(B, self.V, self.Q - k, D, device=chunks.device, dtype=chunks.dtype)
                    pooled = torch.cat([pooled, pad], dim=2)
            pooled = pooled.expand(B, self.V, self.Q, D) if pooled.size(2) == 1 else pooled
        elif self.pool == 'max':
            k = min(self.Q, seq_len)
            pooled, _ = chunks.topk(k=k, dim=2)
            if k < self.Q:
                pad = torch.zeros(B, self.V, self.Q - k, D, device=chunks.device, dtype=chunks.dtype)
                pooled = torch.cat([pooled, pad], dim=2)
        elif self.pool == 'linear':
            pooled = self.pool_proj(chunks.mean(dim=2)).view(B, self.V, self.Q, D)
        else:
            raise ValueError(f'Unsupported pooling {self.pool}')

        global_tokens = pooled.reshape(B, self.V * self.Q, D)
        if prev_supernodes is not None:
            if prev_supernodes.shape != global_tokens.shape:
                raise ValueError(
                    f"prev_supernodes shape {tuple(prev_supernodes.shape)} does not match expected {(B, self.V * self.Q, D)}"
                )
            prev_supernodes = prev_supernodes.to(global_tokens.device, dtype=global_tokens.dtype)
            global_tokens = global_tokens + prev_supernodes

        global_expand = global_tokens.unsqueeze(1).expand(B, self.V, -1, -1)
        chunks_with_pool = torch.cat([chunks, global_expand], dim=2)
        seq = chunks_with_pool.view(B * self.V, seq_len + self.V * self.Q, D)
        residual = seq
        seq_norm = self.norm(seq)
        attn_out = self._self_attention(seq_norm)
        seq = residual + attn_out
        seq = seq + self.ff(self.norm(seq))
        seq = seq.view(B, self.V, seq_len + self.V * self.Q, D)

        point_features = seq[:, :, :seq_len, :].reshape(B, N_pad, D)
        if pad_len > 0:
            point_features = point_features[:, :-pad_len, :]

        supernodes = seq[:, :, -self.V * self.Q:, :]
        supernodes = supernodes.mean(dim=1)

        return point_features, supernodes

    def _self_attention(self, seq_norm: torch.Tensor) -> torch.Tensor:
        if self.use_rope or self.use_flash_attn:
            return self._attention_with_custom_proj(seq_norm)
        attn_out, _ = self.attn(seq_norm, seq_norm, seq_norm, need_weights=False)
        return attn_out

    def _attention_with_custom_proj(self, seq_norm: torch.Tensor) -> torch.Tensor:
        attn = self.attn
        B, L, D = seq_norm.shape
        qkv = F.linear(seq_norm, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = _get_rope_cache(L, self.head_dim, seq_norm.device, seq_norm.dtype, base=self.rope_base)
            q, k = _apply_rope(q, k, cos, sin)

        dropout_p = attn.dropout if self.training else 0.0
        if self.use_flash_attn and seq_norm.is_cuda:
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=None, causal=False)  # type: ignore[arg-type]
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, L, D)
        return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)


class MSPTBlock(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout: float, act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, V=16, Q=1, attn_pool='mean', use_rope=False,
                 rope_base=10000.0, use_flash_attn=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = ChunkedGlobalPoolAttention(hidden_dim, heads=num_heads, dropout=dropout, V=V, Q=Q,
                                               pool=attn_pool, use_rope=use_rope, rope_base=rope_base,
                                               use_flash_attn=use_flash_attn)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, supernodes=None):
        if supernodes is None:
            B = fx.size(0)
            supernodes = fx.new_zeros(B, self.Attn.V * self.Attn.Q, self.Attn.dim)
        else:
            if supernodes.device != fx.device or supernodes.dtype != fx.dtype:
                supernodes = supernodes.to(fx.device, dtype=fx.dtype)

        attn_input = self.ln_1(fx)

        def attn_forward(inp, sup):
            return self.Attn(inp, sup)

        if self.training:
            attn_out, supernodes = checkpoint(attn_forward, attn_input, supernodes, use_reentrant=True)
        else:
            attn_out, supernodes = self.Attn(attn_input, supernodes)

        fx = fx + attn_out

        if self.training:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx)), supernodes
        return fx, supernodes


class MSPTUnstructured(nn.Module):
    def __init__(self, pos_dim=2, fx_dim=1, out_dim=1, num_blocks=5, n_hidden=256, dropout=0.1,
                 num_heads=8, Time_Input=False, act='gelu', mlp_ratio=1, ref=8, unified_pos=False,
                 Q=1, V=32, attn_pool='mean', use_rope=False, rope_base=10000.0,
                 chunking_mode='linear', distribute_blocks=False, use_flash_attn=False):
        super().__init__()
        self.__name__ = 'MSPT_Unstructured'
        self.ref = ref
        self.unified_pos = unified_pos
        self.V = V
        self.Q = Q
        self.chunking_mode = chunking_mode.lower()
        if self.chunking_mode not in ALLOWED_CHUNKING_MODES:
            raise ValueError(f"Unsupported chunking_mode '{self.chunking_mode}'. Expected one of {ALLOWED_CHUNKING_MODES}.")

        input_dim = fx_dim + (ref ** 3 if unified_pos else pos_dim)
        self.preprocess = MLP(input_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.distribute_blocks = distribute_blocks
        self._block_devices = None
        self._block_dispatch_configured = False
        self._primary_device = None

        self.blocks = nn.ModuleList([
            MSPTBlock(num_heads=num_heads, hidden_dim=n_hidden, dropout=dropout, act=act,
                           mlp_ratio=mlp_ratio, last_layer=(i == num_blocks - 1), out_dim=out_dim,
                           V=V, Q=Q, attn_pool=attn_pool, use_rope=use_rope, rope_base=rope_base,
                           use_flash_attn=use_flash_attn)
            for i in range(num_blocks)
        ])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_grid(self, pos: torch.Tensor) -> torch.Tensor:
        batchsize, _, _ = pos.shape
        device = pos.device
        gridx = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=torch.float)
        gridx = gridx.view(1, self.ref, 1, 1, 1).repeat(batchsize, 1, self.ref, self.ref, 1)
        gridy = torch.linspace(0, 2, self.ref, device=device, dtype=torch.float)
        gridy = gridy.view(1, 1, self.ref, 1, 1).repeat(batchsize, self.ref, 1, self.ref, 1)
        gridz = torch.linspace(-4, 4, self.ref, device=device, dtype=torch.float)
        gridz = gridz.view(1, 1, 1, self.ref, 1).repeat(batchsize, self.ref, self.ref, 1, 1)
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).reshape(batchsize, self.ref ** 3, 3)
        dist = torch.sqrt(((pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2).sum(dim=-1))
        return dist.reshape(batchsize, pos.shape[1], self.ref ** 3).contiguous()

    def _apply_chunking(self, features: torch.Tensor, positions: torch.Tensor | None):
        if self.chunking_mode != 'balltree':
            return features, None
        if positions is None:
            raise ValueError('Positions are required for balltree chunking')

        B, N, F = features.shape
        if positions.dim() == 3:
            if positions.size(0) != B:
                raise ValueError('Positions batch dimension must match features for balltree chunking')
            if B != 1:
                raise NotImplementedError('Balltree chunking currently supports batch size 1')
            positions_for_partition = positions[0]
        elif positions.dim() == 2:
            positions_for_partition = positions
        else:
            raise ValueError('Positions must be either 2D or 3D tensor for balltree chunking')

        device = features.device
        chunk_size = math.ceil(float(N) / float(self.V))
        N_padded = int(chunk_size * self.V)
        pad_len = N_padded - N

        if pad_len > 0:
            features = torch.cat([features, features.new_zeros(B, pad_len, F)], dim=1)
            positions_for_partition = torch.cat(
                [positions_for_partition, positions_for_partition.new_zeros(pad_len, positions_for_partition.size(-1))],
                dim=0
            )

        chunk_indices = chunk_points_balltree(positions_for_partition, self.V)
        perm = torch.cat(chunk_indices, dim=0)
        features = features[:, perm, :]

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=device)

        meta = {
            'inv_perm': inv_perm,
            'pad_len': pad_len,
            'original_length': N,
            'perm': perm,
        }
        return features, meta

    @staticmethod
    def _apply_perm(tensor: torch.Tensor | None, meta):
        if tensor is None or meta is None:
            return tensor
        perm = meta['perm']
        pad_len = meta['pad_len']
        B, N, F = tensor.shape
        if pad_len > 0:
            tensor = torch.cat([tensor, tensor.new_zeros(B, pad_len, F)], dim=1)
        return tensor[:, perm, :]

    @staticmethod
    def _restore_chunking(features: torch.Tensor, meta):
        if meta is None:
            return features
        inv_perm = meta['inv_perm']
        if inv_perm.device != features.device:
            inv_perm = inv_perm.to(features.device)
        features = features[:, inv_perm, :]
        pad_len = meta['pad_len']
        if pad_len > 0:
            features = features[:, :-pad_len, :]
        return features

    def _configure_block_dispatch(self, fx: torch.Tensor):
        if self._block_dispatch_configured:
            return
        self._block_dispatch_configured = True
        if not self.distribute_blocks:
            return
        if not torch.cuda.is_available():
            return

        num_gpus = torch.cuda.device_count()
        num_blocks = len(self.blocks)
        if num_gpus <= 1 or num_gpus != num_blocks:
            return

        self._primary_device = fx.device
        self._block_devices = [torch.device(f'cuda:{idx}') for idx in range(num_gpus)]
        for block, device in zip(self.blocks, self._block_devices):
            block.to(device)

    def forward(self, pos, fx=None, T=None, geo=None):
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if fx is not None and fx.dim() == 2:
            fx = fx.unsqueeze(0)

        pos_features = self.get_grid(pos) if self.unified_pos else pos
        pos_chunk, chunk_meta = self._apply_chunking(pos_features, pos if self.chunking_mode == 'balltree' else None)
        fx_chunk = self._apply_perm(fx, chunk_meta)

        if fx_chunk is not None:
            tokens = torch.cat((pos_chunk, fx_chunk), dim=-1)
        else:
            tokens = pos_chunk

        tokens = self.preprocess(tokens)
        tokens = tokens + self.placeholder[None, None, :]

        if self.Time_Input and T is not None:
            if T.dim() == 2 and T.size(1) == 1:
                T = T.squeeze(1)
            if T.dim() == 0:
                T = T.unsqueeze(0)
            time_emb = timestep_embedding(T, self.n_hidden).to(tokens.device)
            time_emb = time_emb.unsqueeze(1).repeat(1, tokens.shape[1], 1)
            time_emb = self.time_fc(time_emb)
            tokens = tokens + time_emb

        if self._primary_device is None:
            self._primary_device = tokens.device
        self._configure_block_dispatch(tokens)

        supernodes = None
        if self._block_devices:
            for block, device in zip(self.blocks, self._block_devices):
                if tokens.device != device:
                    tokens = tokens.to(device, non_blocking=True)
                if supernodes is not None and supernodes.device != device:
                    supernodes = supernodes.to(device, non_blocking=True)
                tokens, supernodes = block(tokens, supernodes)
            if tokens.device != self._primary_device:
                tokens = tokens.to(self._primary_device, non_blocking=True)
            if supernodes is not None and supernodes.device != self._primary_device:
                supernodes = supernodes.to(self._primary_device, non_blocking=True)
        else:
            for block in self.blocks:
                tokens, supernodes = block(tokens, supernodes)

        tokens = self._restore_chunking(tokens, chunk_meta)
        return tokens


class MSPTStructured2D(nn.Module):
    def __init__(self, pos_dim=2, fx_dim=1, out_dim=1, num_blocks=5, n_hidden=256, dropout=0.1,
                 num_heads=8, Time_Input=False, act='gelu', mlp_ratio=1, ref=8, unified_pos=False,
                 H=16, W=16, Q=1, V=32, attn_pool='mean', use_rope=False, rope_base=10000.0,
                 use_flash_attn=False):
        super().__init__()
        self.__name__ = 'MSPT_2D'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        if unified_pos:
            input_dim = fx_dim + self.ref * self.ref
        else:
            input_dim = fx_dim + pos_dim
        self.preprocess = MLP(input_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
        self.blocks = nn.ModuleList([
            MSPTBlock(num_heads=num_heads, hidden_dim=n_hidden, dropout=dropout, act=act,
                           mlp_ratio=mlp_ratio, last_layer=(i == num_blocks - 1), out_dim=out_dim,
                           V=V, Q=Q, attn_pool=attn_pool, use_rope=use_rope, rope_base=rope_base,
                           use_flash_attn=use_flash_attn)
            for i in range(num_blocks)
        ])
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_grid_reference(self, batchsize, device):
        size_x, size_y = self.H, self.W
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=torch.float).view(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=torch.float).view(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        grid = torch.cat((gridx, gridy), dim=-1)
        gridx_ref = torch.linspace(0, 1, self.ref, device=device, dtype=torch.float).view(1, self.ref, 1, 1).repeat(batchsize, 1, self.ref, 1)
        gridy_ref = torch.linspace(0, 1, self.ref, device=device, dtype=torch.float).view(1, 1, self.ref, 1).repeat(batchsize, self.ref, 1, 1)
        grid_ref = torch.cat((gridx_ref, gridy_ref), dim=-1)
        dist = torch.sqrt(((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2).sum(dim=-1))
        return dist.reshape(batchsize, size_x * size_y, self.ref * self.ref).contiguous()

    def forward(self, pos, fx=None, T=None, geo=None):
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if fx is not None and fx.dim() == 2:
            fx = fx.unsqueeze(0)
        if pos.shape[1] != self.H * self.W:
            raise ValueError('Structured MSPT expects N == H * W tokens')

        if self.unified_pos:
            pos_features = self.get_grid_reference(pos.shape[0], pos.device)
        else:
            pos_features = pos
        if fx is not None:
            tokens = torch.cat((pos_features, fx), dim=-1)
        else:
            tokens = pos_features
        tokens = self.preprocess(tokens)
        tokens = tokens + self.placeholder[None, None, :]

        if self.Time_Input and T is not None:
            if T.dim() == 2 and T.size(1) == 1:
                T = T.squeeze(1)
            if T.dim() == 0:
                T = T.unsqueeze(0)
            time_emb = timestep_embedding(T, self.n_hidden).to(tokens.device)
            time_emb = time_emb.unsqueeze(1).repeat(1, tokens.shape[1], 1)
            time_emb = self.time_fc(time_emb)
            tokens = tokens + time_emb

        supernodes = None
        for block in self.blocks:
            tokens, supernodes = block(tokens, supernodes)
        return tokens


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        geotype = getattr(args, 'geotype', 'unstructured')
        common_kwargs = dict(
            pos_dim=args.space_dim,
            fx_dim=args.fun_dim,
            out_dim=args.out_dim,
            num_blocks=args.n_layers,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            num_heads=args.n_heads,
            Time_Input=getattr(args, 'time_input', False),
            act=getattr(args, 'act', 'gelu'),
            mlp_ratio=getattr(args, 'mlp_ratio', 1),
            ref=getattr(args, 'ref', 8),
            unified_pos=bool(getattr(args, 'unified_pos', 0)),
            Q=getattr(args, 'mspt_Q', 1),
            V=getattr(args, 'mspt_V', 32),
            attn_pool=getattr(args, 'mspt_pool', 'mean'),
            use_rope=bool(getattr(args, 'mspt_use_rope', 0)),
            rope_base=getattr(args, 'mspt_rope_base', 10000.0),
            use_flash_attn=bool(getattr(args, 'mspt_use_flash_attn', 0)),
        )
        if geotype == 'unstructured':
            self.model = MSPTUnstructured(
                chunking_mode=getattr(args, 'mspt_chunking', 'linear'),
                distribute_blocks=bool(getattr(args, 'mspt_distribute_blocks', 0)),
                **common_kwargs,
            )
        elif geotype == 'structured_2D':
            shapelist = getattr(args, 'shapelist', None)
            if not shapelist or len(shapelist) != 2:
                raise ValueError('MSPT structured_2D requires shapelist with two entries (H, W)')
            self.model = MSPTStructured2D(H=shapelist[0], W=shapelist[1], **common_kwargs)
        else:
            raise ValueError(f'MSPT does not support geotype {geotype}')
        self.__name__ = 'MSPT'

    def forward(self, x, fx=None, T=None, geo=None):
        return self.model(x, fx, T, geo=geo)
