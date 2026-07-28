import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from layers.Basic import MLP
from layers.Embedding import timestep_embedding, unified_pos_embedding
from einops import rearrange, repeat
from layers.flash_bias_triton import flash_bias_func


class Spatial_Bias_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.scale = dim_head ** -0.5
        # Separate projection layers for query, key, and value
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.pos_weight = nn.Linear(self.dim_head, 1, bias=False)

    def forward(self, x, q_bias, k_bias):
        # q_bias: [batch_size, seq_len, 1, 5]
        # k_bias: [batch_size, seq_len, 1, 5]
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)
        pos_weight = self.pos_weight(rearrange(x, 'b n (h d) -> b n h d', h=self.heads))  # B H N 1
        ## Fast computation with FlashBias https://arxiv.org/abs/2505.12044
        attn_output = flash_bias_func(
            q.half(), k.half(), v.half(),
            (pos_weight * q_bias.repeat(1, 1, self.heads, 1)).half(),
            k_bias.repeat(1, 1, self.heads, 1).half(),
            None,
            False,
            self.scale
        )
        out = rearrange(attn_output.float(), 'b n h d -> b n (h d)')
        return self.to_out(out)


class Transformer_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        # Attention accelerated by FlashBias
        self.Attn = Spatial_Bias_Attention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                           dropout=dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, q_bias, k_bias):
        fx = self.Attn(self.ln_1(fx), q_bias, k_bias) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    ## speed up with FlashBias: Fast Computation of Attention with Bias
    ## https://arxiv.org/abs/2505.12044
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Transformer_spatial_bias'
        self.args = args
        ## embedding
        if args.unified_pos and args.geotype != 'unstructured':  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref)
            self.preprocess = MLP(args.fun_dim + args.ref ** len(args.shapelist), args.n_hidden * 2,
                                  args.n_hidden, n_layers=0, res=False, act=args.act)
        else:
            self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                                  n_layers=0, res=False, act=args.act)
        if args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))

        ## models
        self.blocks = nn.ModuleList([Transformer_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
                                                       dropout=args.dropout,
                                                       act=args.act,
                                                       mlp_ratio=args.mlp_ratio,
                                                       out_dim=args.out_dim,
                                                       last_layer=(_ == args.n_layers - 1))
                                     for _ in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cartesian_to_spherical_2d(self, coords):
        x, y = coords[..., 0], coords[..., 1]
        r = torch.sqrt(x ** 2 + y ** 2)  # add epsilon to avoid zero division
        phi = torch.atan2(y, x)  # azimuthal angle
        return torch.stack((r, phi), dim=-1)

    def cartesian_to_spherical_3d(self, coords):
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-8)  # add epsilon to avoid zero division
        theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))  # polar angle
        phi = torch.atan2(y, x)  # azimuthal angle
        return torch.stack((r, theta, phi), dim=-1)

    def get_matrix(self, original_pos):
        if original_pos.shape[-1] == 2:
            spherical_pos = self.cartesian_to_spherical_2d(original_pos)
            q_bias = torch.cat(
                [2 * spherical_pos[:, :, None, 0:1] * torch.cos(spherical_pos[:, :, None, 1:2]),
                 2 * spherical_pos[:, :, None, 0:1] * torch.sin(spherical_pos[:, :, None, 1:2]),
                 -spherical_pos[:, :, None, 0:1] ** 2,
                 torch.ones_like(spherical_pos[:, :, None, 0:1]).cuda()],
                dim=-1)
            k_bias = torch.cat(
                [spherical_pos[:, :, None, 0:1] * torch.cos(spherical_pos[:, :, None, 1:2]),
                 spherical_pos[:, :, None, 0:1] * torch.sin(spherical_pos[:, :, None, 1:2]),
                 torch.ones_like(spherical_pos[:, :, None, 0:1]).cuda(),
                 -spherical_pos[:, :, None, 0:1] ** 2],
                dim=-1)
        elif original_pos.shape[-1] == 3:
            spherical_pos = self.cartesian_to_spherical_3d(original_pos)
            q_bias = torch.cat(
                [2 * spherical_pos[:, :, None, 0:1] * torch.sin(spherical_pos[:, :, None, 1:2]) * torch.cos(
                    spherical_pos[:, :, None, 2:3]),
                 2 * spherical_pos[:, :, None, 0:1] * torch.sin(spherical_pos[:, :, None, 1:2]) * torch.sin(
                     spherical_pos[:, :, None, 2:3]),
                 2 * spherical_pos[:, :, None, 0:1] * torch.cos(spherical_pos[:, :, None, 1:2]),
                 - spherical_pos[:, :, None, 0:1] ** 2,
                 torch.ones_like(spherical_pos[:, :, None, 0:1]).cuda()],
                dim=-1)
            k_bias = torch.cat(
                [spherical_pos[:, :, None, 0:1] * torch.sin(spherical_pos[:, :, None, 1:2]) * torch.cos(
                    spherical_pos[:, :, None, 2:3]),
                 spherical_pos[:, :, None, 0:1] * torch.sin(spherical_pos[:, :, None, 1:2]) * torch.sin(
                     spherical_pos[:, :, None, 2:3]),
                 spherical_pos[:, :, None, 0:1] * torch.cos(spherical_pos[:, :, None, 1:2]),
                 torch.ones_like(spherical_pos[:, :, None, 0:1]).cuda(),
                 - spherical_pos[:, :, None, 0:1] ** 2],
                dim=-1)
        else:
            print("please check input shape")
        return q_bias, k_bias

    def forward(self, x, fx, T=None, geo=None):
        q_bias, k_bias = self.get_matrix(x)
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx, q_bias, k_bias)
        return fx
