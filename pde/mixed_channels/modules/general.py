from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# @torch.compile
class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, bias=True):
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

    def forward_raw(
        self,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return the full 6*dim modulation vector before chunking."""
        return self.linear(self.silu(emb))


# @torch.compile
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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# @torch.compile
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


# @torch.compile
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


# @torch.compile
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


# @torch.compile
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
