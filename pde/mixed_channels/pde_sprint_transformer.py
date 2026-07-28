from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from timm.models.layers import DropPath

from pde.mixed_channels.modules.attention import TokenInitializer, WindowAttention2DTime
from pde.mixed_channels.modules.embedding import (
    CarrierTokenAttention2DTimestep,
    LabelEmbedder,
    OverlapPatchEmbed,
    PosEmbMLPSwinv1D,
    SimplePatchEmbed,
    TimestepEmbedder,
)
from pde.mixed_channels.modules.general import (
    AdaLayerNormZero,
    Downsample,
    FinalLayer,
    GatedMlp,
    Mlp,
    Upsample,
)
from pde.mixed_channels.utils import (
    calc_entropy_map_gpu,
    window_partition,
    window_reverse,
)


class PDEStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        periodic=False,
        carrier_token_active: bool = True,
        mlp_ratio: float = 4.0,
        use_gated_mlp: bool = False,
        adjust_hidden: bool = True,
        drop_path: float = 0.0,
        use_output_proj: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.periodic = periodic
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.carrier_token_active = carrier_token_active

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
                drop_path=drop_path,
                use_output_proj=use_output_proj,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        if self.carrier_token_active:
            self.global_tokenizer = TokenInitializer(dim, window_size)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ):

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

            hidden_states, ct = block(
                hidden_states,
                ct,
                emb=cond,
                attn_mask=attn_mask,
            )

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
        window_size=7,
        last=False,
        do_propagation=False,
        carrier_token_active=True,
        use_output_proj=False,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
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
        self.norm2 = nn.LayerNorm(dim)
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
            self.hat_norm1 = nn.LayerNorm(dim)
            self.hat_norm2 = nn.LayerNorm(dim)
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
        num_windows_total = int(B // Bc)
        raw_modulation = self.adain_2.forward_raw(emb)
        raw_modulation = raw_modulation.repeat_interleave(num_windows_total, dim=0)
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = (
            raw_modulation.chunk(6, dim=1)
        )

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


class PDEImpl(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

    def __init__(
        self,
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
        output_activation: Optional[str] = None,
        use_upsample_activation: bool = False,
        class_dropout_prob: float = 0.1,
        num_classes=1000,
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

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."

        self.num_encoder_layers = len(depth) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.periodic = periodic
        self.use_carrier_tokens = carrier_token_active
        self.max_hidden_size = max_hidden_size

        # SPRINT properties
        self.path_drop_prob = path_drop_prob
        self.sprint_fusion_type = sprint_fusion_type
        self.sprint_drop_mode = sprint_drop_mode
        self.block_size_sprint = block_size_sprint
        self.tokens_per_block_sprint = tokens_per_block_sprint
        self.sprint_noise_scale = 0.1

        assert self.max_hidden_size >= hidden_size, (
            f"max_hidden_size {max_hidden_size} must be greater than or equal to hidden_size {hidden_size}."
        )

        self.sprint_mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.trunc_normal_(self.sprint_mask_token, std=0.02)

        if self.sprint_fusion_type == "linear":
            self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        elif self.sprint_fusion_type == "gated":
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.fusion_proj_f = nn.Linear(hidden_size, hidden_size, bias=True)
        elif self.sprint_fusion_type == "mlp":
            self.fusion_proj = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {self.sprint_fusion_type}")

        self.f_time_embedder = TimestepEmbedder(hidden_size)
        self.h_time_embedder = TimestepEmbedder(hidden_size)
        self.f_label_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        self.h_label_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        self.f_stage = PDEStage(
            dim=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            depth=2,
            drop_path=0.0,
            periodic=periodic,
            carrier_token_active=carrier_token_active,
            mlp_ratio=mlp_ratio,
            use_gated_mlp=use_gated_mlp,
            adjust_hidden=adjust_hidden,
            use_output_proj=use_output_proj,
        )

        self.h_stage = PDEStage(
            dim=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            depth=2,
            drop_path=0.0,
            periodic=periodic,
            carrier_token_active=carrier_token_active,
            mlp_ratio=mlp_ratio,
            use_gated_mlp=use_gated_mlp,
            adjust_hidden=adjust_hidden,
            use_output_proj=use_output_proj,
        )

        self.sprint_spread_conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=2,
            padding=1,
            groups=hidden_size,
            bias=False,
        )
        nn.init.constant_(self.sprint_spread_conv.weight, 1.0 / 4.0)

        self.sprint_fusion_activation = nn.SiLU()

        if self.sprint_drop_mode == "learned":
            self.importance_scorer = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_size // 4, 1, kernel_size=1),  # (B, 1, H, W)
            )

        dit_stage_args = {
            "drop_path": 0.0,
            "periodic": periodic,
            "carrier_token_active": carrier_token_active,
            "mlp_ratio": mlp_ratio,
            "use_gated_mlp": use_gated_mlp,
            "adjust_hidden": adjust_hidden,
            "use_output_proj": use_output_proj,
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
                PDEStage(
                    dim=hidden_size_layer,
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=depth[i],
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
        self.latent = PDEStage(
            dim=hidden_size_latent,
            num_heads=num_heads,
            window_size=window_size,
            depth=depth[self.num_encoder_layers],
            **dit_stage_args,
        )

        # hidden_size_layer0 = min(hidden_size * 2, max_hidden_size)
        hidden_size_layer0 = hidden_size
        if hidden_size_layer0 >= max_hidden_size:
            keep_dim = True
        else:
            keep_dim = False

        # double hidden size for last decoder layer 0
        self.__setattr__(
            "up1_0",
            Upsample(
                hidden_size_layer0 * 2,
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
            PDEStage(
                dim=hidden_size_layer0,
                num_heads=num_heads,
                window_size=window_size,
                depth=depth[self.num_encoder_layers + 1],
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
            self.__setattr__(
                f"decoder_level_{i}",
                PDEStage(
                    dim=hidden_size_layer,
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=depth[self.num_encoder_layers + i + 1],
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

        self.output_upsample = nn.Conv2d(
            hidden_size_layer0, hidden_size_out, kernel_size=1, bias=True
        )
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

    def _drop_tokens_sprint_blockwise(self, x, entropy_map=None):
        """
        Drop tokens for SPRINT using block-wise selection.
        Within each block, select tokens to keep via topk on importance scores.

        x: (N, C, H, W) tensor of spatial inputs
        entropy_map: (N, H, W) optional tensor of importance scores (higher = more likely to keep)

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

            # Use importance scores directly; add noise during training for stochasticity
            scores = entropy_map
            if self.training:
                scores = scores + self.sprint_noise_scale * torch.rand_like(scores)
            _, ids_keep = torch.topk(
                scores, k=self.tokens_per_block_sprint, dim=-1
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
            self.sprint_mask_token.to(dtype=x_sparse.dtype)
            .unsqueeze(-1)
            .expand(B, C, H_orig * W_orig)
            .contiguous()
        )
        x_sparse_flat = x_sparse.reshape(B, C, H_out * W_out)
        flat_indices = ids_keep_global.reshape(B, H_out * W_out)
        x_full_flat.scatter_(
            2, flat_indices.unsqueeze(1).expand(-1, C, -1), x_sparse_flat
        )
        return x_full_flat.reshape(B, C, H_orig, W_orig).contiguous()

    def _fuse_sprint(self, f_dense, g_full):
        """
        Fuse dense and sparse features for SPRINT.
        f_dense: (N, H, W, C) tensor of dense features
        g_full: (N, H, W, C) tensor of sparse features
        """
        if self.sprint_fusion_type == "gated":
            concat = torch.cat([f_dense, g_full], dim=-1)
            gate = self.fusion_gate(concat)
            update = self.sprint_fusion_activation(self.fusion_proj_f(f_dense))
            h = g_full + (gate * update)
        elif self.sprint_fusion_type == "linear" or self.sprint_fusion_type == "mlp":
            h = torch.cat([f_dense, g_full], dim=-1)
            h = self.fusion_proj(h)
        else:
            raise ValueError(f"Unknown fusion_type: {self.sprint_fusion_type}")
        return h

    def forward(self, x, t, y):
        """
        Forward pass of PDE transformer.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N, ) tensor of diffusion timesteps
        y: (N, ) tensor of class labels [int]
        """

        x = self.x_embedder(x)  # (N, C, H, W)

        if t is None:
            t = torch.Tensor([0]).to(x.device)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            t = t.repeat(x.shape[0])
            t = t.to(x.device)

        # timestep scaling (from 0 - 1 to 0 - 1000)
        t = t * 1000.0

        if y is None:
            y = (
                torch.ones(x.shape[0], dtype=torch.long, device=x.device)
                * self.num_classes
            )

        # f stage
        cond = self.f_time_embedder(t) + self.f_label_embedder(y, self.training)
        x_f = self.f_stage(x, cond)
        _, _, H, W = x_f.shape  # save original shape for SPRINT padding

        importance_map = None
        if self.sprint_drop_mode == "entropy":
            entropy_maps = []
            for b in range(x_f.shape[0]):
                emap = calc_entropy_map_gpu(x_f[b], kernel_size=3, num_bins=32)
                entropy_maps.append(emap)
            importance_map = torch.stack(entropy_maps)
        elif self.sprint_drop_mode == "var":
            importance_map = x_f.var(dim=1)
        elif self.sprint_drop_mode == "l2":
            importance_map = torch.linalg.vector_norm(x_f, ord=2, dim=1)
        elif self.sprint_drop_mode == "learned":
            importance_map = self.importance_scorer(x_f).squeeze(1)
            # multiply with importance map to include it in gradient flow
            # tanh bounds the multiplier to [0, 2], preventing explosion
            x_f = x_f * (
                1.0 + torch.tanh(importance_map.unsqueeze(1))
            )  # 1+tanh(importance_map) to retain base signal

        x, ids_keep = self._drop_tokens_sprint_blockwise(
            x_f, entropy_map=importance_map
        )

        emb_list = []
        for i in range(self.num_encoder_layers + 1):
            t_emb = self.__getattr__(f"t_embedder_{i}")(t)
            y_emb = self.__getattr__(f"y_embedder_{i}")(y, self.training)
            c = t_emb + y_emb
            emb_list.append(c)

        residuals_list = []
        for i, c in enumerate(emb_list[:-1]):
            # encoder
            out_enc_level = self.__getattr__(f"encoder_level_{i}")(x, c)
            residuals_list.append(out_enc_level)
            x = self.__getattr__(f"down{i}_{i + 1}")(out_enc_level)

        c = emb_list[-1]
        x = self.latent(x, c)

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
        x = self.__getattr__(f"decoder_level_0")(x, emb_list[0])

        # pad tokens back for SPRINT
        x_pad = self._pad_tokens_sprint(x, ids_keep, H, W)
        x_pad = self.sprint_spread_conv(x_pad)[:, :, :H, :W]

        x = self._fuse_sprint(x_pad.permute(0, 2, 3, 1), x_f.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2).contiguous()

        # h stage
        cond = self.h_time_embedder(t) + self.h_label_embedder(y, self.training)
        x = self.h_stage(x, cond)

        # output
        x = self.output_upsample(x)
        x = self.output_activation(x)
        x = self.output(x)
        x = self.final_layer(x, emb_list[1])  # (N, T, patch_size ** 2 * out_channels)

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


@dataclass
class PDEOutput(BaseOutput):
    """
    The output of [`PDEOutput`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor


class PDETransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        model_type: Optional[str] = None,
        periodic: bool = True,
        carrier_token_active: bool = False,
        window_size: int = 8,
        patch_size: Optional[int] = 4,
        output_activation: str = "gelu",
        use_upsample_activation: bool = True,
        path_drop_prob: float = 0.05,
        sprint_fusion_type: str = "linear",
        sprint_drop_mode: str = "random",
        block_size_sprint: int = 2,
        tokens_per_block_sprint: int = 1,
        use_gated_mlp: bool = False,
        adjust_hidden: bool = True,
        use_output_proj: bool = False,
        **kwargs,
    ):
        super(PDETransformer, self).__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.periodic = periodic
        self.carrier_token_active = carrier_token_active
        self.window_size = window_size
        self.output_activation = output_activation
        self.use_gated_mlp = use_gated_mlp
        self.adjust_hidden = adjust_hidden
        self.use_upsample_activation = use_upsample_activation
        self.use_output_proj = use_output_proj
        self.path_drop_prob = path_drop_prob
        self.sprint_fusion_type = sprint_fusion_type
        self.sprint_drop_mode = sprint_drop_mode
        self.block_size_sprint = block_size_sprint
        self.tokens_per_block_sprint = tokens_per_block_sprint

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
        assert output_activation in ["gelu", "silu", None], (
            f"Invalid output_activation: {output_activation}."
        )

        args = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "patch_size": patch_size,
            "periodic": periodic,
            "carrier_token_active": carrier_token_active,
            "window_size": window_size,
            "output_activation": output_activation,
            "use_upsample_activation": use_upsample_activation,
            "use_gated_mlp": use_gated_mlp,
            "adjust_hidden": adjust_hidden,
            "use_output_proj": use_output_proj,
            "path_drop_prob": path_drop_prob,
            "sprint_fusion_type": sprint_fusion_type,
            "sprint_drop_mode": sprint_drop_mode,
            "block_size_sprint": block_size_sprint,
            "tokens_per_block_sprint": tokens_per_block_sprint,
        }

        args.update(kwargs)

        if model_type is None:
            self.model = PDEImpl(**args)
        elif model_type not in PDE_models:
            raise ValueError(
                f"model_type {model_type} not found. Available model_types are: {list(PDE_models.keys())}"
            )
        else:
            self.model: PDEImpl = PDE_models[model_type](**args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):

        output = self.model.forward(hidden_states, timestep, class_labels)

        if not return_dict:
            return (output,)

        return PDEOutput(sample=output)


#################################################################################
#                            PDE Transformer Configs                            #
#################################################################################


def PDE_XS(**kwargs):
    return PDEImpl(
        down_factor=2,
        hidden_size=96,
        num_heads=2,
        depth=[1, 2, 4, 2, 1],
        mlp_ratio=4,
        **kwargs,
    )


def PDE_S(**kwargs):
    return PDEImpl(
        down_factor=2,
        hidden_size=96,
        num_heads=4,
        depth=[2, 5, 8, 5, 2],
        mlp_ratio=4,
        **kwargs,
    )


def PDE_B(**kwargs):
    return PDEImpl(
        down_factor=2,
        hidden_size=192,
        num_heads=8,
        depth=[2, 5, 8, 5, 2],
        mlp_ratio=4,
        **kwargs,
    )


def PDE_L(**kwargs):
    return PDEImpl(
        down_factor=2,
        hidden_size=384,
        num_heads=16,
        depth=[2, 5, 8, 5, 2],
        mlp_ratio=4,
        **kwargs,
    )


PDE_models = {
    "PDE-XS": PDE_XS,
    "PDE-S": PDE_S,
    "PDE-B": PDE_B,
    "PDE-L": PDE_L,
}
