import torch.nn as nn


class ConditionedEncoder2DBlock(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int, num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.gn_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = nn.GELU()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        self.mlp_scale_bias = nn.Linear(embed_dim, 2 * in_channels)
        self.gn_2 = nn.GroupNorm(num_groups, in_channels)
        self.activation_2 = nn.GELU()
        self.conv_2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x, embedding):

        scale_and_shift = self.mlp_scale_bias(embedding)
        scale, shift = scale_and_shift.chunk(2, dim=-1)

        x_res = x

        x = self.gn_1(x)
        x = self.activation_1(x)
        x = self.conv_1(x)
        x = self.gn_2(x)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.activation_2(x)
        x = self.conv_2(x)

        x = x + x_res

        return x


class ConditionedEncoder2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        feature_embedding_dim: int,
        num_downsampling_layers: int,
        embedding_dim: int,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_embedding_dim = feature_embedding_dim
        self.num_downsampling_layers = num_downsampling_layers
        self.embedding_dim = embedding_dim

        self.feature_embed = nn.Conv2d(in_channels, feature_embedding_dim, 3, 1, 1)
        self.downsampling_layers = nn.ModuleList()
        for i in range(num_downsampling_layers):
            self.downsampling_layers.append(
                nn.Conv2d(
                    feature_embedding_dim * 2**i,
                    feature_embedding_dim * 2 ** (i + 1),
                    3,
                    2,
                    1,
                )
            )
        self.blocks = nn.ModuleList()
        for i in range(num_downsampling_layers - 1):
            self.blocks.append(
                ConditionedEncoder2DBlock(
                    feature_embedding_dim * 2 ** (i + 1),
                    embedding_dim,
                    num_groups=num_groups,
                )
            )

    def forward(self, x, embedding):

        x = self.feature_embed(x)

        res_list = [x]

        x = self.downsampling_layers[0](x)

        for i in range(self.num_downsampling_layers - 1):
            x = self.blocks[i](x, embedding)
            res_list.append(x)
            x = self.downsampling_layers[i + 1](x)

        res_list.append(x)

        return res_list


class DecoderUpsamplingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_conv = nn.Conv2d(in_channels, out_channels * 2, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.linear_conv(x)
        x = self.shuffle(x)
        return x


class ConditionedDecoder2D(nn.Module):

    def __init__(
        self,
        out_channels: int,
        feature_embedding_dim: int,
        num_upsampling_layers: int,
        embedding_dim: int,
        features_first_layer: int = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.feature_embedding_dim = feature_embedding_dim
        self.num_upsampling_layers = num_upsampling_layers
        self.embedding_dim = embedding_dim

        self.decompress = nn.Conv2d(feature_embedding_dim, out_channels, 3, 1, 1)

        self.blocks = nn.ModuleList()
        for i in range(num_upsampling_layers - 1):
            self.blocks.append(
                ConditionedEncoder2DBlock(
                    feature_embedding_dim * 2 ** (num_upsampling_layers - i - 1),
                    embedding_dim,
                    num_groups=num_groups,
                )
            )

        if features_first_layer is None:
            features_first_layer = feature_embedding_dim

        self.upsampling_layers = nn.ModuleList()

        local_feature_dim = feature_embedding_dim * 2**num_upsampling_layers
        self.upsampling_layers.append(
            DecoderUpsamplingBlock(features_first_layer, local_feature_dim)
        )
        for i in range(num_upsampling_layers - 1):
            local_feature_dim = feature_embedding_dim * 2 ** (
                num_upsampling_layers - i - 1
            )
            self.upsampling_layers.append(
                DecoderUpsamplingBlock(local_feature_dim, local_feature_dim)
            )

    def forward(self, x, embedding, encoder_outputs):

        x = self.upsampling_layers[0](x)
        x += encoder_outputs[::-1][1]

        for i in range(self.num_upsampling_layers - 1):
            x = self.blocks[i](x, embedding)
            x = self.upsampling_layers[i + 1](x)
            x += encoder_outputs[::-1][i + 2]

        x = self.decompress(x)

        return x
