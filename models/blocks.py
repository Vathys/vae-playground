from typing import Dict, Tuple

import torch.nn as nn
from torch import Tensor


class Identity(nn.Module):
    def forward(self, x):
        return x


class AdaIN(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()

    def forward(self, x: Tensor, cond: Tensor):
        def _get_mean_std(a):
            a = a.flatten(start_dim=1)
            return (
                a.mean(dim=1, keepdim=True)[..., None, None],
                a.std(dim=1, keepdim=True)[..., None, None],
            )

        b, s = _get_mean_std(cond)
        x_mean, x_std = _get_mean_std(x)

        x = (x - x_mean) / x_std

        x = x * s + b

        return x


class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, num_channels)
        self.to_shift = nn.Linear(cond_dim, num_channels)

    def forward(self, x: Tensor, cond: Tensor):
        s = self.to_scale(cond)[..., None, None]
        b = self.to_shift(cond)[..., None, None]

        return s * x + b


class NormalizedFiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, num_channels)
        self.to_shift = nn.Linear(cond_dim, num_channels)

    def forward(self, x: Tensor, cond: Tensor):
        def _get_mean_std(a):
            a = a.flatten(start_dim=1)
            return (
                a.mean(dim=1, keepdim=True)[..., None, None],
                a.std(dim=1, keepdim=True)[..., None, None],
            )

        x_mean, x_std = _get_mean_std(x)

        s = self.to_scale(cond)[..., None, None]
        b = self.to_shift(cond)[..., None, None]

        x = s * x + b

        n_x_mean, n_x_std = _get_mean_std(x)

        x = (x - n_x_mean) / (n_x_std + 1e-6)
        x = x * x_std + x_mean

        return x


class Attention(nn.Module):
    def __init__(self, cond_dim, num_channels, attn_dim=256, attn_heads=4):
        super().__init__()

        self.q_proj = nn.Conv2d(num_channels, attn_dim, kernel_size=1)

        self.k_proj = nn.Linear(cond_dim, attn_dim)
        self.v_proj = nn.Linear(cond_dim, attn_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=attn_heads, batch_first=True
        )

        self.out_proj = nn.Conv2d(attn_dim, num_channels, kernel_size=1)

    def forward(self, x: Tensor, cond: Tensor):
        B, C, H, W = x.shape
        S = H * W

        q = self.q_proj(x)
        q = q.view(B, -1, S).permute(0, 2, 1)

        k = self.k_proj(cond).unsqueeze(1)
        v = self.v_proj(cond).unsqueeze(1)

        attn_out, _ = self.mha(query=q, key=k, value=v)
        attn_out = attn_out.permute(0, 2, 1).view(B, -1, H, W)

        attn_out = self.out_proj(attn_out)

        return x + attn_out


class WeightNormConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        weight_norm=True,
        scale=False,
        transpose=False,
    ):
        """Intializes a Conv2d augmented with weight normalization.

        (See torch.nn.utils.weight_norm for detail.)

        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            output_padding: for inferring output shape
              (only for transposed convolution).
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
            transpose: True if transposed convolution, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()

        def norm_func(module: nn.Module) -> nn.Module:
            if weight_norm and scale:
                return nn.utils.parametrizations.weight_norm(module)
            else:
                return module

        if transpose:
            self.conv = norm_func(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias,
                )
            )
        else:
            self.conv = norm_func(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)


class Block(nn.Module):
    ACTIVATIONS = {
        "none": lambda dim: Identity(),
        "relu": lambda dim: nn.ReLU(),
        "prelu": lambda dim: nn.PReLU(dim),
        "prelu1": lambda dim: nn.PReLU(1),
        "leaky_relu": lambda dim: nn.LeakyReLU(),
        "selu": lambda dim: nn.SELU(),
        "celu": lambda dim: nn.CELU(),
        "gelu": lambda dim: nn.GELU(),
        "silu": lambda dim: nn.SiLU(),
        "sigmoid": lambda dim: nn.Sigmoid(),
        "tanh": lambda dim: nn.Tanh(),
    }

    NORMS = {
        "none": lambda dim: Identity(),
        "batch": lambda dim: nn.BatchNorm2d(dim),
        "layer": lambda dim: nn.GroupNorm(1, dim),
        "instance": lambda dim: nn.GroupNorm(dim, dim),
    }

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        weight_norm=True,
        scale=False,
        transpose=False,
        norm="batch",
        activation="leaky_relu",
    ):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim

        self.conv = WeightNormConv2d(
            in_dim,
            out_dim,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias,
            weight_norm,
            scale,
            transpose,
        )

        if norm in self.NORMS.keys():
            self.norm_block = self.NORMS[norm](out_dim)
        else:
            raise ValueError(f"norm {norm} not supported")
        if activation in self.ACTIVATIONS.keys():
            self.activation_block = self.ACTIVATIONS[activation](out_dim)
        else:
            raise ValueError(f"activation {activation} not supported")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_block(x)
        x = self.activation_block(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        bottleneck,
        weight_norm,
        transpose=False,
        norm="batch",
        activation="leaky_relu",
    ):
        """Initializes a Standard Block.

        Args:
            dim: number of input and output features.
            bottleneck: True if use bottleneck, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            transpose: True if transposed convolution, False otherwise.
        """
        super().__init__()

        if bottleneck:
            self.block = nn.Sequential(
                Block(
                    dim,
                    dim,
                    (1, 1),
                    stride=1,
                    padding=0,
                    bias=False,
                    weight_norm=weight_norm,
                    scale=False,
                    transpose=transpose,
                    norm=norm,
                    activation=activation,
                ),
                Block(
                    dim,
                    dim,
                    (3, 3),
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_norm=weight_norm,
                    scale=False,
                    transpose=transpose,
                    norm=norm,
                    activation=activation,
                ),
                Block(
                    dim,
                    dim,
                    (1, 1),
                    stride=1,
                    padding=0,
                    bias=True,
                    weight_norm=weight_norm,
                    scale=True,
                    transpose=transpose,
                ),
            )
        else:
            self.block = nn.Sequential(
                Block(
                    dim,
                    dim,
                    (3, 3),
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_norm=weight_norm,
                    scale=False,
                    transpose=transpose,
                    norm=norm,
                    activation=activation,
                ),
                Block(
                    dim,
                    dim,
                    (3, 3),
                    stride=1,
                    padding=1,
                    bias=True,
                    weight_norm=weight_norm,
                    scale=True,
                    transpose=transpose,
                    norm=norm,
                    activation=activation,
                ),
            )

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.block(x)


class ResidualConvBlock(ConvBlock):
    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return x + self.block(x)


class EncoderScale(nn.Module):
    def __init__(
        self,
        blocks: Dict[str, nn.Module],
        down: Block,
        cond_dim: int = None,
        mix_type: str = "film",
    ):
        super().__init__()
        self.blocks = nn.ModuleDict(blocks)
        self.down = down

        if cond_dim is not None:
            if mix_type == "film":
                self.mix_block = FiLM(cond_dim, self.down.out_channels)
            elif mix_type == "adain":
                self.mix_block = AdaIN(cond_dim, self.down.out_channels)
            elif mix_type == "norm_film":
                self.mix_block = NormalizedFiLM(cond_dim, self.down.out_channels)
            elif mix_type == "attention":
                self.mix_block = Attention(cond_dim, self.down.out_channels)
            else:
                raise ValueError(f"mix_type {mix_type} not available...")
        else:
            self.mix_block = None

    def forward(self, x, cond=None):
        for _, b in self.blocks.items():
            x = b(x)

        x = self.down(x)
        if self.mix_block is not None and cond is not None:
            x = self.mix_block(x, cond)

        return x


class DecoderScale(nn.Module):
    def __init__(
        self,
        blocks: Dict[str, nn.Module],
        up: Block,
        cond_dim: int = None,
        mix_type: str = "film",
    ):
        super().__init__()
        self.blocks = nn.ModuleDict(blocks)
        self.up = up

        if cond_dim is not None:
            if mix_type == "film":
                self.mix_block = FiLM(cond_dim, self.up.out_channels)
            elif mix_type == "adain":
                self.mix_block = AdaIN(cond_dim, self.up.out_channels)
            elif mix_type == "norm_film":
                self.mix_block = NormalizedFiLM(cond_dim, self.up.out_channels)
            elif mix_type == "attention":
                self.mix_block = Attention(cond_dim, self.up.out_channels)
            else:
                raise ValueError(f"mix_type {mix_type} not available...")
        else:
            self.mix_block = None

    def forward(self, x, cond=None):
        for _, b in self.blocks.items():
            x = b(x)

        x = self.up(x)
        if self.mix_block is not None and cond is not None:
            x = self.mix_block(x, cond)

        return x


def build_encoder(cfg: dict) -> Tuple[nn.Module, int]:
    """Builds an encoder that successively scales down via convolutions.

    :param cfg: Encoder config

    Config includes:
        - in_channels: number of input channels
        - base_dim: starting dimension for scaling
        - scales: number of times to scale the image
        - num_blocks: number of convolution blocks per scale
        - residual: boolean indicating whether to make blocks residual.
        - channel_multiplier: multiplier to increase channels per scale.
        - bottleneck: whether scale block should include a bottleneck
        - weight_norm: whether to normalize weights
        - norm: normalization block to use
        - activation: activation to use
        - cond_dim: dimension of conditioning vector (default: None)
        - mix_type: type of mixing to apply to conditioning (default: film)

    :type cfg: dict
    :return: Tuple with encoder, output dimension and output size
    :rtype: Tuple[Module, int, int]
    """
    BlockType = ResidualConvBlock if cfg["residual"] else ConvBlock
    cond_dim = cfg.get("cond_dim", None)
    mix_type = cfg.get("mix_type", "film")

    in_block = Block(
        cfg["in_channels"],
        cfg["base_dim"],
        (3, 3),
        stride=1,
        padding=1,
        bias=True,
        weight_norm=cfg["weight_norm"],
        scale=True,
        norm=cfg["norm"],
        activation=cfg["activation"],
    )

    scales = nn.ModuleDict()
    dim = cfg["base_dim"]

    for i in range(cfg["scales"]):
        blocks = {}
        for j in range(cfg["num_blocks"]):
            blocks[f"scale_{i}_block_{j}"] = BlockType(
                dim,
                bottleneck=cfg["bottleneck"],
                weight_norm=cfg["weight_norm"],
                norm=cfg["norm"],
                activation=cfg["activation"],
            )

        out_dim = int(dim * cfg["channel_multiplier"])
        down = Block(
            dim,
            out_dim,
            (3, 3),
            stride=2,
            padding=1,
            bias=True,
            weight_norm=cfg["weight_norm"],
            scale=True,
            norm=cfg["norm"],
            activation=cfg["activation"],
        )
        scales[f"scale_{i}"] = EncoderScale(blocks, down, cond_dim, mix_type)
        dim = out_dim

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_block = in_block
            self.scales = scales

        def forward(self, x, cond=None):
            x = self.in_block(x)
            for _, s in self.scales.items():
                x = s(x, cond)
            return x

    return Encoder(), dim


def build_decoder(cfg: dict) -> Tuple[nn.Module, int]:
    """Builds a decoder that successively scales up via convolutions.

    :param cfg: Decoder config:

    Config includes:
        - in_channels: number of input channels
        - out_channels: number of output channels;
          if none, then no final block is applied
        - scales: number of times to scale the image
        - num_blocks: number of convolution blocks per scale
        - residual: boolean indicating whether to make blocks residual.
        - channel_multiplier: multiplier to increase channels per scale.
        - bottleneck: whether scale block should include a bottleneck
        - weight_norm: whether to normalize weights
        - norm: normalization block to use
        - activation: activation to use
        - final_activation: activation to use in the final convolution.
        - cond_dim: dimension of conditioning vector (default: None)
        - mix_type: type of mixing to apply to conditioning (default: film)

    :type cfg: dict
    :return: Tuple with decoder, output dimension and output size
    :rtype: Tuple[Module, int, int]
    """
    BlockType = ResidualConvBlock if cfg["residual"] else ConvBlock
    cond_dim = cfg.get("cond_dim", None)
    mix_type = cfg.get("mix_type", "film")

    dim = cfg["in_channels"]

    scales = nn.ModuleDict()

    for i in reversed(range(cfg["scales"])):
        blocks = {}
        for j in range(cfg["num_blocks"]):
            blocks[f"scale_{i}_block_{j}"] = BlockType(
                dim,
                bottleneck=cfg["bottleneck"],
                weight_norm=cfg["weight_norm"],
                transpose=True,
                norm=cfg["norm"],
                activation=cfg["activation"],
            )

        out_dim = int(dim * cfg["channel_multiplier"])
        up = Block(
            dim,
            out_dim,
            (3, 3),
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
            weight_norm=cfg["weight_norm"],
            scale=True,
            transpose=True,
            norm=cfg["norm"],
            activation=cfg["activation"],
        )
        scales[f"scale_{i}"] = DecoderScale(blocks, up, cond_dim, mix_type)

        dim = out_dim

    if cfg["out_channels"] is not None:
        out_block = Block(
            dim,
            cfg["out_channels"],
            (3, 3),
            stride=1,
            padding=1,
            bias=True,
            weight_norm=cfg["weight_norm"],
            scale=True,
            norm="none",
            activation=cfg["final_activation"],
        )
        dim = cfg["out_channels"]
    else:
        out_block = None

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scales = scales
            self.out_block = out_block

        def forward(self, x, cond=None):
            for _, s in self.scales.items():
                x = s(x, cond)
            if self.out_block:
                x = self.out_block(x)

            return x

    return Decoder(), dim
