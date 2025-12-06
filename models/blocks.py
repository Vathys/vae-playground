from typing import Tuple
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


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
            output_padding: for inferring output shape (only for transposed convolution).
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
            transpose: True if transposed convolution, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()
        if weight_norm and scale:
            norm_func = nn.utils.parametrizations.weight_norm
        else:
            norm_func = lambda x: x
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


def build_encoder(cfg: dict) -> Tuple[nn.Module, int, int]:
    """Builds an encoder that successively scales down via convolutions.

    :param cfg: Encoder config

    Config includes:
        - in_channels: number of input channels
        - in_size: size of (square) input image
        - base_dim: starting dimension for scaling
        - scales: number of times to scale the image
        - num_blocks: number of convolution blocks per scale
        - residual: boolean indicating whether to make blocks residual.
        - channel_multiplier: multiplier to increase channels per scale.
        - min_spatial: minimum size when scaling; images won't be scaled above this size.
        - bottleneck: whether scale block should include a bottleneck
        - weight_norm: whether to normalize weights
        - norm: normalization block to use
        - activation: activation to use

    :type cfg: dict
    :return: Tuple with encoder, out dimension and output size
    :rtype: Tuple[Module, int, int]
    """
    BlockType = ResidualConvBlock if cfg["residual"] else ConvBlock

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

    core = nn.Sequential()
    dim = cfg["base_dim"]
    spatial = cfg["in_size"]

    for i in range(cfg["scales"]):
        for j in range(cfg["num_blocks"]):
            core.add_module(
                f"scale_{i}_block_{j}",
                BlockType(
                    dim,
                    bottleneck=cfg["bottleneck"],
                    weight_norm=cfg["weight_norm"],
                    norm=cfg["norm"],
                    activation=cfg["activation"],
                ),
            )

        if spatial > cfg["min_spatial"]:
            out_dim = int(dim * cfg["channel_multiplier"])
            core.add_module(
                f"scale_{i}_down",
                Block(
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
                ),
            )
            spatial //= 2
            dim = out_dim
        else:
            print(f"Skipping scale {i}. Cannot scale beyond {cfg['min_spatial']}.")

    encoder = nn.Sequential(in_block, core)
    return encoder, dim, spatial


def build_decoder(cfg: dict) -> nn.Module:
    """Builds a decoder that successively scales up via convolutions.

    :param cfg: Decoder config:

    Config includes:
        - in_channels: number of input channels
        - in_size: size of (square) input image
        - out_channels: number of output channels
        - scales: number of times to scale the image
        - num_blocks: number of convolution blocks per scale
        - residual: boolean indicating whether to make blocks residual.
        - channel_multiplier: multiplier to increase channels per scale.
        - bottleneck: whether scale block should include a bottleneck
        - weight_norm: whether to normalize weights
        - norm: normalization block to use
        - activation: activation to use
        - final_activation: activation to use in the final convolution.

    :type cfg: dict
    :param start_dim: Description
    :type start_dim: int
    :param start_hw: Description
    :type start_hw: int
    :return: Description
    :rtype: Module
    """
    BlockType = ResidualConvBlock if cfg["residual"] else ConvBlock

    dim = cfg["in_channels"]
    spatial = cfg["in_size"]

    core = nn.Sequential()

    for i in reversed(range(cfg["scales"])):
        for j in range(cfg["num_blocks"]):
            core.add_module(
                f"scale_{i}_block_{j}",
                BlockType(
                    dim,
                    bottleneck=cfg["bottleneck"],
                    weight_norm=cfg["weight_norm"],
                    transpose=True,
                    norm=cfg["norm"],
                    activation=cfg["activation"],
                ),
            )

        out_dim = int(dim * cfg["channel_multiplier"])
        core.add_module(
            f"scale_{i}_up",
            Block(
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
            ),
        )
        dim = out_dim
        spatial *= 2

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

    return nn.Sequential(core, out_block)
