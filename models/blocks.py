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
        "None": lambda dim: Identity(),
        "ReLU": lambda dim: nn.ReLU(),
        "PReLU": lambda dim: nn.PReLU(dim),
        "PReLU1": lambda dim: nn.PReLU(1),
        "LeakyReLU": lambda dim: nn.LeakyReLU(),
        "SELU": lambda dim: nn.SELU(),
        "CELU": lambda dim: nn.CELU(),
        "GELU": lambda dim: nn.GELU(),
        "SiLU": lambda dim: nn.SiLU(),
        "Sigmoid": lambda dim: nn.Sigmoid(),
    }

    NORMS = {
        "None": lambda dim: Identity(),
        "BatchNorm": lambda dim: nn.BatchNorm2d(dim),
        "LayerNorm": lambda dim: nn.GroupNorm(1, dim),
        "InstanceNorm": lambda dim: nn.GroupNorm(dim, dim),
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
        norm="BatchNorm",
        activation="LeakyReLU",
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
        norm="BatchNorm",
        activation="ReLU",
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
