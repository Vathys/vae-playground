from typing import Dict, List, NotRequired, Optional, Tuple, TypedDict, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

T = TypeVar("T")

MList = Union[T, List[T]]


def make_coord_grid(h, w, device):
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)  # [2, H, W]


def concat_coords(x):
    B, _, H, W = x.shape

    coords = make_coord_grid(H, W, x.device)
    coords = coords.expand(B, 2, H, W)

    return torch.cat([coords, x], dim=1)


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
        kernel_size,
        bottleneck,
        weight_norm,
        transpose=False,
        norm="batch",
        activation="leaky_relu",
        add_coord_channel=False,
    ):
        """Initializes a Standard Block.

        Args:
            dim: number of input and output features.
            bottleneck: True if use bottleneck, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            transpose: True if transposed convolution, False otherwise.
        """
        super().__init__()

        self.add_coord_channel = add_coord_channel

        if add_coord_channel:
            indim = dim + 2
        else:
            indim = dim

        if bottleneck:
            self.block = nn.Sequential(
                Block(
                    indim,
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
                    (kernel_size, kernel_size),
                    stride=1,
                    padding=kernel_size // 2,
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
                    indim,
                    dim,
                    (kernel_size, kernel_size),
                    stride=1,
                    padding=kernel_size // 2,
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
                    (kernel_size, kernel_size),
                    stride=1,
                    padding=kernel_size // 2,
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
        if self.add_coord_channel:
            x = concat_coords(x)
        return self.block(x)


class ResidualConvBlock(ConvBlock):
    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        if self.add_coord_channel:
            nx = concat_coords(x)
        else:
            nx = x
        return x + self.block(nx)


class ScaleBlock(nn.Module):
    def __init__(
        self,
        blocks: Dict[str, nn.Module],
        scaling: Block,
        cond_dim: Optional[int],
        mix_type: str,
    ):
        super().__init__()
        self.blocks = nn.ModuleDict(blocks)
        self.scaling = scaling

        if cond_dim is not None:
            if mix_type == "film":
                self.mix_block = FiLM(cond_dim, self.scaling.out_channels)
            elif mix_type == "adain":
                self.mix_block = AdaIN(cond_dim, self.scaling.out_channels)
            elif mix_type == "norm_film":
                self.mix_block = NormalizedFiLM(cond_dim, self.scaling.out_channels)
            elif mix_type == "attention":
                self.mix_block = Attention(cond_dim, self.scaling.out_channels)
            else:
                raise ValueError(f"mix_type {mix_type} not available...")
        else:
            self.mix_block = None

    def forward(self, x, cond=None):
        for _, b in self.blocks.items():
            x = b(x)

        x = self.scaling(x)
        if self.mix_block is not None and cond is not None:
            x = self.mix_block(x, cond)

        return x


class NetworkConfig(TypedDict, total=True):
    in_channels: int
    base_dim: NotRequired[int]
    out_dim: NotRequired[int]
    scales: Union[int, List[float]]
    num_blocks: Union[int, List[int]]
    channel_multiplier: NotRequired[float]
    norm: str
    activation: str
    transpose: bool
    kernel_size: NotRequired[int]
    residual: Union[bool, List[bool]]
    bottleneck: Union[bool, List[bool]]
    weight_norm: Union[bool, List[bool]]
    final_activation: NotRequired[str]
    cond_dim: NotRequired[int]
    mix_type: NotRequired[str]
    add_coord_channel: NotRequired[bool]


def build_network(cfg: NetworkConfig) -> Tuple[nn.Module, int]:
    """Build a convolutional network with successive scaling.

    :param cfg: Network configuration:

    Config includes:
        - in_channels (int; required): number of input channels.
        - base_dim (int; optional): starting dimension for scaling
          if not included, then no block is applied and in_channels
          is used as base_dim.
        - out_dim (int; optional): ending dimension; if not included,
          then no block is applied.
        - scales (int, list(float); required): number of times to scale
          the image; if int, then scale number of times by channel_multiplier
        - num_blocks (int, list(int); required): number of convolution blocks
          per scale; if int, then same number of blocks are applied per scale
        - channel_multiplier (float, optional): if scales is not list, then
          use this number to scale channels
        - norm (string, list(string); required): type of normalization block;
          if list, then normalization is determined per scale
        - activation (string, list(string); required): type of activation; if
          list, then activation is determined per scale
        - transpose(bool; required): whether to apply transpose convolutions
        - kernel_size (int; optional): default kernel size for non-bottleneck
          convolutions (default: 3)
        - residual (bool, list(bool); optional): whether to make block residual;
          if list, then determined per scale
        - bottleneck (bool, list(bool); optional): whether to add convolutional
          bottleneck in a block; if list, then bottleneck is determined per scale
        - weight_norm (bool, list(bool); optional): whether to normalize weights;
          if list, then weight norm is determined per scale
        - final_activation (string; optional): activation to use in final
          convolution; if none given, then default is standard activation; if out
          channels is not given, then ignored (default: none)
        - upscale_out (bool; optional): apply predictive upscaling in out block
          (default: false)
        - cond_dim (int, optional): dimension of conditional vector
        - mix_type (string, optional): how to mix conditional vector into network
          (default: adain)
        - add_coord_channel (bool, optional): add a coord channel before scaling to help with
          absolute positioning while scaling (default: false)

    :type cfg: Dict[str, Optional[Any]]
    :return: Network and out dimension
    :rtype: Tuple[Module, int]
    """

    def by_scale(val: Union[T, List[T]], num: int) -> List[T]:
        if isinstance(val, List):
            assert len(val) == num
            return val
        else:
            return [val] * num

    cond_dim = cfg.get("cond_dim", None)
    mix_type = cfg.get("mix_type", "adain")

    in_channels = cfg["in_channels"]
    base_dim = cfg.get("base_dim", None)
    out_dim = cfg.get("out_dim", None)
    ksize = cfg.get("kernel_size", 3)
    assert ksize % 2 == 1

    if isinstance(cfg["scales"], int):
        assert "channel_multiplier" in cfg
        scale_list = [cfg["channel_multiplier"]] * cfg["scales"]
    else:
        scale_list = cfg["scales"]

    num_scales = len(scale_list)

    num_blocks = by_scale(cfg["num_blocks"], num_scales)
    norm = by_scale(cfg["norm"], num_scales)
    activation = by_scale(cfg["activation"], num_scales)
    transpose = cfg["transpose"]

    residual = cfg.get("residual", False)
    residual = by_scale(residual, num_scales)

    bottleneck = cfg.get("bottleneck", False)
    bottleneck = by_scale(bottleneck, num_scales)

    weight_norm = cfg.get("weight_norm", False)
    weight_norm = by_scale(weight_norm, num_scales)

    upscale_out = cfg.get("upscale_out", False)
    final_activation = cfg.get("final_activation", "none")

    add_coord_channel = cfg.get("add_coord_channel", False)

    if base_dim is not None:
        if add_coord_channel:
            in_channels += 2

        in_block = Block(
            in_channels,
            base_dim,
            (ksize, ksize),
            stride=1,
            padding=ksize // 2,
            bias=True,
            weight_norm=weight_norm[0],
            scale=True,
            norm=norm[0],
            activation=activation[0],
            transpose=False,
        )
        dim = base_dim
    else:
        in_block = None
        dim = in_channels

    scales = nn.ModuleDict()

    for i, scale in enumerate(scale_list):
        blocks = {}
        BlockType = ResidualConvBlock if residual[i] else ConvBlock
        block_norm = norm[i]
        block_act = activation[i]
        block_bn = bottleneck[i]
        block_wn = weight_norm[i]

        for j in range(num_blocks[i]):
            blocks[f"block{j}"] = BlockType(
                dim,
                kernel_size=ksize,
                bottleneck=block_bn,
                weight_norm=block_wn,
                norm=block_norm,
                activation=block_act,
                transpose=transpose,
                add_coord_channel=(j == 0 and add_coord_channel),
            )

        ndim = int(dim * scale)
        scaling = Block(
            dim,
            ndim,
            (ksize, ksize),
            stride=2,
            padding=ksize // 2,
            output_padding=1 if transpose else 0,
            bias=True,
            weight_norm=block_wn,
            scale=True,
            norm=block_norm,
            activation=block_act,
            transpose=transpose,
        )
        scales[f"scale_{i}"] = ScaleBlock(
            blocks=blocks, scaling=scaling, cond_dim=cond_dim, mix_type=mix_type
        )
        dim = ndim

    out_blocks = []

    if upscale_out:
        if add_coord_channel:
            out_indim = dim + 2
        else:
            out_indim = dim
        block_up = Block(
            out_indim,
            dim,
            kernel_size=(ksize, ksize),
            stride=2,
            padding=ksize // 2,
            output_padding=1,
            bias=True,
            weight_norm=weight_norm[-1],
            scale=False,
            norm=norm[-1],
            activation=activation[-1],
            transpose=True,
        )
        block_down = Block(
            dim,
            dim,
            kernel_size=(ksize, ksize),
            stride=2,
            padding=ksize // 2,
            bias=True,
            weight_norm=weight_norm[-1],
            scale=True,
            norm=norm[-1],
            activation=activation[-1],
            transpose=False,
        )
        out_blocks.append(block_up)
        out_blocks.append(block_down)

    if out_dim is not None:
        out_block = Block(
            dim,
            out_dim,
            (ksize, ksize),
            stride=1,
            padding=ksize // 2,
            bias=True,
            weight_norm=weight_norm[-1],
            scale=True,
            norm="none",
            activation=final_activation,
        )
        dim = out_dim
        out_blocks.append(out_block)

    if len(out_blocks) > 0:
        out_block = nn.Sequential(*out_blocks)
    else:
        out_block = None

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_block = in_block
            self.scales = scales
            self.out_block = out_block

        def forward(self, x, cond=None):
            if self.in_block is not None:
                if add_coord_channel:
                    x = concat_coords(x)

                x = self.in_block(x)

            for _, s in self.scales.items():
                x = s(x, cond)

            if self.out_block is not None:
                if add_coord_channel:
                    x = concat_coords(x)

                x = self.out_block(x)

            return x

    return Network(), dim
