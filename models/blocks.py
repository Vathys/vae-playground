from typing import List, NotRequired, Tuple, TypedDict, TypeVar, Union

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


class AdaIN(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()

        self.style_net = nn.Linear(cond_dim, in_dim * 2)

    @staticmethod
    def _calc_stats(x: Tensor) -> Tuple[Tensor, Tensor]:
        eps = 1e-6

        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)

        return mean, std

    def forward(self, x: Tensor, cond: Tensor):
        # x: [B, in_dim, H, W], cond: [B, cond_dim]
        style = self.style_net(cond)[..., None, None]
        gamma, beta = style.chunk(2, dim=1)

        mu_x, sigma_x = self._calc_stats(x)

        x_norm = (x - mu_x) / sigma_x

        return (1 + gamma) * x_norm + beta


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.film_net = nn.Linear(cond_dim, in_dim * 2)

    def forward(self, x: Tensor, cond: Tensor):
        # x: [B, in_dim, H, W], cond: [B, cond_dim]
        style = self.film_net(cond)[..., None, None]
        gamma, beta = style.chunk(2, dim=1)

        return (1 + gamma) * x + beta


class StatisticPreservingAdaIN(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.in_dim = in_dim

        self.style_net = nn.Linear(cond_dim, in_dim * 2)

    @staticmethod
    def _calc_stats(x: Tensor) -> Tuple[Tensor, Tensor]:
        eps = 1e-6

        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)

        return mean, std

    def forward(self, x: Tensor, cond: Tensor):
        # x: [B, in_dim, H, W], cond: [B, cond_dim]
        mu_x, sigma_x = self._calc_stats(x)

        style = self.style_net(cond)[..., None, None]
        gamma_c, beta_c = style.chunk(2, dim=1)

        x_norm = (x - mu_x) / sigma_x

        x_prime = (1 + gamma_c) * x_norm + beta_c

        mu_x_prime, sigma_x_prime = self._calc_stats(x_prime)

        x_prime_norm = (x_prime - mu_x_prime) / sigma_x_prime

        return sigma_x * x_prime_norm + mu_x


class CrossAttentionMix(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        C = in_dim // 8

        self.query = nn.Conv2d(in_dim, C, 1)
        self.key = nn.Linear(cond_dim, C)
        self.value = nn.Linear(cond_dim, in_dim)

        self.proj = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, cond: Tensor):
        B, C, H, W = x.shape

        q = (
            self.query(x).flatten(start_dim=2).permute(0, 2, 1).contiguous()
        )  # [B, HW, C']
        k = self.key(cond).unsqueeze(1)  # [B, 1, C']
        v = self.value(cond).unsqueeze(1)  # [B, 1, C]

        attn = q @ k.permute(0, 2, 1)  # [B, HW, 1]
        attn = attn.softmax(dim=1)

        out = attn @ v  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W).contiguous()

        return x + self.gamma * self.proj(out)


class WeightNormConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        weight_norm=True,
        scale=False,
        transpose=False,
        upsample_mode="nearest",
    ):
        """Intializes a Conv2d augmented with weight normalization.

        (See torch.nn.utils.weight_norm for detail.)

        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
            transpose: True if upsample before applying convolution, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()

        def norm_func(module: nn.Module) -> nn.Module:
            if weight_norm and scale:
                return nn.utils.parametrizations.weight_norm(module)
            else:
                return module

        if transpose:
            self.upsample = (
                nn.Identity()
                if stride == 1
                else nn.Upsample(scale_factor=stride, mode=upsample_mode)
            )
            self.conv = norm_func(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    bias=bias,
                )
            )
        else:
            self.upsample = nn.Identity()
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
        x = self.upsample(x)
        return self.conv(x)


class Block(nn.Module):
    ACTIVATIONS = {
        "none": lambda dim: nn.Identity(),
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
        "none": lambda dim: nn.Identity(),
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
            stride=stride,
            padding=padding,
            bias=bias,
            weight_norm=weight_norm,
            scale=scale,
            transpose=transpose,
            upsample_mode="bilinear",
        )

        if norm in self.NORMS.keys():
            self.norm_block = self.NORMS[norm](out_dim)
        else:
            raise ValueError(f"norm {norm} not supported")
        if activation in self.ACTIVATIONS.keys():
            self.activation_block = self.ACTIVATIONS[activation](out_dim)
        else:
            raise ValueError(f"activation {activation} not supported")

    def forward_conv(self, x):
        return self.conv(x)

    def forward_norm(self, x):
        return self.norm_block(x)

    def forward_act(self, x):
        return self.activation_block(x)

    def forward(self, x, **kwargs):
        x = self.forward_conv(x)
        x = self.forward_norm(x)
        x = self.forward_act(x)

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
        residual=False,
        add_coord_channel=False,
        **kwargs,
    ):
        super().__init__()

        self.residual = residual
        self.bottleneck = bottleneck
        self.add_coord_channel = add_coord_channel

        if add_coord_channel:
            indim = dim + 2
        else:
            indim = dim

        if bottleneck:
            self.c1 = Block(
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
            )
            self.c2 = Block(
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
            )
            self.c3 = Block(
                dim,
                dim,
                (1, 1),
                stride=1,
                padding=0,
                bias=True,
                weight_norm=weight_norm,
                scale=True,
                transpose=transpose,
            )
            self.components = [self.c1, self.c2, self.c3]
        else:
            self.c1 = Block(
                indim,
                dim,
                (kernel_size, kernel_size),
                stride=1,
                padding=kernel_size // 2,
                bias=False,
                weight_norm=weight_norm,
                scale=False,
                transpose=transpose,
                norm="none",
                activation="none",
            )
            self.c2 = Block(
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
            )
            self.components = [self.c1, self.c2]

    def forward(self, x, **kwargs):
        identity = x

        if self.add_coord_channel:
            x = concat_coords(x)

        for comp in self.components:
            x = comp(x)

        return x + identity if self.residual else x


class ConditionedBlock(ConvBlock):
    MIXERS = {
        "adain": AdaIN,
        "film": FiLM,
        "attn": CrossAttentionMix,
        "spadain": StatisticPreservingAdaIN,
    }

    def __init__(
        self,
        dim,
        cond_dim,
        kernel_size,
        bottleneck,
        weight_norm,
        transpose=False,
        norm="batch",
        activation="leaky_relu",
        mix_type="adain",
        residual=False,
        add_coord_channel=False,
        **kwargs,
    ):
        internal_norm = "none" if mix_type in ["adain", "film"] else norm

        super().__init__(
            dim=dim,
            kernel_size=kernel_size,
            bottleneck=bottleneck,
            weight_norm=weight_norm,
            transpose=transpose,
            norm=internal_norm,
            activation=activation,
            residual=residual,
            add_coord_channel=add_coord_channel,
            **kwargs,
        )

        def _get_mixer(mix_type, dim, cond_dim):
            if mix_type not in self.MIXERS:
                raise ValueError(f"Mixer {mix_type} is not supported")
            return self.MIXERS[mix_type](dim, cond_dim)

        self.mix1 = _get_mixer(mix_type, dim, cond_dim)
        self.mix2 = _get_mixer(mix_type, dim, cond_dim)

    def forward(self, x, cond, **kwargs):
        identity = x

        if self.add_coord_channel:
            x = concat_coords(x)

        if self.bottleneck:
            x = self.c1.forward_conv(x)
            x = self.c1.forward_norm(x)
            x = self.mix1(x, cond)
            x = self.c1.forward_act(x)

            x = self.c2(x)

            x = self.c3.forward_conv(x)
            x = self.c3.forward_norm(x)
            x = self.mix2(x, cond)
            x = self.c3.forward_act(x)
        else:
            x = self.c1.forward_conv(x)
            x = self.c1.forward_norm(x)
            x = self.mix1(x, cond)
            x = self.c1.forward_act(x)

            x = self.c2.forward_conv(x)
            x = self.c2.forward_norm(x)
            x = self.mix2(x, cond)
            x = self.c2.forward_act(x)

        return x + identity if self.residual else x


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
        - add_coord_channel (bool, optional): add a coord channel before scaling to
          help with absolute positioning while scaling (default: false)

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

    if cond_dim is None:
        BlockType = ConvBlock
    else:
        BlockType = ConditionedBlock

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

    core = nn.ModuleDict()

    for i, scale in enumerate(scale_list):
        block_residual = residual[i]
        block_norm = norm[i]
        block_act = activation[i]
        block_bn = bottleneck[i]
        block_wn = weight_norm[i]

        for j in range(num_blocks[i]):
            block_params = {
                "dim": dim,
                "kernel_size": ksize,
                "bottleneck": block_bn,
                "weight_norm": block_wn,
                "norm": block_norm,
                "activation": block_act,
                "residual": block_residual,
                "transpose": transpose,
                "add_coord_channel": (j == 0 and add_coord_channel),
                "cond_dim": cond_dim,
                "mix_type": mix_type,
            }
            core[f"block_{i}_{j}"] = BlockType(**block_params)

        ndim = int(dim * scale)
        scaling = Block(
            dim,
            ndim,
            (ksize, ksize),
            stride=2,
            padding=ksize // 2,
            bias=True,
            weight_norm=block_wn,
            scale=True,
            norm=block_norm,
            activation=block_act,
            transpose=transpose,
        )
        core[f"scale_{i}"] = scaling
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
            self.core = core
            self.out_block = out_block

        def forward(self, x, cond=None):
            if self.in_block is not None:
                if add_coord_channel:
                    x = concat_coords(x)

                x = self.in_block(x)

            for _, s in self.core.items():
                x = s(x, cond=cond)

            if self.out_block is not None:
                if add_coord_channel:
                    x = concat_coords(x)

                x = self.out_block(x)

            return x

    return Network(), dim
