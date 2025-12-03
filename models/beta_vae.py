from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseVAE
from models.blocks import ResidualConvBlock, ConvBlock, Block

EPS = 1e-6


class BetaVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs["in_channels"]
        self.in_size = kwargs["in_size"]
        self.latent_dim = kwargs["latent_dim"]
        self.base_dim = kwargs["base_dim"]
        self.scale = kwargs["scale"]
        self.num_blocks = kwargs["num_blocks"]
        self.residual = kwargs["residual"]
        self.bottleneck = kwargs["bottleneck"]
        self.weight_norm = kwargs["weight_norm"]

        self.loss_type = kwargs["loss_type"]
        self.beta = kwargs["beta"] if "beta" in kwargs else None
        self.gamma = kwargs["gamma"] if "gamma" in kwargs else None
        self.C_max = kwargs["max_capacity"] if "max_capacity" in kwargs else None
        self.C_stop_iter = kwargs["C_stop_iter"] if "C_stop_iter" in kwargs else None

        assert (self.beta is not None) or (
            self.gamma is not None and self.C_max is not None
        )

        self.encoder, self.feature_dim = self._build_encoder()
        self.feature_size = self.in_size // 2**self.scale
        flat_dim = self.feature_dim * self.feature_size**2

        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_var = nn.Linear(flat_dim, self.latent_dim)

        self.project = nn.Linear(self.latent_dim, flat_dim)

        self.decoder = self._build_decoder(self.feature_dim)

    def _build_encoder(self):

        BlockType = ResidualConvBlock if self.residual else ConvBlock

        in_block = Block(
            self.in_channels,
            self.base_dim,
            (3, 3),
            stride=1,
            padding=1,
            bias=True,
            weight_norm=self.weight_norm,
            scale=True,
            norm="LayerNorm",
            activation="SiLU",
        )

        core_block = nn.Sequential()
        dim = self.base_dim
        for i in range(self.scale):
            for j in range(self.num_blocks):
                core_block.add_module(
                    f"scale_{i}_res_{j}",
                    BlockType(
                        dim,
                        bottleneck=self.bottleneck,
                        weight_norm=self.weight_norm,
                        norm="LayerNorm",
                        activation="SiLU",
                    ),
                )
            core_block.add_module(
                f"scale_{i}_out",
                Block(
                    dim,
                    dim * 2,
                    (3, 3),
                    stride=2,
                    padding=1,
                    bias=True,
                    weight_norm=self.weight_norm,
                    scale=True,
                    norm="LayerNorm",
                    activation="SiLU",
                ),
            )
            dim *= 2

        return nn.Sequential(in_block, core_block), dim

    def _build_decoder(self, dim):
        BlockType = ResidualConvBlock if self.residual else ConvBlock

        core_block = nn.Sequential()
        for i in reversed(range(self.scale)):
            for j in range(self.num_blocks):
                core_block.add_module(
                    f"scale_{i}_res_{j}",
                    BlockType(
                        dim,
                        bottleneck=self.bottleneck,
                        weight_norm=self.weight_norm,
                        transpose=True,
                        norm="LayerNorm",
                        activation="SiLU",
                    ),
                )
            core_block.add_module(
                f"scale_{i}_out",
                Block(
                    dim,
                    dim // 2,
                    (3, 3),
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                    weight_norm=self.weight_norm,
                    scale=True,
                    transpose=True,
                    norm="LayerNorm",
                    activation="SiLU",
                ),
            )
            dim //= 2

        assert dim == self.base_dim
        out_block = Block(
            dim,
            self.in_channels,
            (3, 3),
            stride=1,
            padding=1,
            bias=True,
            weight_norm=self.weight_norm,
            scale=True,
            norm="None",
            activation="Sigmoid",
        )

        return nn.Sequential(core_block, out_block)

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        [_, C, H, W] = list(x.size())
        assert C == self.feature_dim
        assert H == self.feature_size
        assert W == self.feature_size

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, z: torch.Tensor):
        x = self.project(z)
        x = x.reshape(-1, self.feature_dim, self.feature_size, self.feature_size)
        x = self.decoder(x)
        return x

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        num_iters = kwargs["global_step"]

        sigma = 1.0

        res_dict = {}

        nll_loss = F.mse_loss(recons, input, reduction="none")
        nll_loss = nll_loss.view(nll_loss.size(0), -1).sum(dim=1) / (2.0 * sigma**2)
        res_dict["nll"] = nll_loss.mean().detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        res_dict["kld"] = kld_loss.mean().detach()

        if self.loss_type == "B":
            loss = nll_loss + self.beta * kld_loss
            loss = loss.mean()
        elif self.loss_type == "H":
            C = torch.clamp(
                torch.tensor([self.C_max], device=next(self.parameters()).device)
                / self.C_stop_iter
                * num_iters,
                0,
                self.C_max,
            )
            res_dict["C"] = C.detach()
            cap_kld_loss = (kld_loss - C).abs()
            loss = nll_loss + self.gamma * cap_kld_loss
            loss = loss.mean()

        res_dict["loss"] = loss

        res_dict["elbo"] = -(nll_loss + kld_loss).mean().detach()

        return res_dict

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim).to(
            next(self.parameters()).device
        )
