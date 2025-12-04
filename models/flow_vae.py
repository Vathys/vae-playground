import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseVAE
from models.blocks import Block, ConvBlock, ResidualConvBlock


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        def m(x):
            return F.softplus(x) - 1

        def h(x):
            return torch.tanh(x)

        def h_prime(x):
            return 1.0 - h(x) ** 2

        inner = (self.w * self.u).sum()
        u = self.u + (m(inner) - inner) * self.w / self.w.norm() ** 2
        activation = (self.w * x).sum(dim=1, keepdim=True) + self.b
        x = x + u * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1.0 + (u * psi).sum(dim=1, keepdim=True)))

        return x, log_det


class RadialFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        def m(x):
            return F.softplus(x)

        def h(r):
            return 1.0 / (a + r)

        def h_prime(r):
            return -h(r) ** 2

        a = torch.exp(self.a)
        b = -a + m(self.b)
        r = (x - self.c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - self.c)
        log_det = (self.d - 1) * torch.log(1.0 + tmp) + torch.log(
            1.0 + tmp + b * h_prime(r) * r
        )

        return x, log_det


class HouseholderFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.v = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        outer = self.v.t() * self.v
        v_sqr = self.v.norm() ** 2
        H = torch.eye(self.d).to(self.v.device) - 2.0 * outer / v_sqr
        x = torch.mm(H, x.t()).t()

        return x, 0


class NiceFlow(nn.Module):
    def __init__(self, dim, mask, final=False):
        super().__init__()

        self.final = final
        if self.final:
            self.scale = nn.Parameter(torch.zeros(1, dim))
        else:
            self.mask = mask
            self.coupling = nn.Sequential(
                nn.Linear(dim // 2, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim // 2),
            )

    def forward(self, x):
        if self.final:
            x = x * torch.exp(self.scale)
            log_det = torch.sum(self.scale)

            return x, log_det
        else:
            [B, W] = list(x.size())
            x = x.reshape(B, W // 2, 2)

            if self.mask:
                on, off = x[:, :, 0], x[:, :, 1]
            else:
                off, on = x[:, :, 0], x[:, :, 1]

            on = on + self.coupling(off)

            if self.mask:
                x = torch.stack((on, off), dim=2)
            else:
                x = torch.stack((off, on), dim=2)

            return x.reshape(B, W), 0


class Flow(nn.Module):
    def __init__(self, dim, type, length):
        super().__init__()

        if type == "planar":
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])
        elif type == "radial":
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        elif type == "householder":
            self.flow = nn.ModuleList([HouseholderFlow(dim) for _ in range(length)])
        elif type == "nice":
            self.flow = nn.ModuleList(
                [NiceFlow(dim, i // 2, i == (length - 1)) for i in range(length)]
            )
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x: Tensor):
        [B, _] = list(x.size())
        log_det = torch.zeros(B, 1).to(x.device)

        for i in range(len(self.flow)):
            x, inc = self.flow[i](x)
            log_det += inc

        return x, log_det


class FlowVAE(BaseVAE):
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

        self.flow_type = kwargs["flow_type"]
        self.flow_length = kwargs["flow_length"]

        self.encoder, self.feature_dim = self._build_encoder()
        self.feature_size = self.in_size // 2**self.scale
        flat_dim = self.feature_dim * self.feature_size**2

        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_var = nn.Linear(flat_dim, self.latent_dim)

        self.flow = Flow(self.latent_dim, self.flow_type, self.flow_length)
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

    def encode(self, data):
        x = data["input"]
        x = self.encoder(x)
        [_, C, H, W] = list(x.size())
        assert C == self.feature_dim
        assert H == self.feature_size
        assert W == self.feature_size

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return {"mu": mu, "log_var": log_var}

    def decode(self, data):
        z = data["z"]
        x = self.project(z)
        x = x.reshape(-1, self.feature_dim, self.feature_size, self.feature_size)
        x = self.decoder(x)
        return {"output": x}

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return self.flow(z)

    def forward(self, data):
        encoded = self.encode(data)
        z, log_det = self.reparametrize(encoded["mu"], encoded["log_var"])
        decoded = self.decode({"z": z, "log_det": log_det})

        return {
            "input": data["input"],
            "output": decoded["output"],
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
        }

    def loss_function(self, data):
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"]
        log_var = data["log_var"]

        sigma = 1.0

        res_dict = {}

        nll_loss = F.mse_loss(x_hat, x, reduction="none")
        nll_loss = nll_loss.view(nll_loss.size(0), -1).sum(dim=1) / (2.0 * sigma**2)
        res_dict["nll"] = nll_loss.mean().detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        res_dict["kld"] = kld_loss.mean().detach()

        loss = nll_loss + kld_loss
        loss = loss.mean()
        res_dict["loss"] = loss

        res_dict["elbo"] = -loss.detach()

        return res_dict

    def sample_test(self, num: int, inter: int = 5, batch_size: int = 1):
        device = next(self.parameters()).device
        anchors = torch.randn(num * 2, self.latent_dim, device=device)

        pairs = anchors.view(num, 2, self.latent_dim)

        all_interps = []

        t_vals = torch.linspace(0, 1, inter, device=device)

        for i in range(num):
            z1, z2 = pairs[i]

            interped = (1 - t_vals[:, None]) * z1[None, :] + t_vals[:, None] * z2[
                None, :
            ]

            all_interps.append(interped)

        all_interps = torch.cat(all_interps, dim=0)

        total = all_interps.size(0)
        if total % batch_size != 0:
            raise ValueError(
                f"Cannot divide {total} vectors evenly into batch_size={batch_size}"
            )

        batched = all_interps.view(total // batch_size, batch_size, self.latent_dim)

        return [{"z": batched[i]} for i in batched.size(0)]
