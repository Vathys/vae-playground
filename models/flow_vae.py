from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseVAE
from models.blocks import build_network
from utils import combine_dict, lerp_z, split_dict


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
        super().__init__(
            use_lpips=kwargs.get("use_lpips", False),
            lpips_weight=kwargs.get("lpips_weight", 1.0),
        )
        self.latent_dim = kwargs["latent_dim"]
        self.flow_type = kwargs["flow_type"]
        self.flow_length = kwargs["flow_length"]

        enc_cfg = kwargs["encoder"]
        dec_cfg = kwargs["decoder"]

        self.encoder, self.enc_out_dim = build_network(enc_cfg)

        self.fc_mu = nn.Conv2d(
            self.enc_out_dim, self.latent_dim, kernel_size=3, stride=1, padding=1
        )
        self.fc_var = nn.Conv2d(
            self.enc_out_dim, self.latent_dim, kernel_size=3, stride=1, padding=1
        )

        dec_cfg["in_channels"] = self.latent_dim
        dec_cfg["base_dim"] = self.enc_out_dim

        self.decoder, self.dec_out_dim = build_network(dec_cfg)

        self.flow = Flow(self.latent_dim, self.flow_type, self.flow_length)

    def encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]
        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return {"mu": mu, "log_var": log_var}

    def decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z = data["z"]
        x = self.decoder(z)
        return {"output": x}

    def sample(
        self, latent_size: Union[int, Tuple[int, int], Sequence[int]], batch_size: int
    ):
        if isinstance(latent_size, int):
            latent_size = (latent_size, latent_size)
        elif isinstance(latent_size, tuple):
            latent_size = latent_size
        else:
            assert len(latent_size) >= 2
            latent_size = latent_size[:2]

        latents = torch.randn(
            batch_size, self.latent_dim, *latent_size, device=self.device
        )

        return {"z": latents, "latent_size": latent_size}

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return self.flow(z)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        encoded = self.encode(data)
        z, log_det = self.reparametrize(encoded["mu"], encoded["log_var"])
        decoded = self.decode({"z": z, "log_det": log_det})

        return {
            "input": data["input"],
            "output": decoded["output"],
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
        }

    def loss_function(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"].flatten(start_dim=1)
        log_var = data["log_var"].flatten(start_dim=1)

        log_sigma = torch.tensor([0.0], device=self.device)

        res_dict = {}

        nll_loss = self._gaussian_nll(x_hat, x, log_sigma)
        nll_loss = nll_loss.flatten(start_dim=1).sum(dim=1)
        nll_loss = nll_loss.mean()
        res_dict["nll"] = nll_loss.detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        kld_loss = kld_loss.mean()
        res_dict["kld"] = kld_loss.detach()

        loss = nll_loss + kld_loss
        res_dict["loss"] = loss

        res_dict["elbo"] = -(nll_loss + kld_loss).detach()

        return res_dict

    def interpolate(
        self,
        encoded_a: Dict[str, Tensor],
        encoded_b: Dict[str, Tensor],
        steps: int = 5,
        batch_size: int = 1,
    ) -> List[Dict[str, Tensor]]:
        anchors_a = self.reparametrize(encoded_a["mu"], encoded_a["log_var"])
        anchors_b = self.reparametrize(encoded_b["mu"], encoded_b["log_var"])

        assert anchors_a.size(0) == anchors_b.size(0)

        B, *size = anchors_a.shape

        t_vals = torch.linspace(0, 1, steps, device=self.device)

        z_interps = (
            lerp_z(anchors_a, anchors_b, t_vals)
            .transpose(0, 1)
            .contiguous()
            .view(B * steps, *size)
        )  # [B * steps, ...]

        split = split_dict({"z": z_interps}, batch_size)

        decoded = [self.decode(batch) for batch in split]

        decoded = combine_dict(decoded)

        return [decoded]
