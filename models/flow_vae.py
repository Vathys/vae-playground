import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseVAE
from models.blocks import build_decoder, build_encoder
from utils import lerp_z


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
        self.latent_dim = kwargs["latent_dim"]
        self.flow_type = kwargs["flow_type"]
        self.flow_length = kwargs["flow_length"]

        enc_cfg = kwargs["encoder"]
        dec_cfg = kwargs["decoder"]

        self.encoder, self.enc_out_dim, self.enc_out_hw = build_encoder(enc_cfg)

        flat_dim = self.enc_out_dim * (self.enc_out_hw**2)

        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_var = nn.Linear(flat_dim, self.latent_dim)
        self.project = nn.Linear(self.latent_dim, flat_dim)

        dec_cfg["in_channels"] = self.enc_out_dim
        dec_cfg["in_size"] = self.enc_out_hw

        self.decoder = build_decoder(dec_cfg)

        self.flow = Flow(self.latent_dim, self.flow_type, self.flow_length)

    def encode(self, data):
        x = data["input"]
        x = self.encoder(x)
        [_, C, H, W] = list(x.size())
        assert C == self.enc_out_dim
        assert H == self.enc_out_hw
        assert W == self.enc_out_hw

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return {"mu": mu, "log_var": log_var}

    def decode(self, data):
        z = data["z"]
        x = self.project(z)
        x = x.reshape(-1, self.enc_out_dim, self.enc_out_hw, self.enc_out_hw)
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

            interped = lerp_z(z1, z2, t_vals)

            all_interps.append(interped)

        all_interps = torch.cat(all_interps, dim=0)

        total = all_interps.size(0)
        if total % batch_size != 0:
            raise ValueError(
                f"Cannot divide {total} vectors evenly into batch_size={batch_size}"
            )

        batched = all_interps.view(total // batch_size, batch_size, self.latent_dim)

        return [{"z": batched[i]} for i in range(batched.size(0))]
