import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseVAE
from models.blocks import build_decoder, build_encoder
from utils import lerp_z


class VanillaVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]

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
        return eps * std + mu

    def forward(self, data):
        encoded = self.encode(data)
        z = self.reparametrize(encoded["mu"], encoded["log_var"])
        decoded = self.decode({"z": z})

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
