from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.base import BaseVAE
from models.blocks import build_network
from utils import lerp_z


class VanillaVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]

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
        return eps * std + mu

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        encoded = self.encode(data)
        z = self.reparametrize(encoded["mu"], encoded["log_var"])
        decoded = self.decode({"z": z})

        return {
            "input": data["input"],
            "output": decoded["output"],
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
        }

    def loss_function(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"].flatten(start_dim=1)
        log_var = data["log_var"].flatten(start_dim=1)

        var = torch.tensor([1.0], device=device, requires_grad=True)

        res_dict = {}

        nll_loss = (x_hat - x).pow(2) / var
        nll_loss = (nll_loss + torch.log(var)) / 2
        nll_loss = nll_loss.view(nll_loss.size(0), -1).sum(dim=1)
        res_dict["nll"] = nll_loss.mean().detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        res_dict["kld"] = kld_loss.mean().detach()

        loss = nll_loss + kld_loss
        loss = loss.mean()
        res_dict["loss"] = loss

        res_dict["elbo"] = -(nll_loss + kld_loss).mean().detach()

        return res_dict

    def sample_test(
        self,
        latent_size: Union[int, Tuple[int, int], Sequence[int]],
        num: int,
        inter: int = 5,
        batch_size: int = 1,
    ):
        samples = self.sample(latent_size, num * 2)

        anchors = samples["z"]
        latent_size = samples["latent_size"]

        pairs = anchors.view(num, 2, self.latent_dim, *latent_size)

        all_interps = []

        t_vals = torch.linspace(0, 1, inter, device=self.device)

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

        batched = all_interps.view(
            total // batch_size, batch_size, self.latent_dim, *latent_size
        )

        return [{"z": batched[i]} for i in range(batched.size(0))]
