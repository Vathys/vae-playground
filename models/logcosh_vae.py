from typing import Dict, Sequence, Tuple, Union, List

import math
import torch
import torch.nn as nn
from torch import Tensor

from models.base import BaseVAE
from models.blocks import build_network
from utils import lerp_z, split_dict, combine_dict


class LogCoshVAE(BaseVAE):
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
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"].flatten(start_dim=1)
        log_var = data["log_var"].flatten(start_dim=1)

        res_dict = {}

        # Here we apply the negative log likelihood
        # where the residual is a sech probability distribution
        # We make this comparable to the vanilla vae implementation
        # by assuming sigma=1 and ignore the constant term
        exp_term = math.pi * (x_hat - x) / 2.0
        nll_loss = exp_term + torch.log(1.0 + torch.exp(-2 * exp_term))
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
