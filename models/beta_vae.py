import math
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.base import BaseVAE
from models.blocks import build_decoder, build_encoder
from utils import lerp_z


class BetaVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.loss_type = kwargs["loss_type"]
        self.beta = kwargs.get("beta", None)
        self.gamma = kwargs.get("gamma", None)
        self.C_max = kwargs.get("max_capacity", None)
        self.C_stop_epoch = kwargs.get("C_stop_epoch", 75)
        self.C_type = kwargs.get("C_type", "linear")

        enc_cfg = kwargs["encoder"]
        dec_cfg = kwargs["decoder"]

        assert (self.beta is not None) or (
            self.gamma is not None and self.C_max is not None
        )

        self.encoder, self.enc_out_dim = build_encoder(enc_cfg)

        self.fc_mu = nn.Conv2d(
            self.enc_out_dim, self.latent_dim, kernel_size=1, stride=1
        )
        self.fc_var = nn.Conv2d(
            self.enc_out_dim, self.latent_dim, kernel_size=1, stride=1
        )

        self.project = nn.Conv2d(
            self.latent_dim, self.enc_out_dim, kernel_size=1, stride=1
        )

        dec_cfg["in_channels"] = self.enc_out_dim

        self.decoder, _ = build_decoder(dec_cfg)

    def encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]
        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return {"mu": mu, "log_var": log_var}

    def decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z = data["z"]
        x = self.project(z)
        x = self.decoder(x)
        return {"output": x}

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

        current_epoch = data["current_epoch"]

        var = torch.tensor([1.0], device=device, requires_grad=True)

        res_dict = {}

        nll_loss = (x_hat - x).pow(2) / var
        nll_loss = (nll_loss + torch.log(var)) / 2
        nll_loss = nll_loss.view(nll_loss.size(0), -1).sum(dim=1)
        res_dict["nll"] = nll_loss.mean().detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        res_dict["kld"] = kld_loss.mean().detach()

        loss = nll_loss

        if self.loss_type == "B":
            loss += self.beta * kld_loss
        elif self.loss_type == "H":
            if self.C_type == "linear":
                C = torch.clamp(
                    torch.tensor([float(self.C_max)], device=device, requires_grad=True)
                    / self.C_stop_epoch
                    * current_epoch,
                    0,
                    self.C_max,
                )
            elif self.C_type == "exp":
                C = torch.clamp(
                    torch.tensor([float(self.C_max)], device=device, requires_grad=True)
                    * (1 - math.exp(-3 * current_epoch / self.C_stop_epoch)),
                    0,
                    self.C_max,
                )
            else:
                raise ValueError(f"C annealing function {self.C_type} not available")
            res_dict["C"] = C.detach()
            cap_kld_loss = (kld_loss - C).abs()
            loss += self.gamma * cap_kld_loss

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
        device = next(self.parameters()).device
        if isinstance(latent_size, int):
            latent_size = (latent_size, latent_size)
        elif isinstance(latent_size, tuple):
            latent_size = latent_size
        else:
            assert len(latent_size) >= 2
            latent_size = latent_size[:2]

        anchors = torch.randn(num * 2, self.latent_dim, *latent_size, device=device)

        pairs = anchors.view(num, 2, self.latent_dim, *latent_size)

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

        batched = all_interps.view(
            total // batch_size, batch_size, self.latent_dim, *latent_size
        )

        return [{"z": batched[i]} for i in range(batched.size(0))]
