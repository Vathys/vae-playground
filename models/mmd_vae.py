from typing import Dict, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from models.base import BaseVAE
from models.blocks import build_network
from utils import lerp_z, split_dict, combine_dict


class MMDVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.kernel_type = kwargs.get("kernel_type", "imq")
        self.kernel_scales = kwargs.get(
            "kernel_scales", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        self.kernel_bandwidth = kwargs.get("kernel_bandwidth", 1.0)
        self.lmbda = kwargs.get("lambda", 0.01)

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
            "z": z,
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
        }

    def _compute_rbf(self, z1: Tensor, z2: Tensor) -> Tensor:
        D = z1.size(-1)
        C_base = 2.0 * D * self.kernel_bandwidth**2

        k = 0

        for scale in self.scales:
            C = scale * C_base
            k += torch.exp(
                -torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C
            )

        return k

    def _compute_inv_mult_quad(self, z1: Tensor, z2: Tensor) -> Tensor:
        D = z1.size(-1)
        C_base = 2.0 * D * self.kernel_bandwidth**2

        k = 0

        for scale in self.scales:
            C = scale * C_base
            k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

        return k

    def _compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = x1.flatten(start_dim=1)
        x2 = x2.flatten(start_dim=1)

        if self.kernel_type == "rbf":
            result = self._compute_rbf(x1, x2)
        elif self.kernel_type == "imq":
            result = self._compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError(f"kernel_type {self.kernel_type} is not supported...")

        return result

    def _compute_mmd(self, z: Tensor):
        prior_z = torch.randn_like(z)

        B = z.shape[0]

        k_z_prior = self._compute_kernel(prior_z, prior_z)
        k_z = self._compute_kernel(z, z)
        k_cross = self._compute_kernel(prior_z, z)

        mmd_z = (k_z - k_z.diag().diag()).sum() / ((B - 1) * B)
        mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((B - 1) * B)
        mmd_cross = k_cross.sum() / (B**2)

        mmd = mmd_z + mmd_z_prior - 2 * mmd_cross

        return mmd

    def loss_function(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"].flatten(start_dim=1)
        log_var = data["log_var"].flatten(start_dim=1)
        z = data["z"]

        res_dict = {}

        nll_loss = (x_hat - x).pow(2) / 2.0
        nll_loss = nll_loss.view(nll_loss.size(0), -1).sum(dim=1)
        nll_loss = nll_loss.mean()
        res_dict["nll"] = nll_loss.detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        kld_loss = kld_loss.mean()
        res_dict["kld"] = kld_loss.detach()

        mmd_loss = self._compute_mmd(z)
        res_dict["mmd"] = mmd_loss.detach()

        loss = (
            nll_loss
            + (1 - self.alph) * kld_loss
            + (self.alph + self.lam - 1) * mmd_loss
        )
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
