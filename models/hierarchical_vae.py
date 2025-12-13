import warnings
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.base import BaseVAE
from models.blocks import build_network
from utils import lerp_z


class HierarchicalVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent1_dim = kwargs["latent_dim1"]
        self.latent2_dim = kwargs["latent_dim2"]

        enc1_cfg = kwargs["encoder1"]
        enc2_cfg = kwargs["encoder2"]
        dec2_cfg = kwargs["decoder2"]
        dec1_cfg = kwargs["decoder1"]

        self.encoder1, self.enc_out_dim1 = build_network(enc1_cfg)

        self.fc_mu1 = nn.Conv2d(
            self.enc_out_dim1, self.latent1_dim, kernel_size=3, stride=1, padding=1
        )
        self.fc_var1 = nn.Conv2d(
            self.enc_out_dim1, self.latent1_dim, kernel_size=3, stride=1, padding=1
        )

        enc2_cfg["in_channels"] = self.enc_out_dim1

        self.encoder2, self.enc_out_dim2 = build_network(enc2_cfg)

        self.fc_mu2 = nn.Conv2d(
            self.enc_out_dim2, self.latent2_dim, kernel_size=3, stride=1, padding=1
        )
        self.fc_var2 = nn.Conv2d(
            self.enc_out_dim2, self.latent2_dim, kernel_size=3, stride=1, padding=1
        )

        dec2_cfg["in_channels"] = self.latent2_dim
        dec2_cfg["base_dim"] = self.enc_out_dim2

        self.decoder2, self.dec_out_dim2 = build_network(dec2_cfg)

        self.prior_fc_mu = nn.Conv2d(
            self.dec_out_dim2, self.latent1_dim, kernel_size=1, stride=1
        )
        self.prior_fc_var = nn.Conv2d(
            self.dec_out_dim2, self.latent1_dim, kernel_size=1, stride=1
        )

        self.project1 = nn.Conv2d(
            self.latent1_dim,
            self.dec_out_dim2,
            kernel_size=dec1_cfg["kernel_size"],
            stride=1,
            padding=dec1_cfg["kernel_size"] // 2,
        )

        dec1_cfg["in_channels"] = self.dec_out_dim2 * 2

        self.decoder1, self.dec_out_dim1 = build_network(dec1_cfg)

    def encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]

        enc1 = self.encoder1(x)

        mu1 = self.fc_mu1(enc1)
        log_var1 = self.fc_var1(enc1)

        enc2 = self.encoder2(enc1)

        mu2 = self.fc_mu2(enc2)
        log_var2 = self.fc_var2(enc2)

        return {"mu": [mu1, mu2], "log_var": [log_var1, log_var2]}

    def decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z1 = data["z1"]
        z2 = data["z2"]

        x1 = self.project1(z1)

        dec2 = self.decoder2(z2)

        prior_mu = self.prior_fc_mu(dec2)
        prior_log_var = self.prior_fc_var(dec2)

        dec1 = self.decoder1(torch.concat([x1, dec2], dim=1))

        return {"output": dec1, "prior_mu": prior_mu, "prior_log_var": prior_log_var}

    def sample(
        self, latent_size: Union[int, Tuple[int, int], Sequence[int]], batch_size: int
    ):
        if isinstance(latent_size, int):
            warnings.warn(
                f"received only 1 latent size {latent_size}...\n"
                "predicting latent size #2 from latent size #1..."
            )
            latent_size1 = (latent_size, latent_size)
            latent_size2 = (latent_size // 2, latent_size // 2)
        elif isinstance(latent_size, tuple):
            warnings.warn(
                f"received only 1 latent size {latent_size}...\n"
                "predicting latent size #2 from latent size #1..."
            )
            latent_size1 = latent_size
            latent_size2 = (latent_size[0] // 2, latent_size[1] // 2)
        else:
            assert len(latent_size) >= 4
            latent_size1 = (latent_size[0], latent_size[1])
            latent_size2 = (latent_size[2], latent_size[3])

        latents1 = torch.randn(
            batch_size, self.latent1_dim, *latent_size1, device=self.device
        )
        latents2 = torch.randn(
            batch_size, self.latent2_dim, *latent_size2, device=self.device
        )

        return {
            "z1": latents1,
            "z2": latents2,
            "latent_size1": latent_size1,
            "latent_size2": latent_size2,
        }

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        encoded = self.encode(data)

        z1 = self.reparametrize(encoded["mu"][0], encoded["log_var"][0])
        z2 = self.reparametrize(encoded["mu"][1], encoded["log_var"][1])

        decoded = self.decode({"z1": z1, "z2": z2})

        return {
            "input": data["input"],
            "output": decoded["output"],
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
            "prior_mu": decoded["prior_mu"],
            "prior_log_var": decoded["prior_log_var"],
        }

    def loss_function(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        x = data["input"]
        x_hat = data["output"]
        mu1 = data["mu"][0].flatten(start_dim=1)
        log_var1 = data["log_var"][0].flatten(start_dim=1)
        mu2 = data["mu"][1].flatten(start_dim=1)
        log_var2 = data["log_var"][1].flatten(start_dim=1)
        prior_mu = data["prior_mu"].flatten(start_dim=1)
        prior_log_var = data["prior_log_var"].flatten(start_dim=1)

        var = torch.tensor([1.0], device=device, requires_grad=True)

        res_dict = {}

        nll_loss = (x_hat - x).pow(2) / var
        nll_loss = (nll_loss + torch.log(var)) / 2
        nll_loss = nll_loss.view(nll_loss.size(0), -1).sum(dim=1)
        res_dict["nll"] = nll_loss.mean().detach()

        kld2_loss = 0.5 * torch.sum(mu2.pow(2) + log_var2.exp() - 1.0 - log_var2, dim=1)
        res_dict["kld2"] = kld2_loss.mean().detach()

        kld1_loss = 0.5 * torch.sum(
            ((mu1 - prior_mu).pow(2) / prior_log_var.exp())
            + (log_var1.exp() / prior_log_var.exp())
            + prior_log_var
            - log_var1
            - 1.0,
            dim=1,
        )
        res_dict["kld1"] = kld1_loss.mean().detach()

        loss = nll_loss + kld2_loss + kld1_loss
        loss = loss.mean()
        res_dict["loss"] = loss

        res_dict["elbo"] = -loss.detach()

        return res_dict

    def sample_test(
        self,
        latent_size: Union[int, Tuple[int, int], Sequence[int]],
        num: int,
        inter: int = 5,
        batch_size: int = 1,
    ):
        p_num = num // 2

        samples = self.sample(latent_size, p_num * 2)

        anchors1 = samples["z1"]
        anchors2 = samples["z2"]
        latent_size1 = samples["latent_size1"]
        latent_size2 = samples["latent_size2"]

        pair1 = anchors1.view(p_num, 2, self.latent1_dim, *latent_size1)
        pair2 = anchors2.view(p_num, 2, self.latent2_dim, *latent_size2)

        all_z1 = []
        all_z2 = []

        t_vals = torch.linspace(0, 1, inter, device=self.device)

        for i in range(p_num):
            z11, z12 = pair1[i]
            z21, z22 = pair2[i]

            interped1 = lerp_z(z11, z12, t_vals)
            fixed2 = z21.expand(inter, self.latent2_dim, *latent_size2)

            all_z1.append(interped1)
            all_z2.append(fixed2)

            interped2 = lerp_z(z21, z22, t_vals)
            fixed1 = z11.expand(inter, self.latent1_dim, *latent_size1)

            all_z1.append(fixed1)
            all_z2.append(interped2)

        if num % 2 == 1:
            extra_samples = self.sample(latent_size, 2)
            extra_anchors1 = extra_samples["z1"]
            extra_anchor2 = extra_samples["z2"][0]

            z1 = extra_anchors1[0]
            z2 = extra_anchors1[1]

            interped1 = lerp_z(z1, z2, t_vals)
            fixed2 = extra_anchor2.expand(inter, self.latent2_dim, *latent_size2)

            all_z1.append(interped1)
            all_z2.append(fixed2)

        all_z1 = torch.cat(all_z1, dim=0)
        all_z2 = torch.cat(all_z2, dim=0)

        total = all_z1.size(0)
        if total % batch_size != 0:
            raise ValueError(
                f"Cannot divide {total} vectors evenly into batch_size={batch_size}"
            )

        batched_z1 = all_z1.view(
            total // batch_size, batch_size, self.latent1_dim, *latent_size1
        )
        batched_z2 = all_z2.view(
            total // batch_size, batch_size, self.latent2_dim, *latent_size2
        )

        return [
            {"z1": batched_z1[i], "z2": batched_z2[i]}
            for i in range(batched_z1.size(0))
        ]
