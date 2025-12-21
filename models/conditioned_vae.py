import math
from typing import Dict, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from external.magface import load_magface
from models.base import BaseVAE
from models.blocks import build_network
from utils import lerp_z, slerp_z, split_dict, combine_dict


class ConditionedVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.cond_weight = kwargs["cond_weight"]

        self.C_type = kwargs.get("C_type", None)
        self.gamma = kwargs.get("gamma", 10)
        self.C_max = kwargs.get("max_capacity", 50)
        self.C_stop_epoch = kwargs.get("C_stop_epoch", 75)
        self.k = 0.01

        self.residual_type = kwargs["residual_type"]
        self.learn_log_sigma = kwargs.get("learn_log_sigma", False)

        identity_model_path = kwargs["identity_model_path"]
        self.identity_model = load_magface(identity_model_path)
        for param in self.identity_model.parameters():
            param.requires_grad = False

        self.identity_model = self.identity_model
        self.id_dim = 512

        enc_cfg = kwargs["encoder"]
        mix_cfg = kwargs["mixer"]
        dec_cfg = kwargs["decoder"]

        self.encoder, self.enc_out_dim = build_network(enc_cfg)

        self.fc_mu = nn.Conv2d(
            self.enc_out_dim, self.latent_dim, kernel_size=3, stride=1, padding=1
        )
        self.fc_var = nn.Conv2d(
            self.enc_out_dim, self.latent_dim, kernel_size=3, stride=1, padding=1
        )

        mix_cfg["in_channels"] = self.latent_dim
        mix_cfg["base_dim"] = self.enc_out_dim

        self.mixer, self.mix_out_dim = build_network(mix_cfg)

        dec_cfg["in_channels"] = self.mix_out_dim

        self.decoder, self.dec_out_dim = build_network(dec_cfg)

        if self.learn_log_sigma:
            self.log_sigma = torch.nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.log_sigma = torch.tensor(0.0, dtype=torch.float32)

    def _extract_identity(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, (112, 112), mode="bilinear", align_corners=False, antialias=False
        )

        feat = self.identity_model(x)

        return feat

    def encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]
        cond = self._extract_identity(x)

        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return {"mu": mu, "log_var": log_var, "cond": cond}

    def decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z = data["z"]
        cond = data.get("cond", None)
        z = self.mixer(z, cond)

        x = self.decoder(z)
        return {"output": x}

    def sample(
        self, latent_size: Union[int, Tuple[int, int], Sequence[int]], batch_size: int
    ) -> Dict[str, Tensor]:
        if isinstance(latent_size, int):
            latent_size = (latent_size, latent_size)
        elif isinstance(latent_size, tuple):
            latent_size = latent_size
        else:
            assert len(latent_size) >= 2
            latent_size = latent_size[:2]

        z_latents = torch.randn(
            batch_size, self.latent_dim, *latent_size, device=self.device
        )
        c_rad = 1 + 50 * torch.rand(batch_size, 1, device=self.device)
        c_latents = torch.randn(batch_size, self.id_dim, device=self.device)
        c_latents = c_rad * c_latents / c_latents.norm(p=2, dim=1, keepdim=True)

        return {"z": z_latents, "cond": c_latents, "latent_size": latent_size}

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        encoded = self.encode(data)
        z = self.reparametrize(encoded["mu"], encoded["log_var"])
        decoded = self.decode({"z": z, "cond": encoded["cond"]})

        return {
            "input": data["input"],
            "output": decoded["output"],
            "z": z,
            "cond": encoded["cond"],
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
        }

    def _gaussian_nll(self, x_hat: Tensor, x: Tensor):
        nll = torch.pow((x - x_hat) / self.log_sigma.exp(), 2) / 2
        nll = nll + self.log_sigma + 0.5 * math.log(2 * math.pi)
        return nll

    def _sech_nll(self, x_hat: Tensor, x: Tensor):
        exp_term = math.pi * (x - x_hat) / (2.0 * self.log_sigma.exp())
        nll = exp_term + torch.log(1 + torch.exp(-2 * exp_term))
        nll = nll + self.log_sigma
        return nll

    def loss_function(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"].flatten(start_dim=1)
        log_var = data["log_var"].flatten(start_dim=1)
        z = data["z"]
        cond = data["cond"]

        cond = torch.roll(cond, shifts=1, dims=0)
        rand_face = self.decode({"z": z, "cond": cond})
        cond_hat = self._extract_identity(rand_face["output"])

        current_epoch = data["current_epoch"]

        res_dict = {}

        if self.residual_type == "gaussian":
            nll_loss = self._gaussian_nll(x_hat, x)
        else:
            nll_loss = self._sech_nll(x_hat, x)

        nll_loss = nll_loss.flatten(start_dim=1).sum(dim=1)
        nll_loss = nll_loss.mean()
        res_dict["nll"] = nll_loss.detach()
        if self.learn_log_sigma:
            res_dict["sigma"] = self.log_sigma.exp().detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        kld_loss = kld_loss.mean()
        res_dict["kld"] = kld_loss.detach()

        cond_loss = (cond[:, None, :] @ cond_hat[:, :, None]).squeeze() / (
            cond.norm(dim=1) * cond_hat.norm(dim=1) + 1e-6
        )
        cond_loss = (1.0 - cond_loss).mean()
        res_dict["cond_loss"] = cond_loss.detach()

        loss = nll_loss + self.cond_weight * cond_loss

        if self.C_type is not None:
            if self.C_type == "linear":
                C = torch.clamp(
                    torch.tensor(float(self.C_max), device=device)
                    / self.C_stop_epoch
                    * current_epoch,
                    0,
                    self.C_max,
                )
            elif self.C_type == "exp":
                C = torch.clamp(
                    torch.tensor(float(self.C_max), device=device)
                    * (
                        1
                        - math.exp(math.log(self.k) * current_epoch / self.C_stop_epoch)
                    ),
                    0,
                    self.C_max,
                )
            else:
                raise ValueError(f"C annealing function {self.C_type} not available")
            res_dict["C"] = C
            cap_kld_loss = (kld_loss - C).abs()
            loss += self.gamma * cap_kld_loss
        else:
            loss += kld_loss

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
        z_anchors_a = self.reparametrize(encoded_a["mu"], encoded_a["log_var"])
        z_anchors_b = self.reparametrize(encoded_b["mu"], encoded_b["log_var"])

        c_anchors_a = encoded_a["cond"]
        c_anchors_b = encoded_b["cond"]

        assert z_anchors_a.size(0) == z_anchors_b.size(0)
        assert c_anchors_a.size(0) == c_anchors_b.size(0)

        B, *z_size = z_anchors_a.shape

        t_vals = torch.linspace(0, 1, steps, device=self.device)

        z_interp = (
            lerp_z(z_anchors_a, z_anchors_b, t_vals)
            .transpose(0, 1)
            .contiguous()
            .view(B * steps, *z_size)
        )

        mag_a = c_anchors_a.norm(p=2, dim=1, keepdim=True)
        dir_a = c_anchors_a / mag_a
        mag_b = c_anchors_b.norm(p=2, dim=1, keepdim=True)
        dir_b = c_anchors_b / mag_b

        dir_interp = slerp_z(dir_a, dir_b, t_vals)
        mag_interp = lerp_z(mag_a, mag_b, t_vals)

        c_interp = (
            (dir_interp * mag_interp)
            .transpose(0, 1)
            .contiguous()
            .view(B * steps, self.id_dim)
        )

        z_split = split_dict(
            {
                "z": z_interp,
                "cond": c_anchors_a.unsqueeze(1)
                .expand(B, steps, self.id_dim)
                .contiguous()
                .view(B * steps, self.id_dim),
            },
            batch_size,
        )

        c_split = split_dict(
            {
                "z": z_anchors_a.unsqueeze(1)
                .expand(B, steps, *z_size)
                .contiguous()
                .view(B * steps, *z_size),
                "cond": c_interp,
            },
            batch_size,
        )

        z_decoded = [self.decode(batch) for batch in z_split]
        c_decoded = [self.decode(batch) for batch in c_split]

        z_decoded = combine_dict(z_decoded)
        c_decoded = combine_dict(c_decoded)

        return [z_decoded, c_decoded]
