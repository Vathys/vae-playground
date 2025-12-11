import math
from typing import Dict, Tuple, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms as T

from external.magface import load_magface
from models.base import BaseVAE
from models.blocks import build_decoder, build_encoder
from utils import lerp_z, slerp_z


class ConditionedVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.loss_type = kwargs.get("loss_type", None)
        self.beta = kwargs.get("beta", None)
        self.gamma = kwargs.get("gamma", None)
        self.C_max = kwargs.get("max_capacity", None)
        self.C_stop_epoch = kwargs.get("C_stop_epoch", 75)
        self.C_type = kwargs.get("C_type", "linear")

        identity_model_path = kwargs["identity_model_path"]
        self.identity_model = load_magface(identity_model_path).requires_grad_(False)
        self.identity_model.eval()

        self.id_preprocess = T.Resize((112, 112))
        self.identity_model = self.identity_model
        self.id_dim = 512

        enc_cfg = kwargs["encoder"]
        dec_cfg = kwargs["decoder"]

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

    def _extract_identity(self, x: torch.Tensor) -> torch.Tensor:
        x = self.id_preprocess(x)

        feat = self.identity_model(x)
        feat = feat / feat.norm(dim=1, keepdim=True)

        return feat

    def encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]
        cond = data.get("cond", None)
        x = self.encoder(x, cond)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return {"mu": mu, "log_var": log_var}

    def decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z = data["z"]
        cond = data.get("cond", None)

        x = self.project(z)
        x = self.decoder(x, cond)
        return {"output": x}

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data["input"]
        feat = self._extract_identity(x)

        encoded = self.encode({**data, "cond": feat})

        z = self.reparametrize(encoded["mu"], encoded["log_var"])

        decoded = self.decode({"z": z, "cond": feat})

        return {
            "input": data["input"],
            "output": decoded["output"],
            "cond": feat,
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

        if self.loss_type == "B" and self.beta is not None:
            loss += self.beta * kld_loss
        elif (
            self.loss_type == "H" and self.gamma is not None and self.C_max is not None
        ):
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
        else:
            loss += kld_loss

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

        p_num = num // 2

        z_anchors = torch.randn(p_num * 2, self.latent_dim, *latent_size, device=device)
        f_rad = 1 + 0.05 * torch.randn(p_num * 2, 1, device=device)
        f_anchors = torch.randn(p_num * 2, self.id_dim, device=device)
        f_anchors = f_rad * f_anchors / f_anchors.norm(dim=1, keepdim=True)

        z_pairs = z_anchors.view(p_num, 2, self.latent_dim, *latent_size)
        f_pairs = f_anchors.view(p_num, 2, self.id_dim)

        all_z = []
        all_f = []

        t_vals = torch.linspace(0, 1, inter, device=device)

        for i in range(p_num):
            z1 = z_pairs[i, 0]
            z2 = z_pairs[i, 1]
            f1 = f_pairs[i, 0]
            f2 = f_pairs[i, 1]

            z_interp = lerp_z(z1, z2, t_vals)
            f_fixed = f1.expand(inter, self.id_dim)

            all_z.append(z_interp)
            all_f.append(f_fixed)

            mag_f1 = f1.norm(keepdim=True)
            dir_f1 = f1 / mag_f1
            mag_f2 = f2.norm(keepdim=True)
            dir_f2 = f2 / mag_f2

            dir_interp = slerp_z(dir_f1, dir_f2, t_vals)
            mag_interp = lerp_z(mag_f1, mag_f2, t_vals)

            f_interp = dir_interp * mag_interp
            z_fixed = z1.expand(inter, self.latent_dim, *latent_size)

            all_z.append(z_fixed)
            all_f.append(f_interp)

        if num % 2 == 1:
            extra_z_anchors = torch.randn(
                2, self.latent_dim, *latent_size, device=device
            )
            extra_f_rad = 1 + 0.05 * torch.randn(1, 1, device=device)
            extra_f_anchor = torch.randn(1, self.id_dim, device=device)
            extra_f_anchor = (
                extra_f_rad * extra_f_anchor / extra_f_anchor.norm(dim=1, keepdim=True)
            )

            z1 = extra_z_anchors[0]
            z2 = extra_z_anchors[1]

            z_interp = lerp_z(z1, z2, t_vals)
            f_fixed = extra_f_anchor.expand(inter, self.id_dim)

            all_z.append(z_interp)
            all_f.append(f_fixed)

        all_z = torch.cat(all_z, dim=0)
        all_f = torch.cat(all_f, dim=0)

        total = all_z.size(0)
        if total % batch_size != 0:
            raise ValueError(
                f"Cannot divide {total} vectors evenly into batch_size={batch_size}"
            )

        batched_z = all_z.view(
            total // batch_size, batch_size, self.latent_dim, *latent_size
        )
        batched_f = all_f.view(total // batch_size, batch_size, self.id_dim)

        return [
            {"z": batched_z[i], "cond": batched_f[i]} for i in range(batched_z.size(0))
        ]
