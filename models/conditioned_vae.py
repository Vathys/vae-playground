import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from models.base import BaseVAE
from models.blocks import build_decoder, build_encoder
from external.magface import load_magface
from utils import lerp_z, slerp_z


class ConditionedVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.loss_type = kwargs.get("loss_type", None)
        self.beta = kwargs["beta"] if "beta" in kwargs else None
        self.gamma = kwargs["gamma"] if "gamma" in kwargs else None
        self.C_max = kwargs["max_capacity"] if "max_capacity" in kwargs else None
        self.C_stop_iter = kwargs["C_stop_iter"] if "C_stop_iter" in kwargs else None

        identity_model_path = kwargs["identity_model_path"]
        self.identity_model = load_magface(identity_model_path).requires_grad_(False)
        self.identity_model.eval()

        self.id_preprocess = T.Resize((112, 112))
        self.identity_model = self.identity_model
        self.id_dim = 512

        self.id_weight = nn.Linear(self.id_dim, self.latent_dim)
        self.id_bias = nn.Linear(self.id_dim, self.latent_dim)

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

    def _extract_identity(self, x: torch.Tensor) -> torch.Tensor:
        x = self.id_preprocess(x)

        feat = self.identity_model(x)
        feat = feat / feat.norm(dim=1, keepdim=True)

        return feat

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
        feat = data["feat"]

        id_weight = self.id_weight(feat)
        id_bias = self.id_bias(feat)

        z_mean = z.mean(dim=1, keepdim=True)
        z_std = z.std(dim=1, keepdim=True)

        z = id_weight * z + id_bias
        z = (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)
        z = z * z_std + z_mean

        x = self.project(z)
        x = x.reshape(-1, self.enc_out_dim, self.enc_out_hw, self.enc_out_hw)
        x = self.decoder(x)
        return {"output": x}

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data):
        x = data["input"]
        feat = self._extract_identity(x)

        encoded = self.encode({**data, "feat": feat})

        z = self.reparametrize(encoded["mu"], encoded["log_var"])

        decoded = self.decode({"z": z, "feat": feat})

        return {
            "input": data["input"],
            "output": decoded["output"],
            "feat": feat,
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
        }

    def loss_function(self, data):
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"]
        log_var = data["log_var"]

        num_iters = data["global_step"]

        sigma = 1.0

        res_dict = {}

        nll_loss = F.mse_loss(x_hat, x, reduction="none")
        nll_loss = nll_loss.view(nll_loss.size(0), -1).sum(dim=1) / (2.0 * sigma**2)
        res_dict["nll"] = nll_loss.mean().detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        res_dict["kld"] = kld_loss.mean().detach()

        loss = nll_loss

        if self.loss_type == "B" and self.beta is not None:
            loss += self.beta * kld_loss
        elif (
            self.loss_type == "H" and self.gamma is not None and self.C_max is not None
        ):
            C = torch.clamp(
                torch.tensor([self.C_max], device=next(self.parameters()).device)
                / self.C_stop_iter
                * num_iters,
                0,
                self.C_max,
            )
            res_dict["C"] = C.detach()
            cap_kld_loss = (kld_loss - C).abs()
            loss += self.gamma * cap_kld_loss
        else:
            loss += kld_loss

        loss = loss.mean()
        res_dict["loss"] = loss

        res_dict["elbo"] = -(nll_loss + kld_loss).mean().detach()

        return res_dict

    def sample_test(self, num: int, inter: int = 5, batch_size: int = 1):
        device = next(self.parameters()).device

        p_num = num // 2

        z_anchors = torch.randn(p_num * 2, self.latent_dim, device=device)
        f_rad = 1 + 0.05 * torch.randn(p_num * 2, 1, device=device)
        f_anchors = torch.randn(p_num * 2, self.id_dim, device=device)
        f_anchors = f_rad * f_anchors / f_anchors.norm(dim=1, keepdim=True)

        z_pairs = z_anchors.view(p_num, 2, self.latent_dim)
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
            z_fixed = z1.expand(inter, self.latent_dim)

            all_z.append(z_fixed)
            all_f.append(f_interp)

        if num % 2 == 1:
            extra_z_anchors = torch.randn(2, self.latent_dim, device=device)
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

        batched_z = all_z.view(total // batch_size, batch_size, self.latent_dim)
        batched_f = all_f.view(total // batch_size, batch_size, self.id_dim)

        return [
            {"z": batched_z[i], "feat": batched_f[i]} for i in range(batched_z.size(0))
        ]
