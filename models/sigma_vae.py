import math
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.base import BaseVAE
from models.blocks import build_network
from utils import combine_dict, lerp_z, split_dict


def softclip(tensor, min):
    return min + F.softplus(tensor - min)


class SigmaVAE(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__(
            use_lpips=kwargs.get("use_lpips", False),
            lpips_weight=kwargs.get("lpips_weight", 1.0),
        )
        self.latent_dim = kwargs["latent_dim"]
        self.residual_type = kwargs["residual_type"]
        self.variant = kwargs["variant"]
        self.epoch_end = kwargs.get("anneal_stop_epoch", 60)
        self.log_sigma_start = kwargs.get("logsigma_start", 0)
        self.log_sigma_end = kwargs.get("logsigma_end", -6)

        self.C_type = kwargs.get("C_type", None)
        self.gamma = kwargs.get("gamma", 10)
        self.C_max = kwargs.get("max_capacity", 50)
        self.k = 0.01

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

        self.log_sigma = 0
        if self.variant == "learn":
            self.log_sigma = torch.nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32),
                requires_grad=True,
            )

        self.C = 0
        if self.C_type == "learn":
            self.C = torch.nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32), requires_grad=True
            )

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

    def loss_function(
        self, data: Dict[str, Tensor], stage: str | None = None
    ) -> Dict[str, Tensor]:
        x = data["input"]
        x_hat = data["output"]
        mu = data["mu"].flatten(start_dim=1)
        log_var = data["log_var"].flatten(start_dim=1)

        current_epoch = data["current_epoch"]

        res_dict = {}

        if self.variant == "optimal":
            log_sigma = (x_hat - x).pow(2).mean().sqrt().log()
            log_sigma = softclip(log_sigma, -6)
        elif self.variant == "learn":
            log_sigma = self.log_sigma
            log_sigma = softclip(log_sigma, -6)
        elif self.variant == "std":
            log_sigma = torch.zeros([], dtype=torch.float32, device=self.device)
        elif self.variant == "anneal":
            sig_diff = self.log_sigma_end - self.log_sigma_start
            log_sigma = torch.clamp(
                torch.tensor(
                    self.log_sigma_start + (sig_diff * current_epoch / self.epoch_end)
                ),
                min=min(self.log_sigma_start, self.log_sigma_end),
                max=max(self.log_sigma_end, self.log_sigma_start),
            )
        else:
            raise ValueError(f"sigma variant {self.variant} is not supported")

        if self.residual_type == "gaussian":
            nll_loss = self._gaussian_nll(x_hat, x, log_sigma)
        elif self.residual_type == "sech":
            nll_loss = self._sech_nll(x_hat, x, log_sigma)
        else:
            raise ValueError(
                f"residual distribution {self.residual_type} is not supported"
            )

        nll_loss = nll_loss.flatten(start_dim=1).sum(dim=1)
        nll_loss = nll_loss.mean()
        res_dict["nll"] = nll_loss.detach()
        res_dict["sigma"] = log_sigma.exp().detach()

        kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
        kld_loss = kld_loss.mean()
        res_dict["kld"] = kld_loss.detach()

        loss = nll_loss

        if self.C_type is not None:
            if self.C_type == "linear":
                C = torch.clamp(
                    torch.tensor(float(self.C_max), device=self.device)
                    / self.epoch_end
                    * current_epoch,
                    0,
                    self.C_max,
                )
            elif self.C_type == "exp":
                C = torch.clamp(
                    torch.tensor(float(self.C_max), device=self.device)
                    * (1 - math.exp(math.log(self.k) * current_epoch / self.epoch_end)),
                    0,
                    self.C_max,
                )
            elif self.C_type == "learn":
                C = self.C
            else:
                raise ValueError(f"C annealing function {self.C_type} not available")
            res_dict["C"] = C.detach()
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
