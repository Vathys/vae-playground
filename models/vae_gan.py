from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseVAE
from models.blocks import build_network
from utils import combine_dict, lerp_z, split_dict


class VAEGAN(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__(
            use_lpips=kwargs.get("use_lpips", False),
            lpips_weight=kwargs.get("lpips_weight", 1.0),
        )
        self.latent_dim = kwargs["latent_dim"]

        self.adv_weight = kwargs.get("adversarial_weight", 1.0)
        self.fm_weight = kwargs.get("feature_matching_weight", 0.0)
        self.instance_noise_factor = kwargs.get("instance_noise_factor", 0.1)
        self.label_smooth_factor = kwargs.get("label_smooth_factor", 0.1)

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

        self.decoder, _ = build_network(dec_cfg)

        # Used only for setting optimizer parameters
        self.generator = nn.ModuleList(
            [self.encoder, self.fc_mu, self.fc_var, self.decoder]
        )

        disc_cfg = kwargs["discriminator"]

        self.discriminator, _ = build_network(disc_cfg)

        self.disc_features = []
        self._register_disc_hooks()

    def _register_disc_hooks(self):
        def hook_fn(module, input, output):
            self.disc_features.append(output)

        for name, module in self.discriminator.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name != "" and "final" not in name and "out" not in name:
                    module.register_forward_hook(hook_fn)

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

    def discriminate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.disc_features = []

        d_out: Tensor = self.discriminator(x)

        d_out = d_out.flatten(start_dim=2).mean(dim=2)

        features = self.disc_features.copy()
        self.disc_features = []

        return {"logits": d_out, "features": features}

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

        real_outputs = self.discriminate(data["input"])
        fake_outputs = self.discriminate(decoded["output"])

        return {
            "input": data["input"],
            "output": decoded["output"],
            "mu": encoded["mu"],
            "log_var": encoded["log_var"],
            "d_real": real_outputs["logits"],
            "d_fake": fake_outputs["logits"],
            "real_features": real_outputs["features"],
            "fake_features": fake_outputs["features"],
        }

    def loss_function(
        self, data: Dict[str, Tensor], stage: str | None = None
    ) -> Dict[str, Tensor]:
        x = data["input"]
        x_hat = data["output"]

        res_dict = {}

        if stage is None or stage == "1":
            mu = data["mu"].flatten(start_dim=1)
            log_var = data["log_var"].flatten(start_dim=1)
            log_sigma = torch.tensor([0.0], device=self.device)

            fake_out = self.discriminate(x_hat)
            d_fake = fake_out["logits"]

            nll_loss = self._gaussian_nll(x_hat, x, log_sigma)
            nll_loss = nll_loss.flatten(start_dim=1).sum(dim=1)
            nll_loss = nll_loss.mean()
            res_dict["nll"] = nll_loss.detach()

            kld_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var, dim=1)
            kld_loss = kld_loss.mean()
            res_dict["kld"] = kld_loss.detach()

            real_labels = torch.ones_like(d_fake)
            adv_loss = F.binary_cross_entropy_with_logits(
                d_fake, real_labels, reduction="none"
            )
            adv_loss = adv_loss
            adv_loss = adv_loss.mean()
            res_dict["adv_loss"] = adv_loss.detach()

            fm_loss = torch.zeros(x.size(0), device=self.device)
            if self.fm_weight > 0:
                real_out = self.discriminate(x)
                real_features = real_out["features"]
                fake_features = fake_out["features"]

                for real_feat, fake_feat in zip(real_features, fake_features):
                    feat_loss = F.mse_loss(
                        fake_feat, real_feat.detach(), reduction="none"
                    )
                    fm_loss += feat_loss.flatten(start_dim=1).sum(dim=1)
                fm_loss = fm_loss / len(real_features)
                fm_loss = fm_loss.mean()
                res_dict["fm_loss"] = fm_loss.detach()

            loss = (
                (1 - self.adv_weight) * nll_loss
                + kld_loss
                + self.adv_weight * adv_loss
                + self.fm_weight * fm_loss
            )
            loss = loss.mean()
            res_dict["loss"] = loss

            res_dict["elbo"] = -loss.detach()
        elif stage == "2":
            # Instance Noise
            noisy_x = (
                x
                + torch.randn_like(x, device=self.device, requires_grad=False)
                * self.instance_noise_factor
            )
            noisy_x_hat = (
                x_hat
                + torch.randn_like(x_hat, device=self.device, requires_grad=False)
                * self.instance_noise_factor
            )

            real_out = self.discriminate(noisy_x)
            fake_out = self.discriminate(noisy_x_hat)
            d_fake = fake_out["logits"]
            d_real = real_out["logits"]

            # Label Smoothing
            smoother = (
                torch.rand_like(d_real, device=self.device, requires_grad=False)
                * self.label_smooth_factor
            )
            real_labels = (1 - smoother) + smoother * 0.5
            fake_labels = smoother * 0.5

            d_real_loss = F.binary_cross_entropy_with_logits(
                d_real, real_labels, reduction="none"
            )
            d_fake_loss = F.binary_cross_entropy_with_logits(
                d_fake, fake_labels, reduction="none"
            )
            d_loss = d_real_loss + d_fake_loss
            d_loss = d_loss.mean()

            res_dict = {
                "loss": d_loss,
                "d_real_loss": d_real_loss.mean().detach(),
                "d_fake_loss": d_fake_loss.mean().detach(),
                "d_acc_real": (torch.sigmoid(d_real) > 0.5).float().mean().detach(),
                "d_acc_fake": (torch.sigmoid(d_fake) < 0.5).float().mean().detach(),
            }

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
