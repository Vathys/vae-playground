from collections import Counter
from pprint import pprint
from typing import Dict, Tuple

import lightning as L
import torch
import torchvision.utils as vutils
from torch import Tensor, optim
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance as KID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM

from models import getVAE


def kl_per_dim(mu: Tensor, log_var: Tensor):
    return 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0)


def kl_stats(
    mu: Tensor, log_var: Tensor, threshold=0.01
) -> Tuple[Tensor, Tensor, Tensor]:
    mu = mu.flatten(start_dim=1)  # flatten into [B, latent_dim]
    log_var = log_var.flatten(start_dim=1)

    kp = kl_per_dim(mu, log_var)
    kp_mean = kp.mean(dim=0)
    active = (kp_mean > threshold).sum()
    mean_kl = kp_mean.mean()
    mean_variance = log_var.exp().mean()

    return active, mean_kl, mean_variance


class Validator:
    def __init__(self, metrics):
        # Reconstruction Metrics
        if "lpips" in metrics:
            self.lpips = LPIPS(net_type="vgg").eval()
        else:
            self.lpips = None
        if "ssim" in metrics:
            self.ssim = SSIM(data_range=1.0).eval()
        else:
            self.ssim = None
        if "psnr" in metrics:
            self.psnr = PSNR(data_range=1.0).eval()
        else:
            self.psnr = None

        # Generation Metrics
        if "fid" in metrics:
            self.fid = FID(feature=2048, normalize=True).eval()
        else:
            self.fid = None
        if "kid" in metrics:
            self.kid = KID(feature=2048, subset_size=50, normalize=True).eval()
        else:
            self.kid = None
        if "inception_score" in metrics:
            self.inception = InceptionScore(feature=2048, normalize=True).eval()
        else:
            self.inception = None

    def to(self, device):
        if self.ssim is not None:
            self.ssim.to(device)

        if self.lpips is not None:
            self.lpips.to(device)

        if self.psnr is not None:
            self.psnr.to(device)

        if self.fid is not None:
            self.fid.to(device)

        if self.kid is not None:
            self.kid.to(device)

        if self.inception is not None:
            self.inception.to(device)

    def validate(self, x_hat, x, mu, log_var):
        res = {}

        if self.ssim is not None:
            res["ssim"] = self.ssim(x_hat, x)

        if self.lpips is not None:
            res["lpips"] = self.lpips(x_hat, x)

        if self.psnr is not None:
            res["psnr"] = self.psnr(x_hat, x)

        active_dim, mean_kl, mean_variance = kl_stats(mu, log_var)

        res["active_dim"] = active_dim.to(torch.float32)
        res["mean_kl"] = mean_kl.to(torch.float32)
        res["mean_variance"] = mean_variance.to(torch.float32)

        return res

    def update(self, x, real):
        if self.fid is not None:
            self.fid.update(x, real)

        if self.kid is not None:
            self.kid.update(x, real)

        if self.inception is not None and not real:
            self.inception.update(x)

    def compute(self):
        res = {}

        if self.fid is not None:
            fid_val = self.fid.compute()
            res["fid"] = fid_val
            self.fid.reset()

        if self.kid is not None:
            kid_mean, kid_std = self.kid.compute()
            res["kid_mean"] = kid_mean
            res["kid_std"] = kid_std
            self.kid.reset()

        if self.inception is not None:
            is_mean, is_std = self.inception.compute()
            res["inception_score_mean"] = is_mean
            res["inception_score_std"] = is_std
            self.inception.reset()

        return res


class VAEExperiment(L.LightningModule):
    def __init__(self, model_params, experiment_params) -> None:
        super().__init__()

        self.params = experiment_params
        self.model_params = model_params
        self.model = getVAE(self.model_params["name"], **self.model_params)

        self.test_input = None

        self.train_batch_resolutions = []
        self.val_batch_resolutions = []

        self.val_cond_norms = []

        self.validator = Validator(experiment_params["metrics"])

        self.save_hyperparameters()
        self.automatic_optimization = False

    def state_dict(self):
        state_dict = self.model.state_dict()
        return {f"model.{key}": val for key, val in state_dict.items()}

    def on_train_start(self):
        print("--------------------------")
        print("Model Parameters")
        print("--------------------------")
        pprint(self.model_params)

        print("--------------------------")
        print("Experiment Parameters")
        print("--------------------------")
        pprint(self.params)

    def forward(self, data) -> Dict[str, Tensor]:
        return self.model(data)

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()

        if not isinstance(optim, list):
            optim = [optim]

        _, _, H, W = batch["input"].shape
        if H > W:
            self.train_batch_resolutions.append(H)
        else:
            self.train_batch_resolutions.append(W)

        assert len(optim) < 3, "Only 2 stage optimization supported"

        optim[0].zero_grad()

        results = self.forward(batch)

        results["global_step"] = self.global_step
        results["current_epoch"] = self.current_epoch

        train_loss = self.model.loss_function(results)

        self.manual_backward(train_loss["loss"])
        if self.params["clip_gradient"]:
            self.clip_gradients(
                optim[0],
                gradient_clip_val=self.params["gradient_clip_val"],
                gradient_clip_algorithm=self.params["gradient_clip_algorithm"],
            )
        optim[0].step()

        self.log_dict(
            {f"train/{key}": val.item() for key, val in train_loss.items()},
            sync_dist=True,
        )

        if len(optim) > 1:
            optim[1].zero_grad()

            results = self.forward(batch)

            results["global_step"] = self.global_step
            results["current_epoch"] = self.current_epoch

            train_loss_2 = self.model.loss_function(results, stage="2")

            self.manual_backward(train_loss_2["loss"])
            if self.params["clip_gradient"]:
                self.clip_gradients(
                    optim[1],
                    gradient_clip_val=self.params["gradient_clip_val"],
                    gradient_clip_algorithm=self.params["gradient_clip_algorithm"],
                )
            optim[1].step()

            self.log_dict(
                {
                    f"train/stage2/{key}": val.item()
                    for key, val in train_loss_2.items()
                },
                sync_dist=True,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.log_grads()

    def on_train_epoch_end(self):
        schs = self.lr_schedulers()

        if schs is not None:
            if not isinstance(schs, list):
                schs = [schs]

            for sch in schs:
                sch.step()

        res_tensor = torch.tensor(
            self.train_batch_resolutions, dtype=torch.int, device=self.device
        )

        self.logger.experiment.add_histogram(
            "train/batch_resolutions", res_tensor, self.global_step
        )

        count = Counter(res_tensor.tolist())

        self.logger.experiment.add_text(
            "train/batch_resolutions", str(dict(count)), self.global_step
        )

        self.train_batch_resolutions.clear()

    def validation_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        if isinstance(optimizers, list) and len(optimizers) > 1:
            has_stage2 = True
        else:
            has_stage2 = False

        _, _, H, W = batch["input"].shape
        if H > W:
            self.val_batch_resolutions.append(H)
        else:
            self.val_batch_resolutions.append(W)

        if self.test_input is None:
            self.test_input = {"input": batch["input"][:36]}

        results = self.forward(batch)

        results["global_step"] = self.global_step
        results["current_epoch"] = self.current_epoch

        val_loss = self.model.loss_function(results)

        self.log_dict(
            {f"val/{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
        )

        if has_stage2:
            val_loss_2 = self.model.loss_function(results, stage="2")

            self.log_dict(
                {f"val/stage2/{key}": val.item() for key, val in val_loss_2.items()},
                sync_dist=True,
            )

        if "cond" in results:
            cond_norms = results["cond"].norm(dim=1)
            self.val_cond_norms.extend(cond_norms.tolist())

        x = results["input"]
        x_hat = results["output"]
        mu = results["mu"]
        log_var = results["log_var"]

        self.validator.to(self.device)
        metrics = self.validator.validate(x_hat, x, mu, log_var)

        self.log_dict(
            {f"val/metrics/{key}": val for key, val in metrics.items()}, sync_dist=True
        )

        self.validator.update(x, real=True)
        self.validator.update(x_hat, real=False)

    def on_validation_epoch_end(self):
        self.validator.to(self.device)
        metrics = self.validator.compute()

        self.log_dict(
            {f"val/metrics/{key}": val for key, val in metrics.items()}, sync_dist=True
        )

        res_tensor = torch.tensor(
            self.val_batch_resolutions, dtype=torch.int, device=self.device
        )

        self.logger.experiment.add_histogram(
            "val/batch_resolutions", res_tensor, self.global_step
        )

        count = Counter(res_tensor.tolist())

        self.logger.experiment.add_text(
            "val/batch_resolutions", str(dict(count)), self.global_step
        )

        self.val_batch_resolutions.clear()

        if len(self.val_cond_norms) > 0:
            cond_tensor = torch.tensor(self.val_cond_norms, device=self.device)

            self.logger.experiment.add_histogram(
                "val/cond_norms", cond_tensor, self.global_step
            )

            self.val_cond_norms.clear()

    def on_validation_end(self) -> None:
        self.sample_images()

    def log_grads(self):
        grad_txt = ""
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                if name.endswith("bias"):
                    continue

                self.logger.experiment.add_histogram(
                    f"grad/{name}", param.grad.detach(), self.global_step
                )
                norm = torch.norm(param.grad.detach(), 2)
                grad_txt += f"{name} {norm.item():.3f}  \n"

        # self.logger.experiment.add_text("grad/summary", grad_txt, self.global_step)

    def sample_images(self):
        output = self.model.forward(self.test_input)["output"]
        grid = vutils.make_grid(output, nrow=6)
        self.logger.experiment.add_image("reconstruction", grid, self.global_step)

        anchors_a = {"input": self.test_input["input"][:6, ...]}
        anchors_b = {"input": self.test_input["input"][6:12, ...]}

        encoded_a = self.model.encode(anchors_a)
        encoded_b = self.model.encode(anchors_b)

        interp_list = self.model.interpolate(
            encoded_a, encoded_b, steps=6, batch_size=6 * 6
        )

        for i, interp in enumerate(interp_list):
            grid = vutils.make_grid(interp["output"], nrow=6)
            self.logger.experiment.add_image(
                f"interpolated {i}", grid, self.global_step
            )

    def configure_optimizers(self):
        optims = []
        scheds = []

        def get_optimizer(**optim_params):
            model = (
                getattr(self.model, optim_params["submodel"])
                if "submodel" in optim_params
                else self.model
            )
            optimizer = optim.AdamW(
                model.parameters(),
                lr=optim_params["lr"],
                weight_decay=optim_params["weight_decay"],
            )

            # Use Linear LR as warmup and keep constant
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=optim_params["warmup_epochs"],
                last_epoch=-1,
            )

            return optimizer, scheduler

        optimizer1, scheduler1 = get_optimizer(**self.params["optim1"])
        optims.append(optimizer1)
        scheds.append(scheduler1)

        if "optim2" in self.params and self.params["optim2"] is not None:
            optimizer2, scheduler2 = get_optimizer(**self.params["optim2"])
            optims.append(optimizer2)
            scheds.append(scheduler2)

        return optims, scheds
