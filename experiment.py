from collections import Counter
from typing import Dict

import lightning as L
import torch
import torchvision.utils as vutils
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import Tensor, optim
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance as KID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM

from models import getVAE


class VAEExperiment(L.LightningModule):
    def __init__(self, model_params, experiment_params) -> None:
        super().__init__()

        self.model = getVAE(model_params["name"], **model_params)
        self.params = experiment_params

        self.test_input = None

        self.train_batch_resolutions = []
        self.val_batch_resolutions = []

        self.val_cond_norms = []

        # Validation Models

        # Reconstruction Metrics
        if "lpips" in experiment_params["metrics"]:
            self.lpips = LPIPS(net_type="vgg").eval()
        else:
            self.lpips = None
        if "ssim" in experiment_params["metrics"]:
            self.ssim = SSIM(data_range=1.0).eval()
        else:
            self.ssim = None
        if "psnr" in experiment_params["metrics"]:
            self.psnr = PSNR(data_range=1.0).eval()
        else:
            self.psnr = None

        # Generation Metrics
        if "fid" in experiment_params["metrics"]:
            self.fid = FID(feature=2048, normalize=True).eval()
        else:
            self.fid = None
        if "kid" in experiment_params["metrics"]:
            self.kid = KID(feature=2048, subset_size=50, normalize=True).eval()
        else:
            self.kid = None
        if "inception_score" in experiment_params["metrics"]:
            self.inception = InceptionScore(feature=2048, normalize=True).eval()
        else:
            self.inception = None

        self.save_hyperparameters()

    def state_dict(self):
        state_dict = self.model.state_dict()
        return {f"model.{key}": val for key, val in state_dict.items()}

    def on_train_start(self):
        self.model.to(self.device)

    def forward(self, data) -> Dict[str, Tensor]:
        return self.model(data)

    def training_step(self, batch, batch_idx):
        _, _, H, W = batch["input"].shape
        if H > W:
            self.train_batch_resolutions.append(H)
        else:
            self.train_batch_resolutions.append(W)

        results = self.forward(batch)

        results["global_step"] = self.global_step
        results["current_epoch"] = self.current_epoch

        train_loss = self.model.loss_function(results)

        self.log_dict(
            {f"train/{key}": val.item() for key, val in train_loss.items()},
            sync_dist=True,
        )

        return train_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.log_grads()

    def on_train_epoch_end(self):
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
            {f"val/{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

        if "cond" in results:
            cond_norms = results["cond"].norm(dim=1)
            self.val_cond_norms.extend(cond_norms.tolist())

        x = results["input"]
        x_hat = results["output"]

        if self.ssim is not None:
            self.log("val/metrics/ssim", self.ssim(x_hat, x), sync_dist=True)
        if self.lpips is not None:
            self.log("val/metrics/lpips", self.lpips(x_hat, x), sync_dist=True)
        if self.psnr is not None:
            self.log("val/metrics/psnr", self.psnr(x_hat, x), sync_dist=True)

        if self.fid is not None:
            self.fid.update(x, real=True)
            self.fid.update(x_hat, real=False)

        if self.kid is not None:
            self.kid.update(x, real=True)
            self.kid.update(x_hat, real=False)

        if self.inception is not None:
            self.inception.update(x_hat)

        return val_loss

    def on_validation_epoch_end(self):
        if self.fid is not None:
            fid_val = self.fid.compute()
            self.log("val/metrics/fid", fid_val, sync_dist=True)
            self.fid.reset()

        if self.kid is not None:
            kid_mean, kid_std = self.kid.compute()
            self.log("val/metrics/kid_mean", kid_mean, sync_dist=True)
            self.log("val/metrics/kid_std", kid_std, sync_dist=True)
            self.kid.reset()

        if self.inception is not None:
            is_mean, is_std = self.inception.compute()
            self.log("val/metrics/inception_score_mean", is_mean, sync_dist=True)
            self.log("val/metrics/inception_score_std", is_std, sync_dist=True)
            self.inception.reset()

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
            optimizer = optim.Adam(
                model.parameters(),
                lr=optim_params["lr"],
                weight_decay=optim_params["weight_decay"],
            )

            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=optim_params["warmup_epochs"],
                max_epochs=optim_params["max_epochs"],
                warmup_start_lr=optim_params["warmup_start_lr"],
                eta_min=optim_params["eta_min"],
            )

            return optimizer, scheduler

        optimizer1, scheduler1 = get_optimizer(
            **self.params["optim1"], max_epochs=self.params["max_epochs"]
        )
        optims.append(optimizer1)
        scheds.append(scheduler1)

        if "optim2" in self.params and self.params["optim2"] is not None:
            optimizer2, scheduler2 = get_optimizer(
                **self.params["optim2"], max_epochs=self.params["max_epochs"]
            )
            optims.append(optimizer2)
            scheds.append(scheduler2)

        return optims, scheds
