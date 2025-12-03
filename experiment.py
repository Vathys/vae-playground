import torch
from torch import Tensor
from torch import optim
from models import BaseVAE
import lightning as L
import torchvision.utils as vutils
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class VAExperiment(L.LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super().__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None

        self.test_input = None
        self.test_latents = None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(
            *results,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        self.log_dict(
            {f"train/{key}": val.item() for key, val in train_loss.items()},
            sync_dist=True,
        )

        return train_loss["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.log_grads()

    def validation_step(self, batch, batch_idx):
        real_img = batch
        self.curr_device = real_img.device

        if self.test_input is None or self.test_latents is None:
            self.initialize_image_inputs(batch)

        results = self.forward(real_img)
        val_loss = self.model.loss_function(
            *results,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        self.log_dict(
            {f"val/{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

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

        self.logger.experiment.add_text("grad/summary", grad_txt, self.global_step)

    def initialize_image_inputs(self, batch):
        if self.test_input is None:
            self.test_input = batch[:25].to(self.curr_device)

        if self.test_latents is None:
            z = self.model.sample_latent(25)

            self.test_latents = z

    def sample_images(self):
        recons = self.model.forward(self.test_input)[0]
        grid = vutils.make_grid(recons.data, nrow=5)
        self.logger.experiment.add_image("reconstruction", grid, self.global_step)

        try:
            samples = self.model.decode(self.test_latents)
            grid = vutils.make_grid(samples.data, nrow=5)
            self.logger.experiment.add_image("samples", grid, self.global_step)
        except Warning:
            pass

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
