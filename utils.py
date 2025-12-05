import torch

import logging
import time

from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT

logger = logging.getLogger(__name__)


class LogPerformanceCallback(Callback):

    def __init__(self):
        super().__init__()

        self.start_time = 0.0
        self.last_batch_end_time = 0.0
        self.update_count = 0.0
        self.backward_start_time = 0.0
        self.forward_start_time = 0.0
        self.between_step_time = 0.0

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        pl_module.log(
            "performance/between_step_time",
            time.time() - self.between_step_time,
            on_step=True,
            on_epoch=False,
        )
        self.forward_start_time = time.time()

    @rank_zero_only
    def on_before_backward(self, trainer, pl_module, loss):
        super().on_before_backward(trainer, pl_module, loss)
        forward_time = time.time() - self.forward_start_time
        pl_module.log(
            "performance/forward_time", forward_time, on_step=True, on_epoch=False
        )
        self.backward_start_time = time.time()

    @rank_zero_only
    def on_after_backward(self, trainer, pl_module):
        super().on_after_backward(trainer, pl_module)

        backward_time = time.time() - self.backward_start_time
        pl_module.log(
            "performance/backward_time", backward_time, on_step=True, on_epoch=False
        )

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.update_count = 0.0
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        self.update_count += 1

        total_elapsed_time = time.time() - self.start_time
        last_elapsed_time = time.time() - self.last_batch_end_time
        self.last_batch_end_time = time.time()

        average_updates_per_second = self.update_count / total_elapsed_time
        last_updates_per_second = 1 / last_elapsed_time

        pl_module.log(
            "performance/average_updates_per_second",
            average_updates_per_second,
            on_step=True,
            on_epoch=False,
        )

        pl_module.log(
            "performance/last_updates_per_second",
            last_updates_per_second,
            on_step=True,
            on_epoch=False,
        )
        self.between_step_time = time.time()


def create_checkerboard_mask(h, w, invert=False):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask


def create_channel_mask(c_in, invert=False):
    mask = torch.cat(
        [
            torch.ones(c_in // 2, dtype=torch.float32),
            torch.zeros(c_in - c_in // 2, dtype=torch.float32),
        ]
    )
    mask = mask.view(1, c_in, 1, 1)
    if invert:
        mask = 1 - mask
    return mask


def lerp_z(z1: torch.Tensor, z2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the lerp between two tensors z1, z2 with time values t.

    :param z1: tensor of size n
    :type z1: torch.Tensor
    :param z2: tensor of size n
    :type z2: torch.Tensor
    :param t: tensor of size t
    :type t: torch.Tensor
    :return: tensor of size [t, n]
    :rtype: torch.Tensor
    """
    return (1 - t[:, None]) * z1[None, :] + t[:, None] * z2[None, :]


def slerp_z(z1: torch.Tensor, z2: torch.Tensor, t: torch.Tensor):
    """
    Calculate the spherical lerp between two tensors z1, z2 with time values t.

    :param z1: tensor of size n
    :type z1: torch.Tensor
    :param z2: tensor of size n
    :type z2: torch.Tensor
    :param t: tensor of size t
    :type t: torch.Tensor
    :return: tensor of size [t, n]
    :rtype: torch.Tensor
    """
    z1 = z1 / z1.norm()
    z2 = z2 / z2.norm()

    dot = (z1 * z2).sum().clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    t1 = (torch.sin((1 - t[:, None]) * theta) / sin_theta) * z1[None, :]
    t2 = (torch.sin(t[:, None] * theta) / sin_theta) * z2[None, :]

    return t1 + t2
