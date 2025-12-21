from abc import abstractmethod
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def device(self):
        return self._device_anchor.device

    def encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def sample(
        self, latent_size: Union[int, Tuple[int, int], Sequence[int]], batch_size: int
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def interpolate(
        self,
        encoded_a: Dict[str, Tensor],
        encoded_b: Dict[str, Tensor],
        steps: int = 5,
        batch_size: int = 1,
    ) -> List[Dict[str, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def loss_function(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass
