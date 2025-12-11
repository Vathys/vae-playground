from abc import abstractmethod
from typing import Dict, List, Sequence, Tuple, Union

import torch.nn as nn
from torch import Tensor


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def sample_test(
        self,
        latent_size: Union[int, Tuple[int, int], Sequence[int]],
        num: int,
        inter: int = 5,
        batch_size: int = 1,
    ) -> List[Dict[str, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def loss_function(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass
