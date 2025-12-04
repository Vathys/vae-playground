import torch.nn as nn

from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, data: dict) -> dict:
        raise NotImplementedError

    def decode(self, data: dict) -> dict:
        raise NotImplementedError

    def sample_test(self, num: int, inter: int = 5, batch_size: int = 1):
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: dict) -> dict:
        pass

    @abstractmethod
    def loss_function(self, data: dict) -> dict:
        pass
