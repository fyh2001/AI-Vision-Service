import torch
from abc import ABC, abstractmethod
from typing import TypeVar

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class BaseModel(ABC):
    def __init__(self):
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @abstractmethod
    async def infer(self, input: InputType) -> OutputType:
        pass
