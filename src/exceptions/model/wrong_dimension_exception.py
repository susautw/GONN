from typing import List
import torch


class WrongDimensionException(Exception):
    def __init__(self,actual: torch.Size,  excepted_dim: List[int]):
        super().__init__("Except dim(%s) ,but got dim(%s)" % (torch.Size(excepted_dim), actual))
