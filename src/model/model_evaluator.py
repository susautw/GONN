from typing import Callable, List, Tuple

import math
import torch

from src.model import Model
from src.optimize.optimizer import Solution
from src.optimize.solution import SolutionEvaluator


class ModelEvaluator(SolutionEvaluator):

    loss_function: Callable[[torch.Tensor, torch.Tensor], float]
    dataset: List[Tuple[List[torch.Tensor], torch.Tensor]]

    def __init__(
            self,
            loss_function: Callable[[torch.Tensor, torch.Tensor], float],
            dataset: List[Tuple[List[torch.Tensor], torch.Tensor]]
    ):
        self.loss_function = loss_function
        self.dataset = dataset

    def evaluate(self, solution: Solution[Model]) -> float:
        model = solution.getvalue()

        loss: float = 0
        for data in self.dataset:
            _input = data[0]
            target = data[1]

            model.clear_output()
            model.predict(_input)
            output = model.get_output()
            output_loss = self.loss_function(output, target.to(model.device, torch.float64))
            if not math.isnan(output_loss):
                loss = loss + output_loss
            else:
                return float('inf')

        return loss / len(self.dataset)
