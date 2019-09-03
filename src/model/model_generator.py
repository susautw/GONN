import random
import torch
from typing import List
from src.model import Model
from src.optimize.solution import SolutionGenerator, Solution


class ModelGenerator(SolutionGenerator):
    distance: float
    possibility_of_invert: float
    possibility_of_regulate: float
    device: torch.device

    def __init__(
            self,
            distance: float = 1,
            possibility_of_invert=0.2,
            possibility_of_regulate=0.1
    ):
        self.distance = distance
        self.possibility_of_invert = possibility_of_invert
        self.possibility_of_regulate = possibility_of_regulate

    def generate(self, solution: Solution[Model]) -> Solution[Model]:
        model = solution.getvalue()
        connection: torch.Tensor = model.connection
        weight: torch.Tensor = model.weight
        self.device = model.device

        connection, weight = self._regulate_neural(connection, weight, removable=model.input_dim + model.output_dim)
        flip_mask = torch.ones_like(connection) * self.possibility_of_invert
        flip_mask = torch.bernoulli(flip_mask).to(torch.int8)
        new_connection = connection.to(torch.int8).__xor__(flip_mask)

        uniform_from_distance = torch.empty_like(weight).uniform_(-1 * self.distance, self.distance)
        new_weight = weight + uniform_from_distance

        new_model = Model(
            model.activation,
            model.input_dim, model.output_dim, connection.size()[0] - model.input_dim - model.output_dim,
            new_connection,
            new_weight,
            model.delay,
            model.device
        )

        return Solution(
            new_model,
            solution.get_generator(),
            solution.get_evaluator()
        )

    @staticmethod
    def _to_cpu_tensor(tensor: torch) -> torch.Tensor:
        return tensor.to(torch.device('cpu'))

    def _regulate_neural(
            self,
            connection: torch.Tensor,
            weight: torch.Tensor,
            removable: int
    ) -> List[torch.Tensor]:
        max_removable = connection.size()[0] - 1
        if random.random() < self.possibility_of_regulate:
            if random.random() < 0.5:
                return self._add_neural(connection, weight)
            elif max_removable > removable:
                remove_n = random.randint(removable, max_removable)
                connection = self._remove_neural(connection, remove_n)
                weight = self._remove_neural(weight, remove_n)

                return [connection, weight]
        return [connection, weight]

    @staticmethod
    def _remove_neural(tensor: torch.Tensor, n: int) -> torch.Tensor:
        tensor = torch.cat([tensor[:n, :], tensor[n + 1:, :]])
        tensor = torch.cat([tensor[:, :n], tensor[:, n + 1:]], 1)
        return tensor

    def _add_neural(
            self,
            connection: torch.Tensor,
            weight: torch.Tensor
    ) -> List[torch.Tensor]:
        model_size: int = connection.size()[0]
        bottom_size = [1, model_size]
        left_size = [model_size + 1, 1]

        connection = torch.cat(
            [
                connection,
                torch.randint(0, 2, bottom_size, dtype=torch.float64, device=self.device)
            ]
        )

        connection = torch.cat(
            [
                connection,
                torch.randint(0, 2, left_size, dtype=torch.float64, device=self.device)
            ],
            1
        )

        weight = torch.cat(
            [weight,
             torch.empty(bottom_size, dtype=torch.float64, device=self.device).uniform_(-1, 1)
             ]
        )
        weight = torch.cat(
            [weight,
             torch.empty(left_size, dtype=torch.float64, device=self.device).uniform_(-1, 1)
             ],
            1
        )

        return [connection, weight]
