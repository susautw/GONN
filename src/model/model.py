import json

import torch
from typing import Callable, List

from src.exceptions.model import WrongDimensionException


class Model:
    # connection way
    connection: torch.Tensor

    # weight
    weight: torch.Tensor

    # completed model
    model: torch.Tensor

    # store output
    output: torch.Tensor

    # activation function
    activation: Callable[[torch.Tensor], torch.Tensor]

    # dimension of model
    input_dim: int
    output_dim: int
    hidden_dim: int
    number_of_neural: int

    # reaction time given
    delay: int

    device: torch.device

    @staticmethod
    def create_randomized_model(
            activation: Callable[[torch.Tensor], torch.Tensor],
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            delay: int = 100,
            device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    ) -> 'model':
        number_of_neural: int = input_dim + output_dim + hidden_dim
        size: List[int] = [number_of_neural, number_of_neural]

        connection = torch.ones(size) * 0.2  # invert
        connection = torch.bernoulli(connection)

        weight = torch.zeros(
            size
        ).uniform_(-1, 1)
        return Model(
            activation,
            input_dim,
            output_dim,
            hidden_dim,
            connection,
            weight,
            delay,
            device
        )

    def __init__(
            self,
            activation: Callable[[torch.Tensor], torch.Tensor],
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            connection: torch.Tensor,
            weight: torch.Tensor,
            delay: int = 100,
            device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    ):
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.number_of_neural = input_dim + output_dim + hidden_dim

        self.delay = delay

        valid_size: List[int] = [self.number_of_neural, self.number_of_neural]
        if not self._tensor_size_valid(connection, valid_size):
            raise WrongDimensionException(connection.size(), valid_size)
        if not self._tensor_size_valid(weight, valid_size):
            raise WrongDimensionException(connection.size(), valid_size)

        self.device = device
        self._model_initialization(connection, weight)

    def _model_initialization(
            self,
            connection: torch.Tensor,
            weight: torch.Tensor
    ) -> None:
        number_of_neural: int = self.number_of_neural
        self.connection = self._convert_tensor(connection)
        # remove input neural interconnection ways
        self.connection[: self.input_dim, : self.input_dim] = 0
        self.connection[:self.input_dim, :] = 0

        # remove outcome way
        self.connection[:, self.input_dim: self.input_dim + self.output_dim] = 0

        self.weight = self._convert_tensor(weight)
        self.model = self.connection * self.weight
        self.model = self.model.to_sparse()

        self.output = torch.zeros(
            [number_of_neural, 1],
            device=self.device,
            dtype=torch.float64
        )

    def _convert_tensor(self, tensor: torch.Tensor):
        return tensor.to(self.device, torch.float64)

    def tick(self, _input: torch.Tensor) -> None:
        if not self._tensor_size_valid(_input, [self.input_dim]):
            raise WrongDimensionException(_input.size(), [self.input_dim])

        extend_input: torch.Tensor = torch.zeros(
            [self.output_dim + self.hidden_dim],
            device=self.device,
            dtype=torch.float64
        )

        _input = torch.cat([_input.to(self.device, torch.float64), extend_input]).unsqueeze(1)

        self.output = self.output + _input
        self.output = self.model.mm(self.output)
        self.output = self.activation(self.output)

    def get_output(self) -> torch.Tensor:
        return self.output[self.input_dim: self.input_dim + self.output_dim].flatten()

    @staticmethod
    def _tensor_size_valid(tensor: torch.Tensor, shape: List[int]) -> bool:
        size: torch.Size = torch.Size(shape)
        return tensor.size() == size

    def predict(self, data: List[torch.Tensor]) -> None:
        for _input in data:
            self.tick(_input)
        # do delay with last input
        last_input: torch.Tensor = data[len(data) - 1]
        for i in range(self.delay):
            self.tick(last_input)

    def clear_output(self) -> None:
        self.output[:] = 0

    def to_json(self):
        dumps = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'delay': self.delay,
            'connection': self.connection.tolist(),
            'weight': self.weight.tolist()
        }
        return json.dumps(dumps)
