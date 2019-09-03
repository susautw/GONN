import unittest
import torch
from typing import List
from src.exceptions.model import WrongDimensionException
from src.model import Model


# fake activation function
def square(_input: torch.Tensor) -> torch.Tensor:
    return _input.pow(2)


class TestModel(unittest.TestCase):

    def test_create_instance_and_initializing(self) -> None:
        instance: Model = Model(
            square,
            10, 10, 10,
            torch.zeros([30, 30]),
            torch.zeros([30, 30]),
            device=torch.device('cpu')
        )

        size: torch.Size = torch.Size([30, 30])
        output_size: torch.Size = torch.Size([30, 1])
        self.assertEqual(instance.connection.size(), size)
        self.assertEqual(instance.weight.size(), size)
        self.assertEqual(instance.output.size(), output_size)

    def test_tensor_size_valid(self) -> None:
        t1: torch.Tensor = torch.tensor([1, 2])

        same_size: bool = Model._tensor_size_valid(t1, [2])
        diff_size: bool = Model._tensor_size_valid(t1, [3])

        self.assertTrue(same_size)
        self.assertFalse(diff_size)

    def test_get_output(self) -> None:
        instance: Model = Model(
            square,
            1, 5, 10,
            torch.zeros([16, 16]),
            torch.zeros([16, 16]),
            device=torch.device('cpu')
        )

        instance.output = torch.arange(
            instance.number_of_neural,
            dtype=torch.float64
        ).reshape_as(instance.output)

        output_size: torch.Size = torch.Size([5])
        output_tensor: torch.Tensor = torch.tensor(
            [1, 2, 3, 4, 5],
            dtype=torch.float64
        )

        self.assertEqual(instance.get_output().size(), output_size)
        self.assertTrue(instance.get_output().equal(output_tensor))

    def test_tick_calculation(self) -> None:
        instance: Model = Model(
            square,
            1, 1, 1,
            torch.tensor(
                [[0, 0, 0],
                 [1, 0, 1],
                 [1, 0, 0]],
                dtype=torch.float64
            ),
            torch.tensor(
                [[0.5, 0.5, 0],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0]],
                dtype=torch.float64
            ),
            device=torch.device('cpu')
        )

        _input: torch.Tensor = torch.tensor(
            [1],
            dtype=torch.float64
        )

        first_output: torch.Tensor = torch.tensor(
            [0.25],
            dtype=torch.float64
        )
        second_output: torch.Tensor = torch.tensor(
            [0.390625],
            dtype=torch.float64
        )

        instance.tick(_input)
        self.assertTrue(instance.get_output().equal(first_output))

        instance.tick(_input)
        self.assertTrue(instance.get_output().equal(second_output))

    def test_tick_input_with_wrong_dim(self) -> None:
        instance: Model = Model(
            square,
            1, 5, 10,
            torch.zeros([16, 16]),
            torch.zeros([16, 16]),
            device=torch.device('cpu')
        )

        wrong_input: torch.Tensor = torch.empty(
            [2],
            dtype=torch.float64
        )

        with self.assertRaises(WrongDimensionException):
            instance.tick(wrong_input)

    def test_predict(self) -> None:
        instance: Model = Model(
            square,
            1, 1, 1,
            torch.tensor(
                [[0, 0, 0],
                 [1, 0, 1],
                 [1, 0, 0]],
                dtype=torch.float64
            ),
            torch.tensor(
                [[0.5, 0.5, 0],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0]],
                dtype=torch.float64
            ),
            delay=100,
            device=torch.device('cpu')
        )

        data: List[torch.Tensor] = [torch.tensor(
            [1],
            dtype=torch.float64
        )]

        instance.predict(data)
        output: torch.Tensor = torch.tensor(
            [0.390625],
            dtype=torch.float64
        )
        self.assertTrue(instance.get_output().equal(output))
