import unittest

import torch

from src.model.model_generator import ModelGenerator


class TestModelGenerator(unittest.TestCase):

    def test_create_instance(self) :
        model_generator: ModelGenerator = ModelGenerator()

    def test_remove_neural(self):
        tensor = torch.arange(25).reshape(5, 5)
        after_remove = torch.tensor(
            [[0, 1, 3, 4],
             [5, 6, 8, 9],
             [15, 16, 18, 19],
             [20, 21, 23, 24]]
        )

        removed_tensor = ModelGenerator._remove_neural(tensor, 2)
        self.assertTrue(removed_tensor.equal(after_remove))

    def test_add_neural(self):
        tensor = torch.empty(5, 5,dtype=torch.float64)
        model_generator = ModelGenerator()
        model_generator.device = torch.device('cpu')
        tensors = model_generator._add_neural(tensor, tensor)
        size = torch.Size([6, 6])

        self.assertEqual(tensors[0].size(), size)
        self.assertEqual(tensors[1].size(), size)
