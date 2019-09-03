from typing import List, Tuple

import torch
import torch.nn.functional as fn
import torchvision
from torchvision import transforms

from src.model import Model
from src.model.model_evaluator import ModelEvaluator
from src.model.model_generator import ModelGenerator
from src.optimize.simulate_annealing import SimulateAnnealing
from src.optimize.solution import Solution


def loss_function(_input: torch.Tensor, target: torch.Tensor) -> float:
    loss = fn.cross_entropy(torch.softmax(_input.unsqueeze(0), 1), target.to(torch.long)).pow(2).sum().sqrt().item()
    return loss


def activation(_output: torch.Tensor) -> torch.Tensor:
    return torch.relu(_output)


def get_dataset(train: bool) -> Tuple[List[Tuple[List[torch.Tensor], torch.Tensor]], int, int]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST("./data", train, transform)
    loader: List[Tuple[torch.Tensor, torch.Tensor]] = torch.utils.data.DataLoader(dataset)
    data = []
    for _input, label in loader:
        data.append(([_input.flatten()], label))
        if len(data) > 1000:
            break

    input_dim = data[0][0][0].size()[0]
    output_dim = len(dataset.classes)
    return data, input_dim, output_dim


def main(argv):
    print('Getting dataset')
    dataset, input_dim, output_dim = get_dataset(True)
    hidden_dim = 1000
    delay = 30
    print('Setup model')
    model = Model.create_randomized_model(
        activation,
        input_dim, output_dim, hidden_dim,
        delay=delay,
        device='cuda'
    )

    torch.cuda.init()
    optimizer = SimulateAnnealing(number_of_new_solution_generated=5)
    model_generator = ModelGenerator(distance=1)
    model_evaluator = ModelEvaluator(loss_function, dataset)
    initial_solution = Solution(model, model_generator, model_evaluator)

    optimizer.set_max_iteration(10000)
    optimizer.set_temperature_reducer(0.999)
    optimizer.set_initial_temperature(10000)
    optimizer.set_initial_solution(initial_solution)
    print('Start optimize procedure...')
    optimizer.optimize()

    best_model: Model = optimizer.get_best_solution().getvalue()

    print(best_model.to_json())
