from typing import TypeVar, List, Optional
from random import random
from math import exp
from src.optimize.optimizer import Optimizer
from src.optimize.solution import Solution

T = TypeVar('T')


class SimulateAnnealing(Optimizer[T]):
    solutions: List[Solution[T]]
    best_solution: Optional[Solution[T]]
    initial_solution: Solution[T]
    max_iteration: int
    initial_temperature: float
    temperature_reducer: float
    number_of_new_solution_generated: int

    def __init__(
            self,
            max_iteration: int = 10000,
            initial_temperature: float = 6000,
            temperature_reducer: float = 0.7,
            number_of_new_solution_generated=100
    ):
        self.solutions = []
        self.best_solution = None
        self.max_iteration = max_iteration
        self.initial_temperature = initial_temperature
        self.temperature_reducer = temperature_reducer
        self.number_of_new_solution_generated = number_of_new_solution_generated

    def optimize(self):
        solutions: List[Solution[T]] = [self.initial_solution]
        temperatures: List[float] = [self.initial_temperature]
        best_solution: Solution[T] = solutions[0]

        for i in range(0, self.max_iteration):
            temperatures.append(self.new_temperature(temperatures[i]))
            new_solutions: List[Solution[T]] = self._generate_new_solutions(solutions[i])
            best_solution_in_new_solutions: Solution[T] = self._find_best_solution(new_solutions)
            print("Best solution in new solutions, loss=%.2f" % best_solution_in_new_solutions.evaluate())

            if self._should_accept_new_solution(
                    temperatures[i + 1],
                    solutions[i],
                    best_solution_in_new_solutions):
                solutions.append(best_solution_in_new_solutions)
            else:
                solutions.append(solutions[i])

            if self._is_better_solution(best_solution, solutions[i + 1]):
                best_solution = solutions[i + 1]

            print("Iteration %5d , loss=%.2f, best_loss=%.2f" % (
                i + 1,
                solutions[i + 1].evaluate(),
                best_solution.evaluate()
            ))

        self.best_solution = best_solution
        self.solutions = solutions

    def new_temperature(self, temperature) -> float:
        return temperature * self.temperature_reducer

    def _generate_new_solutions(self, solution: Solution[T]) -> List[Solution[T]]:
        new_solutions = []
        for i in range(0, self.number_of_new_solution_generated):
            new_solutions.append(solution.generate())
        return new_solutions

    def _find_best_solution(self, solutions: List[Solution[T]]) -> Solution[T]:
        best: Solution[T] = solutions[0]
        for solution in solutions:
            if self._is_better_solution(best, solution):
                best = solution
            print("Evaluating, loss=%f" % solution.evaluate())
        return best

    @staticmethod
    def _is_better_solution(old_solution: Solution[T], new_solution: Solution[T]) -> bool:
        return new_solution.evaluate() < old_solution.evaluate()

    @staticmethod
    def _should_accept_new_solution(
            temperature: float,
            current_solution: Solution[T],
            new_solution: Solution[T]
    ) -> bool:
        return SimulateAnnealing._is_better_solution(current_solution, new_solution) \
               or random() < exp(-1 * (new_solution.evaluate() - current_solution.evaluate()) / temperature)

    def get_solutions(self) -> List[Solution[T]]:
        return self.solutions

    def get_best_solution(self) -> Solution[T]:
        if self.best_solution is None:
            raise Exception("not been optimized")
        return self.best_solution

    def set_max_iteration(self, max_iteration: int) -> None:
        self.max_iteration = max_iteration

    def set_initial_solution(self, initial_solution: Solution[T]) -> None:
        self.initial_solution = initial_solution

    def set_initial_temperature(self, initial_temperature: float) -> None:
        self.initial_temperature = initial_temperature

    def set_temperature_reducer(self, temperature_reducer: float) -> None:
        self.temperature_reducer = temperature_reducer
