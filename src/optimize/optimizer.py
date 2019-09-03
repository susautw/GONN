from abc import abstractmethod, ABCMeta
from typing import List, TypeVar, Generic
import src.optimize.solution

Solution = src.optimize.solution.Solution

T = TypeVar('T')


class Optimizer(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def optimize(self) -> None:
        pass

    @abstractmethod
    def get_solutions(self) -> List[Solution[T]]:
        pass

    @abstractmethod
    def get_best_solution(self) -> Solution[T]:
        pass
