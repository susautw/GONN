from abc import abstractmethod, ABCMeta
from typing import TypeVar, Generic, Optional

T = TypeVar("T")


class SolutionEvaluator(Generic[T], metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, solution: 'Solution[T]') -> float:
        pass


class SolutionGenerator(Generic[T], metaclass=ABCMeta):

    @abstractmethod
    def generate(self, solution: 'Solution[T]') -> 'Solution[T]':
        pass


class Solution(Generic[T]):
    generator: SolutionGenerator
    evaluator: SolutionEvaluator
    evaluation: Optional[float]
    value: T

    def __init__(
            self,
            value: T,
            generator: SolutionGenerator,
            evaluator: SolutionEvaluator
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.evaluation = None
        self.value = value

    def getvalue(self) -> T:
        return self.value

    def generate(self) -> 'Solution[T]':
        return self.generator.generate(self)

    def evaluate(self) -> float:
        if self.evaluation is None:
            self.evaluation = self.evaluator.evaluate(self)
        return self.evaluation

    def get_generator(self) -> SolutionGenerator[T]:
        return self.generator

    def get_evaluator(self) -> SolutionEvaluator[T]:
        return self.evaluator

    def __str__(self) -> str:
        return str(self.value)
