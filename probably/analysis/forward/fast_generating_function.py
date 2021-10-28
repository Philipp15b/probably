from typing import Union, Generator, Set, Iterator, Tuple, Dict

from . import MarginalType
from .distribution import Distribution, CommonDistributionsFactory, Param
import prodigy

from ...pgcl import VarExpr, Expr


class FPSFactory(CommonDistributionsFactory):

    @staticmethod
    def geometric(var: Union[str, VarExpr], p: Param) -> Distribution:
        return prodigy.geometric(var, p)

    @staticmethod
    def uniform(var: Union[str, VarExpr], a: Param, b: Param) -> Distribution:
        raise NotImplementedError(__name__)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: Param) -> Distribution:
        raise NotImplementedError(__name__)

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: Param) -> Distribution:
        raise NotImplementedError(__name__)

    @staticmethod
    def log(var: Union[str, VarExpr], p: Param) -> Distribution:
        raise NotImplementedError(__name__)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: Param, p: Param) -> Distribution:
        raise NotImplementedError(__name__)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        raise NotImplementedError(__name__)


class FPS(Distribution):

    def __init__(self):
        raise NotImplementedError(__name__)

    def __add__(self, other):
        raise NotImplementedError(__name__)

    def __sub__(self, other):
        raise NotImplementedError(__name__)

    def __mul__(self, other):
        return self.dist * other

    def __truediv__(self, other):
        raise NotImplementedError(__name__)

    def __eq__(self, other):
        raise NotImplementedError(__name__)

    def __str__(self):
        raise NotImplementedError(__name__)

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, int]]]:
        raise NotImplementedError(__name__)

    def copy(self, deep: bool = True) -> 'Distribution':
        raise NotImplementedError(__name__)

    def get_probability_of(self, condition: Union[Expr, str]):
        raise NotImplementedError(__name__)

    def get_probability_mass(self) -> Union[Expr, str]:
        raise NotImplementedError(__name__)

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        return self.dist.E(expression)

    def normalize(self) -> 'Distribution':
        raise NotImplementedError(__name__)

    def get_variables(self) -> Set[str]:
        raise NotImplementedError(__name__)

    def get_parameters(self) -> Set[str]:
        raise NotImplementedError(__name__)

    def filter(self, condition: Union[Expr, str]) -> 'Distribution':
        raise NotImplementedError(__name__)

    def is_zero_dist(self) -> bool:
        raise NotImplementedError(__name__)

    def is_finite(self) -> bool:
        raise NotImplementedError(__name__)

    def update(self, expression: Expr) -> 'Distribution':
        raise NotImplementedError(__name__)

    def marginal(self, *variables: Union[str, VarExpr], method: MarginalType = MarginalType.Include) -> 'Distribution':
        raise NotImplementedError(__name__)

    def set_variables(self, *variables: str) -> 'Distribution':
        raise NotImplementedError(__name__)

    def approximate(self, threshold: Union[str, int]) -> Generator['Distribution', None, None]:
        raise NotImplementedError(__name__)
