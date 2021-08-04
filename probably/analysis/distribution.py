from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Union, Set

from probably.pgcl import Expr, VarExpr


class MarginalType(Enum):
    """ Specifies the type of marginalization. """
    Include = auto()
    Exclude = auto()


class CommonDistributionsFactory(ABC):
    """ Abstract Factory Class implementing a Factory for common distributions."""

    @staticmethod
    @abstractmethod
    def geometric(var: Union[str, VarExpr], p: str) -> 'Distribution':
        """ A geometric distribution with parameter `p`."""
        pass

    @staticmethod
    @abstractmethod
    def uniform(var: Union[str, VarExpr], a: str, b: str) -> 'Distribution':
        """ A uniform distribution with bounds [`a`,`b`]."""
        pass

    @staticmethod
    @abstractmethod
    def bernoulli(var: Union[str, VarExpr], p: str) -> 'Distribution':
        """ A bernoulli distribution with parameter `p`."""
        pass

    @staticmethod
    @abstractmethod
    def poisson(var: Union[str, VarExpr], lam: str) -> 'Distribution':
        """ A poisson distribution with parameter `lamb`da."""
        pass

    @staticmethod
    @abstractmethod
    def log(var: Union[str, VarExpr], p: str) -> 'Distribution':
        """ A logarithmic distribution with parameter `p`."""
        pass

    @staticmethod
    @abstractmethod
    def binomial(var: Union[str, VarExpr], n: str, p: str) -> 'Distribution':
        """ A binomial distribution with parameters `n` and `p`."""
        pass

    @staticmethod
    @abstractmethod
    def undefined(*variables: Union[str, VarExpr]) -> 'Distribution':
        """ A distribution where actually no information about the states is given."""
        pass


class Distribution(ABC):
    """ Abstract class that models different representations of probability distributions. """

    @abstractmethod
    def __add__(self, other):
        """ The addition of two distributions. """
        pass

    @abstractmethod
    def __sub__(self, other):
        """ The subtraction of two distributions. """
        pass

    @abstractmethod
    def __mul__(self, other):
        """ The multiplication of two distributions. """
        pass

    @abstractmethod
    def __truediv__(self, other):
        """ The division of two distributions. """
        pass

    @abstractmethod
    def __str__(self):
        """ The string representation of a distribution. """
        pass

    @abstractmethod
    def __iter__(self):
        """ Returns an iterator that iterates over the state, probability pairs of the distribution."""
        pass

    @abstractmethod
    def copy(self, deep: bool = True) -> 'Distribution':
        """ Returns a full copy of itself."""
        pass

    @abstractmethod
    def get_probability_of(self, condition: Union[Expr, str]):
        """
        Returns the probability of a given `condition` or variable.
        :param condition: The condition.
        :return: The probability that the condition is satisfied.
        """
        pass

    @abstractmethod
    def get_probability_mass(self):
        """ Returns the probability mass of the distribution. """
        pass

    @abstractmethod
    def get_expected_value_of(self, expression: Union[Expr, str]):
        """ Returns the expected value of the expression `expression` evaluated in the distribution. """
        pass

    @abstractmethod
    def normalize(self) -> 'Distribution':
        """ Normalizes the probability mass of the distribution."""
        pass

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Returns the free variables of the distribution. """
        pass

    @abstractmethod
    def get_parameters(self) -> Set[str]:
        """ Returns the parameters of the distribution. """
        pass

    @abstractmethod
    def filter(self, condition: Union[Expr, str]) -> 'Distribution':
        """ Filters the distribution such that only the parts which satisfy the `condition` are left."""
        pass

    @abstractmethod
    def is_zero_dist(self) -> bool:
        """ Returns whether the distribution encodes the 0 distribution."""
        pass

    @abstractmethod
    def is_finite(self) -> bool:
        """ Returns whether the distribution has finite support."""
        pass

    @abstractmethod
    def update(self, expression: Expr) -> 'Distribution':
        """ Updates the distribution by the result of the expression. """
        pass

    @abstractmethod
    def marginal(self, *variables: Union[str, VarExpr], method: MarginalType = MarginalType.Include) -> 'Distribution':
        """ Computes the marginal distribution for the given variables (MarginalType.Include),
            or for all but the given variables (MarginalType.Exclude).
        """
        pass
