from typing import Union

import sympy
import sympy as sp

from probably.analysis.distribution import CommonDistributionsFactory
from probably.analysis.exceptions import DistributionParameterError
from probably.analysis.generating_function import GeneratingFunction
from probably.pgcl import VarExpr


class PGFS(CommonDistributionsFactory):
    """Implements PGFs of standard distributions."""

    @staticmethod
    def geometric(var: Union[str, VarExpr], p: str) -> 'Distribution':
        if not (0 < sp.S(p) < 1):
            raise DistributionParameterError(f"parameter of geom distr must be >0 and <=1, was {p}")
        return GeneratingFunction(f"{p} / (1 - (1-{p}) * {var})", closed=True, finite=False)

    @staticmethod
    def uniform(var: Union[str, VarExpr], a: str, b: str) -> 'Distribution':
        if not (0 <= sp.S(a) < sp.S(b)):
            raise DistributionParameterError(f"Distribution parameters must satisfy 0 <= a < b < oo")
        return GeneratingFunction(f"1/({b} - {a} + 1) * {var}**{a} * ({var}**({b} - {a} + 1) - 1) / ({var} - 1)",
                                  closed=True, finite=True)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: str) -> 'Distribution':
        if not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Bernoulli Distribution must be in [0,1], but was {p}")
        return GeneratingFunction(f"1 - {p} + {p} * {var}", closed=True, finite=True)

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: str) -> 'Distribution':
        if sp.S(lam) < 0:
            raise DistributionParameterError(f"Parameter of Poisson Distribution must be in [0, oo), but was {l}")
        return GeneratingFunction(f"exp({lam} * ({var} - 1))", closed=True, finite=False)

    @staticmethod
    def log(var: Union[str, VarExpr], p: str) -> 'Distribution':
        if not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Logarithmic Distribution must be in [0,1], but was {p}")
        return GeneratingFunction(f"log(1-{p}*{var})/log(1-{p})", closed=True, finite=False)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: str, p: str) -> 'Distribution':
        if not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Binomial Distribution must be in [0,1], but was {p}")
        if not (0 <= sp.S(n)):
            raise DistributionParameterError(f"Parameter of Binomial Distribution must be in [0,oo), but was {n}")
        return GeneratingFunction(f"(1-{p}+{p}*{var})**{n}", closed=True, finite=True)

    @staticmethod
    def zero(*variables: Union[str, sympy.Symbol]):
        if variables:
            return GeneratingFunction("0", variables, preciseness=1, closed=True, finite=True)
        return GeneratingFunction("0", preciseness=1, closed=True, finite=True)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> 'Distribution':
        """ A distribution where actually no information about the states is given."""
        return PGFS.zero(*map(str, variables))
