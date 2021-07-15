import sympy as sp

from probably.analysis.exceptions import DistributionParameterError
from probably.analysis.generating_function import GeneratingFunction


class PGFS:
    """Implements PGFs of standard distributions."""
    @classmethod
    def geometric(cls, var, p: str):
        if not (0 < sp.S(p) < 1):
            raise DistributionParameterError(f"parameter of geom distr must be >0 and <=1, was {p}")
        return GeneratingFunction(f"{p} / (1 - (1-{p}) * {var})")

    @classmethod
    def uniform(cls, var, a: str, b: str):
        if not (0 <= sp.S(a) < sp.S(b)):
            raise DistributionParameterError(f"Distribution parameters must satisfy 0 <= a < b < oo")
        return GeneratingFunction(f"1/({b - a + 1}) * {var}**{a} * ({var}**({b - a + 1}) - 1) / ({var} - 1)")

    @classmethod
    def bernoulli(cls, var, p: str):
        if not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Bernoulli Distribution must be in [0,1], but was {p}")
        return GeneratingFunction(f"1 - {p} + {p} * {var}")

    @classmethod
    def poisson(cls, var, l: str):
        if sp.S(l) < 0:
            raise DistributionParameterError(f"Parameter of Poisson Distribution must be in [0, oo), but was {p}")
        return GeneratingFunction(f"exp({l} * ({var} - 1))")

    @classmethod
    def log(cls, var, p: str):
        if not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Logarithmic Distribution must be in [0,1], but was {p}")
        return GeneratingFunction(f"log(1-{p}*{var})/log(1-{p})")

    @classmethod
    def binomial(cls, var, n: str, p: str):
        if not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Binomial Distribution must be in [0,1], but was {p}")
        if not (0 <= sp.S(n)):
            raise DistributionParameterError(f"Parameter of Binomial Distribution must be in [0,oo), but was {n}")
        return GeneratingFunction(f"(1-{p}+{p}*{var})**{n}")