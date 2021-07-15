import sympy as sp

from probably.analysis.exceptions import DistributionParameterError
from probably.analysis.generating_function import GeneratingFunction
from probably.pgcl import RealLitExpr


class PGFS:
    """Implements PGFs of standard distributions."""
    @classmethod
    def geometric(cls, var, p: RealLitExpr):
        if p.to_fraction() <= 0 or p.to_fraction() > 1:
            raise DistributionParameterError(f"parameter of geom distr must be >0 and <=1, was {p}")
        return GeneratingFunction(f"{p} / (1 - (1-{p}) * {var})")

    @classmethod
    def uniform(cls, var, a: int, b: int):
        if not (0 <= a < b):
            raise DistributionParameterError(f"Distribution parameters must satisfy 0 <= a < b < oo")
        return GeneratingFunction(f"1/({b - a + 1}) * {var}**{a} * ({var}**({b - a + 1}) - 1) / ({var} - 1)")
