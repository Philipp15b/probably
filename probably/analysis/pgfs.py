import sympy as sp

from probably.analysis.exceptions import DistributionParameterError
from probably.analysis.generating_function import GeneratingFunction, RealLitExpr


class PGFS:
    """Implements PGFs of standarf distributions."""
    @classmethod
    def geometric(cls, var, p: RealLitExpr):
        if p.to_fraction() <= 0 or p.to_fraction() > 1:
            raise DistributionParameterError(f"parameter of geom distr must be >0 and <=1, was {p}")
        return GeneratingFunction(sp.S(f"{p} / (1 - (1-{p}) * {var})"))
