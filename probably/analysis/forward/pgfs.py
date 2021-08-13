from typing import Union

import sympy as sp

from probably.analysis.forward.distribution import CommonDistributionsFactory, Distribution, Param
from probably.analysis.forward.exceptions import DistributionParameterError
from probably.analysis.forward.generating_function import GeneratingFunction
from probably.pgcl import VarExpr
from probably.pgcl.analyzer.syntax import has_variable


class PGFS(CommonDistributionsFactory):
    """Implements PGFs of standard distributions."""

    @staticmethod
    def geometric(var: Union[str, VarExpr], p: Param) -> Distribution:
        if isinstance(p, str) and not (0 < sp.S(p) < 1):
            raise DistributionParameterError(f"parameter of geom distr must be >0 and <=1, was {p}")
        elif isinstance(p, VarExpr) and has_variable(p):
            raise DistributionParameterError(f"Parameter for geometric distribution cannot depend on a program variable.")
        return GeneratingFunction(f"({p}) / (1 - (1-({p})) * {var})", var, closed=True, finite=False)

    @staticmethod
    def uniform(var: Union[str, VarExpr], a: Param, b: Param) -> Distribution:
        if isinstance(a, str) and isinstance(b, str) and not (0 <= sp.S(a) <= sp.S(b)):
            raise DistributionParameterError(f"Distribution parameters must satisfy 0 <= a < b < oo")
        elif (isinstance(a, VarExpr) and has_variable(a)) or (isinstance(b, VarExpr) and has_variable(b)):
            raise DistributionParameterError(f"Parameter for geometric distribution cannot depend on a program variable.")
        return GeneratingFunction(f"1/(({b}) - ({a}) + 1) * {var}**({a}) * ({var}**(({b}) - ({a}) + 1) - 1) / ({var} - 1)", var,
                                  closed=True, finite=True)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: Param) -> Distribution:
        if isinstance(p, str) and not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Bernoulli Distribution must be in [0,1], but was {p}")
        elif isinstance(p, VarExpr) and has_variable(p):
            raise DistributionParameterError(f"Parameter for geometric distribution cannot depend on a program variable.")
        return GeneratingFunction(f"1 - ({p}) + ({p}) * {var}", var, closed=True, finite=True)

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: Param) -> Distribution:
        if isinstance(lam, str) and sp.S(lam) < 0:
            raise DistributionParameterError(f"Parameter of Poisson Distribution must be in [0, oo), but was {lam}")
        elif isinstance(lam, VarExpr) and has_variable(lam):
            raise DistributionParameterError(f"Parameter for geometric distribution cannot depend on a program variable.")
        return GeneratingFunction(f"exp(({lam}) * ({var} - 1))", var, closed=True, finite=False)

    @staticmethod
    def log(var: Union[str, VarExpr], p: Param) -> Distribution:
        if isinstance(p, str) and not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Logarithmic Distribution must be in [0,1], but was {p}")
        elif isinstance(p, VarExpr) and has_variable(p):
            raise DistributionParameterError(f"Parameter for geometric distribution cannot depend on a program variable.")
        return GeneratingFunction(f"log(1-({p})*{var})/log(1-({p}))", var, closed=True, finite=False)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: Param, p: Param) -> Distribution:
        if isinstance(p, str) and not (0 <= sp.S(p) <= 1):
            raise DistributionParameterError(f"Parameter of Binomial Distribution must be in [0,1], but was {p}")
        if isinstance(n, str) and not (0 <= sp.S(n)):
            raise DistributionParameterError(f"Parameter of Binomial Distribution must be in [0,oo), but was {n}")
        elif (isinstance(n, VarExpr) and has_variable(n)) or (isinstance(p, VarExpr) and has_variable(p)):
            raise DistributionParameterError(f"Parameter for geometric distribution cannot depend on a program variable.")
        return GeneratingFunction(f"(1-({p})+({p})*{var})**({n})", var, closed=True, finite=True)

    @staticmethod
    def zero(*variables: Union[str, sp.Symbol]):
        if variables:
            return GeneratingFunction("0", *variables, preciseness=1, closed=True, finite=True)
        return GeneratingFunction("0", preciseness=1, closed=True, finite=True)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where actually no information about the states is given."""
        return PGFS.zero(*map(str, variables))
