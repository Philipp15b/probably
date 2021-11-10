from typing import Union, Generator, Set, Iterator, Tuple, Dict

from .distribution import MarginalType
from .distribution import Distribution, CommonDistributionsFactory, Param
import prodigy

from ...pgcl import VarExpr, Expr, BinopExpr, UnopExpr, Binop, Unop
from ...pgcl.ast.expressions import IidSampleExpr, GeometricExpr


class FPSFactory(CommonDistributionsFactory):

    @staticmethod
    def geometric(var: Union[str, VarExpr], p: Param) -> Distribution:
        return FPS(str(prodigy.geometric(var, str(p))))

    @staticmethod
    def uniform(var: Union[str, VarExpr], a: Param, b: Param) -> Distribution:
        f = f"1/({b} - {a} + 1) * ({var}^{a}) * (({var}^({b} - {a} + 1) - 1)/({var} - 1))"
        return FPS(f)

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

    """
    This class models a probability distribution in terms of a formal power series.
    These formal powerseries are itself provided by `prodigy` a python binding to GiNaC,
    something similar to a computer algebra system implemented in C++.
    """

    def __init__(self, expression: str):
        self.dist = prodigy.Dist(expression)

    @classmethod
    def from_dist(cls, dist: prodigy.Dist):
        return cls(str(dist))

    def __add__(self, other):
        if isinstance(other, str):
            return FPS.from_dist(self.dist + other)
        elif isinstance(other, FPS):
            return FPS.from_dist(self.dist + other.dist)
        else:
            raise NotImplementedError(f"Addition of {self.dist} and {other} not supported.")

    def __sub__(self, other):
        if isinstance(other, str):
            return FPS.from_dist(self.dist - other)
        elif isinstance(other, FPS):
            return FPS.from_dist(self.dist - other.dist)
        else:
            raise NotImplementedError(f"Subtraction of {self.dist} and {other} not supported.")

    def __mul__(self, other):
        if isinstance(other, str):
            return FPS.from_dist(self.dist * other)
        elif isinstance(other, FPS):
            return FPS.from_dist(self.dist * other.dist)
        else:
            raise NotImplementedError(f"Multiplication of {self.dist} and {other} not supported.")

    def __truediv__(self, other):
        raise NotImplementedError(__name__)

    def __eq__(self, other):
        if isinstance(other, FPS):
            return self.dist == other.dist
        else:
            return False

    def __str__(self):
        return str(self.dist)

    def __repr__(self):
        return self.dist.__repr__()

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, int]]]:
        raise NotImplementedError(__name__)

    def copy(self, deep: bool = True) -> Distribution:
        raise NotImplementedError(__name__)

    def get_probability_of(self, condition: Union[Expr, str]):
        raise NotImplementedError(__name__)

    def get_probability_mass(self) -> Union[Expr, str]:
        return self.dist.mass()

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        return self.dist.E(expression)

    def normalize(self) -> Distribution:
        return FPS.from_dist(self.dist.normalize())

    def get_variables(self) -> Set[str]:
        raise NotImplementedError(__name__)

    def get_parameters(self) -> Set[str]:
        raise NotImplementedError(__name__)

    def filter(self, condition: Union[Expr, str]) -> Distribution:
        if isinstance(condition, BinopExpr):
            if condition.operator == Binop.AND:
                return self.filter(condition.lhs).filter(condition.rhs)
            elif condition.operator == Binop.OR:
                filtered_left = self.filter(condition.lhs)
                return filtered_left + self.filter(condition.rhs) - filtered_left.filter(condition.lhs)

            # Normalize the conditional to variables on the lhs from the relation symbol.
            if isinstance(condition.rhs, VarExpr):
                if condition.operator == Binop.EQ:
                    return self.filter(BinopExpr(operator=Binop.EQ, lhs=condition.rhs, rhs=condition.lhs))
                elif condition.operator == Binop.LEQ:
                    return self.filter(BinopExpr(operator=Binop.GEQ, lhs=condition.rhs, rhs=condition.lhs))
                elif condition.operator == Binop.LE:
                    return self.filter(BinopExpr(operator=Binop.GE, lhs=condition.rhs, rhs=condition.lhs))
                elif condition.operator == Binop.GEQ:
                    return self.filter(BinopExpr(operator=Binop.LEQ, lhs=condition.rhs, rhs=condition.lhs))
                elif condition.operator == Binop.GE:
                    return self.filter(BinopExpr(operator=Binop.LE, lhs=condition.rhs, rhs=condition.lhs))

            # is normalized conditional
            if isinstance(condition.lhs, VarExpr):
                if condition.operator == Binop.EQ:
                    return FPS.from_dist(self.dist.filterEq(str(condition.lhs), str(condition.rhs)))
                elif condition.operator == Binop.LE:
                    return FPS.from_dist(self.dist.filterLess(str(condition.lhs), str(condition.rhs)))
                elif condition.operator == Binop.LEQ:
                    return FPS.from_dist(self.dist.filterLeq(str(condition.lhs), str(condition.rhs)))
                elif condition.operator == Binop.GE:
                    return FPS.from_dist(self.dist.filterGreater(str(condition.lhs), str(condition.rhs)))
                elif condition.operator == Binop.GEQ:
                    return FPS.from_dist(self.dist.filterGeq(str(condition.lhs), str(condition.rhs)))
        elif isinstance(condition, UnopExpr):
            # unary relation
            if condition.operator == Unop.NEG:
                return self - self.filter(condition.expr)
        else:
            raise SyntaxError(f"Filtering Condition has unknown format {condition}.")

    def is_zero_dist(self) -> bool:
        return self.dist.isZero()

    def is_finite(self) -> bool:
        raise NotImplementedError(__name__)

    def update(self, expression: Expr) -> Distribution:
        return FPS.from_dist(self.dist.update(str(expression.lhs), str(expression.rhs)))

    def update_iid(self, sampling_exp: IidSampleExpr, variable: Union[str, VarExpr]) -> 'Distribution':

        sample_dist = sampling_exp.sampling_dist
        if not isinstance(sample_dist, GeometricExpr):
            NotImplementedError("Currently only geometric Distributions are supported.")

        result = self.dist.updateIid(str(variable),
                                     prodigy.geometric("test", str(sample_dist.param)),
                                     str(sampling_exp.variable))
        return FPS.from_dist(result)

    def marginal(self, *variables: Union[str, VarExpr], method: MarginalType = MarginalType.Include) -> Distribution:
        # TODO: Make this work with an arbitrary number of variables to marginalize.
        if len(variables) > 1:
            raise NotImplementedError(__name__)
        else:
            if method == MarginalType.Exclude:
                result = self.dist
                for var in variables:
                    result = result.update(str(var), "0")
                return FPS.from_dist(result)
            elif method == MarginalType.Include:
                for var in variables:
                    return FPS.from_dist(self.dist.marginal(var))

    def set_variables(self, *variables: str) -> Distribution:
        raise NotImplementedError(__name__)

    def approximate(self, threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        raise NotImplementedError(__name__)
