import functools
import logging
from fractions import Fraction

from abc import ABC, abstractmethod
from typing import get_args, Union, Sequence

from probably.analysis.forward import ForwardAnalysisConfig, Distribution, MarginalType, ObserveZeroEventError
from probably.analysis.forward.pgfs import PGFS
from probably.analysis.plotter import Plotter
from probably.pgcl import *
from probably.pgcl.ast.instructions import OptimizationQuery

from probably.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


def _assume(instruction: Instr, instr_type, clsname: str):
    """ Checks that the instruction is given as the right type."""
    assert isinstance(instruction, instr_type), f"The Instruction handled by a {clsname} must be of type" \
                                                f" {instr_type} got {type(instruction)}"


class InstructionHandler(ABC):
    """ Abstract class that defines a strategy for handling a specific program instruction. """

    @staticmethod
    @abstractmethod
    def compute(instruction: Union[Instr, Sequence[Instr]],
                distribution: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:
        pass


class SequenceHandler(InstructionHandler):

    normalization: bool = False

    @classmethod
    def compute(cls, instr: Union[Instr, Sequence[Instr]],
                dist: Distribution,
                config=ForwardAnalysisConfig()) -> Distribution:

        def _show_steps(distribution: Distribution, instruction: Instr):
            res = SequenceHandler.compute(instruction, distribution, config)
            print(f"Instruction: {instruction}\t Result: {res}")
            return res

        def _dont_show_steps(distribution: Distribution, instruction: Instr):
            return SequenceHandler.compute(instruction, distribution, config)

        if isinstance(instr, list):
            func = _show_steps if config.show_intermediate_steps else _dont_show_steps
            return functools.reduce(func, instr, dist)

        if isinstance(instr, SkipInstr):
            return dist

        elif isinstance(instr, WhileInstr):
            logger.info(f"{instr} gets handled")
            return WhileHandler.compute(instr, dist, config)

        elif isinstance(instr, IfInstr):
            logger.info(f"{instr} gets handled")
            return ITEHandler.compute(instr, dist, config)

        elif isinstance(instr, AsgnInstr):
            return AssignmentHandler.compute(instr, dist, config)

        elif isinstance(instr, ChoiceInstr):
            logger.info(f"{instr} gets handled")
            return PChoiceHandler.compute(instr, dist, config)

        elif isinstance(instr, ObserveInstr):
            logger.info(f"{instr} gets handled")
            cls.normalization = True
            return ObserveHandler.compute(instr, dist, config)

        elif isinstance(instr, get_args(Query)):
            logger.info(f"{instr} gets handled")
            if cls.normalization:
                dist = dist.normalize()
                cls.normalization = False
            return QueryHandler.compute(instr, dist, config)

        elif isinstance(instr, LoopInstr):
            logger.info(f"{instr} gets handled")
            return LoopHandler.compute(instr, dist, config)

        raise Exception("illegal instruction")


class QueryHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr, dist: Distribution, config: ForwardAnalysisConfig) -> Distribution:
        _assume(instr, get_args(Query), 'QueryHandler')

        # User wants to compute an expected value of an expression
        if isinstance(instr, ExpectationInstr):
            expression = instr.expr
            if isinstance(expression, (VarExpr, NatLitExpr, RealLitExpr)):
                result = dist.get_expected_value_of(expression)
            elif isinstance(expression, BinopExpr):
                if expression.operator in (Binop.PLUS, Binop.MINUS, Binop.TIMES):
                    result = dist.get_expected_value_of(expression)
                else:
                    raise SyntaxError("Expression cannot be a condition.")
            elif isinstance(expression, UnopExpr) and expression.operator == Unop.NEG:
                result = dist.get_expected_value_of(expression)
            else:
                raise SyntaxError("Expression has wrong format.")

            print(f"Expected value: {result}")
            return dist

        # User wants to compute a marginal, or the probability of a condition.
        elif isinstance(instr, ProbabilityQueryInstr):
            return QueryHandler.__query_probability_of(instr.expr, dist)

        # User wants to Plot something
        elif isinstance(instr, PlotInstr):
            return QueryHandler.__query_plot(instr, dist)

        # User wants to print the current distribution.
        elif isinstance(instr, PrintInstr):
            print(dist)
            return dist

        elif isinstance(instr, OptimizationQuery):
            return QueryHandler.__query_optimization(instr, dist, config)

        else:
            raise SyntaxError("This should not happen.")

    @staticmethod
    def __query_optimization(instr: OptimizationQuery, dist: Distribution, config: ForwardAnalysisConfig):
        logger.debug(f"Computing the optimal value for parameter {instr.parameter}"
                     f"in order to {'maximize' if instr.type == OptimizationType.MAXIMIZE else 'minimize'}"
                     f"the distribution {dist} with respect to {instr.expr}")
        result = config.optimizer.optimize(instr.expr, dist, instr.parameter, method=instr.type)
        if not result:
            print(f"No solutions could be found.")
        elif len(result) == 1:
            print(f"The optimal value is at {instr.parameter}={result[0]}.")
        else:
            print(f"The maximal values can be achieved choosing the parameter {instr.parameter} in {result}.")
        return dist


    @staticmethod
    def __query_plot(instr: PlotInstr, dist: Distribution) -> Distribution:
        # Gather the variables to plot.
        variables = [instr.var_1.var]
        variables += [instr.var_2.var] if instr.var_2 else []

        # User can specify either a probability threshold or a number of terms which are shown in the histogram.
        if instr.prob:
            # User chose a probability or oo (meaning, try to compute the whole histogram).
            if instr.prob.is_infinite():
                Plotter.plot(dist, *variables, threshold=str(1))
            else:
                Plotter.plot(dist, *variables, threshold=str(instr.prob))

        elif instr.term_count:
            # User has chosen a term limit.
            Plotter.plot(dist, *variables, threshold=instr.term_count.value)
            # User did neither specify a term limit nor a threshold probability. We try an iterative approach...
        else:
            p = .1
            inc = .1
            while True:
                Plotter.plot(dist, *variables, threshold=str(p))
                p = p + inc if p + inc < 1 else 1
                cont = input(f"Continue with p={p}? [Y/n]")
                if cont.lower() == 'n':
                    break
        return dist

    @staticmethod
    def __query_probability_of(expression: Expr, dist: Distribution) -> Distribution:
        # Marginal computation
        if isinstance(expression, VarExpr):
            marginal = dist.marginal(expression, method=MarginalType.Include)
            print(f"Marginal distribution of {expression}: {marginal}")
        # Probability of condition computation.
        else:
            sat_part = dist.filter(expression)
            prob = sat_part.get_probability_mass()
            print(f"Probability of {expression}: {prob}")
        return dist


class SampleHandler(InstructionHandler):

    @staticmethod
    @abstractmethod
    def compute(instr: Instr, dist: Distribution, config: ForwardAnalysisConfig) -> Distribution:
        _assume(instr, AsgnInstr, 'SampleHandler')
        assert isinstance(instr.rhs, get_args(DistrExpr)), f"The Instruction handled by a SampleHandler" \
                                                           f" must be of type DistrExpr, got {type(instr)}"

        variable: Var = instr.lhs
        marginal = dist.marginal(variable, method=MarginalType.Exclude)
        factory = config.factory

        # rhs is a categorical expression (explicit finite distr)
        if isinstance(instr.rhs, CategoricalExpr):
            raise NotImplementedError("Categorical expression are currently not supported.")

        # rhs is a uniform distribution
        if isinstance(instr.rhs, DUniformExpr):
            return marginal * factory.uniform(variable, instr.rhs.start, instr.rhs.end)

        # rhs is geometric distribution
        if isinstance(instr.rhs, GeometricExpr):
            return marginal * factory.geometric(variable, instr.rhs.param)

        # rhs is binomial distribution
        if isinstance(instr.rhs, BinomialExpr):
            return marginal * factory.binomial(variable, instr.rhs.n, instr.rhs.p)

        # rhs is poisson distribution
        if isinstance(instr.rhs, PoissonExpr):
            return marginal * factory.poisson(variable, instr.rhs.param)

        # rhs is bernoulli distribution
        if isinstance(instr.rhs, BernoulliExpr):
            return marginal * factory.bernoulli(variable, instr.rhs.param)

        # rhs is logarithmic distribution
        if isinstance(instr.rhs, LogDistExpr):
            return marginal * factory.log(variable, instr.rhs.param)


class AssignmentHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:
        _assume(instr, AsgnInstr, 'AssignmentHandler')

        if isinstance(instr.rhs, get_args(DistrExpr)):
            return SampleHandler.compute(instr, dist, config)

        if not isinstance(instr.rhs, get_args(Expr)):
            raise SyntaxError(f"Assignment {instr} is ill-formed. right-hand-side must be an expression.")

        return dist.update(BinopExpr(operator=Binop.EQ, lhs=VarExpr(instr.lhs), rhs=instr.rhs))


class ObserveHandler(InstructionHandler):

    @staticmethod
    def compute(instruction: Instr,
                distribution: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:

        _assume(instruction, ObserveInstr, 'ObserverHandler')

        sat_part = distribution.filter(instruction.cond)
        if sat_part.get_probability_mass() == 0:
            raise ObserveZeroEventError(f"observed event {instruction.cond} has probability 0")
        return sat_part


class PChoiceHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:
        _assume(instr, ChoiceInstr, 'PChoiceHandlerGF')

        lhs_block = SequenceHandler.compute(instr.lhs, dist)
        rhs_block = SequenceHandler.compute(instr.rhs, dist)
        return lhs_block * str(instr.prob) + rhs_block * f"1-{instr.prob}"


class ITEHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:
        _assume(instr, IfInstr, 'ITEHandler')

        sat_part = dist.filter(instr.cond)
        non_sat_part = dist - sat_part
        return SequenceHandler.compute(instr.true, sat_part, config) + SequenceHandler.compute(instr.false, non_sat_part, config)


class LoopHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr, dist: Distribution, config: ForwardAnalysisConfig) -> Distribution:
        _assume(instr, LoopInstr, 'LoopHandler')
        for i in range(instr.iterations.value):
            logger.debug(f"Computing iteration {i + 1} out of {instr.iterations.value}")
            dist = SequenceHandler.compute(instr.body, dist, config)
        return dist


class WhileHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:

        _assume(instr, WhileInstr, 'WhileHandler')

        user_choice = int(input("While Instruction has only limited support. Choose an option:\n"
                                "[1]: Solve using invariants (Checks whether the invariant over-approximates the loop)\n"
                                "[2]: Fix a maximum number of iterations (This results in an under-approximation)\n"
                                "[3]: Analyse until a certain probability mass is captured (might not terminate!)\n"))
        logger.info(f"User chose {user_choice}")

        if user_choice == 1:
            raise NotImplementedError("Invariants not yet supported")

        elif user_choice == 2:
            max_iter = int(input("Specify a maximum iteration limit: "))
            sat_part = dist.filter(instr.cond)
            non_sat_part = dist - sat_part
            for i in range(max_iter):
                iterated_part = SequenceHandler.compute(instr.body, sat_part)
                iterated_sat = iterated_part.filter(instr.cond)
                iterated_non_sat = iterated_part - iterated_sat
                if iterated_non_sat == PGFS.zero() and iterated_sat == sat_part:
                    print(f"Terminated already after {i} step(s)!")
                    break
                non_sat_part += iterated_non_sat
                sat_part = iterated_sat
            return non_sat_part

        elif user_choice == 3:
            captured_probability_threshold = float(input("Enter the probability threshold: "))
            sat_part = dist.filter(instr.cond)
            non_sat_part = dist - sat_part
            while Fraction(non_sat_part.get_probability_mass()) < captured_probability_threshold:
                logger.info(
                    f"Collected {non_sat_part.get_probability_mass()} / {captured_probability_threshold} "
                    f"({(float((Fraction(non_sat_part.get_probability_mass()) / captured_probability_threshold)) * 100):.2f} %)"
                    f"of the desired mass.")
                iterated_part = SequenceHandler.compute(instr.body, sat_part)
                iterated_sat = iterated_part.filter(instr.cond)
                iterated_non_sat = iterated_part - iterated_sat
                non_sat_part += iterated_non_sat
                sat_part = iterated_sat
            return non_sat_part
        else:
            raise Exception(f"Input '{user_choice}' cannot be handled properly.")
