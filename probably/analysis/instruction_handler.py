import functools
import logging
from abc import ABC, abstractmethod
from typing import get_args, Union, Sequence

from probably.analysis.config import ForwardAnalysisConfig
from probably.analysis.distribution import Distribution, MarginalType
from probably.analysis.exceptions import ObserveZeroEventError
from probably.analysis.generating_function import GeneratingFunction
from probably.analysis.pgfs import PGFS
from probably.analysis.plotter import Plotter
from probably.pgcl import Instr, WhileInstr, ChoiceInstr, IfInstr, LoopInstr, ObserveInstr, AsgnInstr, DistrExpr, \
    BinopExpr, Binop, VarExpr, NatLitExpr, DUniformExpr, GeometricExpr, BinomialExpr, PoissonExpr, BernoulliExpr, \
    LogDistExpr, CategoricalExpr, Queries, PlotInstr, ProbabilityQueryInstr, ExpectationInstr, RealLitExpr, UnopExpr, \
    Unop, Expr, SkipInstr
from probably.pgcl.syntax import check_is_linear_expr

from probably.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


def __assume(instruction: Instr, instr_type, clsname: str):
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

    @staticmethod
    def compute(instr: Union[Instr, Sequence[Instr]],
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

        # Only log instructions when its not a list
        logger.info(f"{instr} gets handled")

        if isinstance(instr, SkipInstr):
            return dist

        elif isinstance(instr, WhileInstr):
            return WhileHandler.compute(instr, dist, config)

        elif isinstance(instr, IfInstr):
            return ITEHandler.compute(instr, dist, config)

        elif isinstance(instr, AsgnInstr):
            return AssignmentHandler.compute(instr, dist, config)

        elif isinstance(instr, ChoiceInstr):
            return PChoiceHandler.compute(instr, dist, config)

        elif isinstance(instr, ObserveInstr):
            return ObserveHandler.compute(instr, dist, config)

        elif isinstance(instr, get_args(Queries)):
            return QueryHandler.compute(instr, dist, config)

        elif isinstance(instr, LoopInstr):
            return LoopHandler.compute(instr, dist, config)

        raise Exception("illegal instruction")


class QueryHandler(InstructionHandler):

    @staticmethod
    def _expectation_handler(expression: Expr, dist: Distribution, config: ForwardAnalysisConfig):
        # Expectations of Variables can just be computed by derivatives and limits.
        if isinstance(expression, BinopExpr):
            if expression.operator in (Binop.AND, Binop.OR, Binop.EQ, Binop.LE, Binop.LEQ):
                raise SyntaxError("The expression cannot be a boolean condition")
            return dist.get_expected_value_of(expression)
        elif isinstance(expression, (VarExpr, NatLitExpr, RealLitExpr)):
            return dist.get_expected_value_of(expression)
        raise SyntaxError("The expression is not of valid type.")

    @staticmethod
    def compute(instr: Instr, dist: Distribution, config: ForwardAnalysisConfig) -> Distribution:

        #__assume(instr, get_args(Queries), 'QueryHandler')

        # User wants to compute an expected value of an expression
        if isinstance(instr, ExpectationInstr):
            expression = instr.expr
            if isinstance(expression, (VarExpr, NatLitExpr, RealLitExpr)):
                result = dist.get_expected_value_of(expression)
            elif isinstance(expression, BinopExpr):
                if expression.operator in (Binop.PLUS, Binop.MINUS, Binop.TIMES):
                    result = dist.get_expected_value_of(expression)
            elif isinstance(expression, UnopExpr) and expression.operator == Unop.NEG:
                result = dist.get_expected_value_of(expression)
            else:
                raise SyntaxError("Expression has wrong format.")

            print(f"Expected value: {result}")
            return dist

        # User wants to compute a marginal, or the probability of a condition.
        elif isinstance(instr, ProbabilityQueryInstr):
            # Marginal computation
            if isinstance(instr.expr, VarExpr):
                marginal = dist.marginal(instr.expr, method=MarginalType.Include)
                print(f"Marginal distribution of {instr.expr}: {marginal}")
            # Probability of condition computation.
            else:
                sat_part = dist.filter(instr.expr)
                prob = sat_part.get_probability_mass()
                print(f"Probability of {instr.expr}: {prob}")
            return dist

        # User wants to Plot something
        elif isinstance(instr, PlotInstr):

            # Gather the variables to plot.
            variables = [instr.var_1.var]
            variables += [instr.var_2.var] if instr.var_2 else []

            # User can specify either a probability threshold or a number of terms which are shown in the histogram.
            if instr.prob:
                # User chose a probability or oo (meaning, try to compute the whole histogram).
                if instr.prob.is_infinite():
                    Plotter.plot(dist, *variables, p=str(1))
                else:
                    Plotter.plot(dist, *variables, p=str(instr.prob))

            elif instr.term_count:
                # User has chosen a term limit.
                Plotter.plot(dist, *variables, n=instr.term_count.value)
            # User did neither specify a term limit nor a threshold probability. We try an iterative approach...
            else:
                p = .1
                inc = .1
                while True:
                    Plotter.plot(dist, *variables, p=str(p))
                    p = p + inc if p + inc < 1 else 1
                    cont = input(f"Continue with p={p}? [Y/n]")
                    if cont.lower() == 'n':
                        break
            return dist
        else:
            raise SyntaxError(f"Type {type(instr)} is not known.")


class SampleGFHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:

        #__assume(instr, AsgnInstr, 'SampleGFHandler')
        assert isinstance(instr.rhs, get_args(DistrExpr)), f"The Instruction handled by a SampleHandler must be of type" \
                                                 f" DistrExpr got {type(instr)}"

        variable = instr.lhs
        marginal = dist.marginal(variable, method=MarginalType.Exclude)

        # rhs is a categorical expression (explicit finite distr)
        if isinstance(instr.rhs, CategoricalExpr):
            raise NotImplementedError("Categorical expression are currently not supported.")

        # rhs is a uniform distribution
        if isinstance(instr.rhs, DUniformExpr):
            start = str(instr.rhs.start.value)
            end = str(instr.rhs.end.value)
            return marginal * PGFS.uniform(variable, start, end)

        # rhs is geometric distribution
        if isinstance(instr.rhs, GeometricExpr):
            param = instr.rhs.param
            return marginal * PGFS.geometric(variable, str(param))

        # rhs is binomial distribution
        if isinstance(instr.rhs, BinomialExpr):
            return marginal * PGFS.binomial(variable, str(instr.rhs.n), str(instr.rhs.p))

        # rhs is poisson distribution
        if isinstance(instr.rhs, PoissonExpr):
            return marginal * PGFS.poisson(variable, str(instr.rhs.param))

        # rhs is bernoulli distribution
        if isinstance(instr.rhs, BernoulliExpr):
            return marginal * PGFS.bernoulli(variable, str(instr.rhs.param))

        # rhs is logarithmic distribution
        if isinstance(instr.rhs, LogDistExpr):
            return marginal * PGFS.log(variable, str(instr.rhs.param))


class AssignmentHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:

        #__assume(instr, AsgnInstr, 'AssignmentHandler')

        if isinstance(instr.rhs, get_args(DistrExpr)):
            return SampleGFHandler.compute(instr, dist, config)

        return dist.update(str(instr))

        # rhs is a modulo expression
        if isinstance(instr.rhs, BinopExpr) and instr.rhs.operator == Binop.MODULO:
            # currently only unnested modulo operations are supported...
            mod_expr = instr.rhs
            if isinstance(mod_expr.lhs, VarExpr) and isinstance(mod_expr.rhs, NatLitExpr):
                result = PGFS.zero(dist.get_variables())
                ap = dist.arithmetic_progression(str(mod_expr.lhs), str(mod_expr.rhs))
                for i in range(mod_expr.rhs.value):
                    func = ap[i]
                    result += func.linear_transformation(mod_expr.lhs.var, NatLitExpr(0)) * GeneratingFunction(
                        f"{mod_expr.lhs}**{i}")
                print(result)
                return result
            else:
                raise NotImplementedError(f"Nested modulo expressions are currently not supported.")

        # rhs is a linear expression
        if check_is_linear_expr(instr.rhs) is None:
            variable = instr.lhs
            return dist.linear_transformation(variable, instr.rhs)

        # rhs is a non-linear expression, precf is finite
        elif dist.is_finite():
            result = sympy.S(0)
            for addend in dist:  # Take the addends of the Taylor expressions
                term = addend.as_coefficients_dict()  # Convert them into a dict separating monomials from coefficients
                new_addend = sympy.S(addend).subs(str(instr.lhs), 1)  # create the updated monomial.
                for monomial in term:  # For each of these monomial probability pairs...
                    var_powers = monomial.as_powers_dict()  # check the individual powers from the variables
                    new_value = sympy.S(str(instr.rhs))
                    for var in dist.get_variables():  # for each variable check its current state
                        if var not in var_powers.keys():
                            new_value = new_value.subs(var, 0)
                        else:
                            new_value = new_value.subs(var, var_powers[var])
                    new_addend *= sympy.S(str(instr.lhs)) ** new_value  # and update
                result += new_addend
            return GeneratingFunction(result, variables=dist.get_variables(), preciseness=dist.precision())

        # rhs is non-linear, precf is infinite support
        else:
            print("The assignment {} is not computable on {}".format(instr, dist))
            error = sympy.S(input("Continue with approximation. Enter an allowed relative error (0, 1.0):\t"))
            if 0 < error < 1:
                expanded = dist.expand_until((1 - error) * dist.coefficient_sum())
                return compute_distribution(instr, expanded)
            else:
                raise NotComputableException("The assignment {} is not computable on {}".format(instr, dist))


class ObserveHandler(InstructionHandler):

    @staticmethod
    def compute(instruction: Instr,
                distribution: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:

        #__assume(instruction, ObserveInstr, 'ObserverHandler')

        try:
            sat_part = distribution.filter(instruction.cond)
            normalized = sat_part.normalize()
        except ZeroDivisionError:
            raise ObserveZeroEventError(f"observed event {instruction.cond} has probability 0")
        return normalized


class PChoiceHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:
        #__assume(instr, ChoiceInstr, 'PChoiceHandlerGF')

        lhs_block = SequenceHandler.compute(instr.lhs, dist)
        rhs_block = SequenceHandler.compute(instr.rhs, dist)
        return lhs_block * str(instr.prob) + rhs_block * f"1-{instr.prob}"


class ITEHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:
        #__assume(instr, IfInstr, 'ITEHandler')

        sat_part = dist.filter(instr.cond)
        non_sat_part = dist - sat_part
        return SequenceHandler.compute(instr.true, sat_part) + SequenceHandler.compute(instr.false, non_sat_part)


class LoopHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr, dist: Distribution, config: ForwardAnalysisConfig) -> Distribution:
        __assume(instr, LoopInstr, 'LoopHandler')
        for i in range(instr.iterations.value):
            logger.debug(f"Computing iteration {i + 1} out of {instr.iterations.value}")
            dist = SequenceHandler.compute(instr.body, dist, config)
        return dist


class WhileHandler(InstructionHandler):

    @staticmethod
    def compute(instr: Instr,
                dist: Distribution,
                config: ForwardAnalysisConfig) -> Distribution:

        #__assume(instr, WhileInstr, 'WhileHandler')

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
            while non_sat_part.get_probability_mass() < captured_probability_threshold:
                logger.info(
                    f"Collected {non_sat_part.get_probability_mass()} / {captured_probability_threshold} "
                    f"({(float((non_sat_part.get_probability_mass() / captured_probability_threshold)) * 100):.2f} %)"
                    f"of the desired mass.")
                iterated_part = SequenceHandler.compute(instr.body, sat_part)
                iterated_sat = iterated_part.filter(instr.cond)
                iterated_non_sat = iterated_part - iterated_sat
                non_sat_part += iterated_non_sat
                sat_part = iterated_sat
            return non_sat_part
        else:
            raise Exception(f"Input '{user_choice}' cannot be handled properly.")
