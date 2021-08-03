r"""
------------------------
Generating Functions
------------------------

Want to calculate the generating function of a program?
You're in the right module!
"""
import functools
import logging

import sympy
from typing import Union, Sequence, get_args

from probably.analysis.config import ForwardAnalysisConfig
from probably.analysis.exceptions import ObserveZeroEventError
from probably.analysis.generating_function import GeneratingFunction, NotComputableException
from probably.analysis.pgfs import PGFS
from probably.pgcl import (Instr, SkipInstr, WhileInstr, IfInstr, AsgnInstr, GeometricExpr,
                           CategoricalExpr, ChoiceInstr, TickInstr, ObserveInstr, DUniformExpr,
                           Expr, BinomialExpr, PoissonExpr, LogDistExpr, BernoulliExpr,
                           DistrExpr, Binop, BinopExpr, VarExpr, NatLitExpr, ExpectationInstr, RealLitExpr, UnopExpr,
                           Unop, Queries, ProbabilityQueryInstr, PlotInstr, LoopInstr)
from probably.pgcl.syntax import check_is_linear_expr

logger = logging.getLogger("probably.analysis.discrete")
logger.setLevel(logging.DEBUG)
fhandler = logging.FileHandler(filename='test.log', mode='a')
fhandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fhandler)


def _while_handler(instr: WhileInstr,
                   input_gf: GeneratingFunction,
                   config: ForwardAnalysisConfig) -> GeneratingFunction:
    user_choice = int(input("While Instruction has only limited support. Choose an option:\n"
                            "[1]: Solve using invariants (Checks whether the invariant over-approximates the loop)\n"
                            "[2]: Fix a maximum number of iterations (This results in an under-approximation)\n"
                            "[3]: Analyse until a certain probability mass is captured (might not terminate!)\n"))
    logger.info(f"User chose {user_choice}")
    if user_choice == 1:
        raise NotImplementedError("Invariants not yet supported")

    elif user_choice == 2:
        max_iter = int(input("Specify a maximum iteration limit: "))
        sat_part, non_sat_part, approx = input_gf.safe_filter(instr.cond)
        for i in range(max_iter):
            iterated_part = compute_distribution(instr.body, sat_part)
            iterated_sat, iterated_non_sat, iterated_approx = iterated_part.safe_filter(instr.cond)
            if iterated_non_sat == GeneratingFunction("0") and iterated_sat == sat_part:
                print(f"Terminated already after {i} step(s)!")
                break
            non_sat_part += iterated_non_sat
            sat_part = iterated_sat
        return non_sat_part
    elif user_choice == 3:
        captured_probability_threshold = sympy.S(input("Enter the probability threshold: "))
        sat_part, non_sat_part, approx = input_gf.safe_filter(instr.cond)
        while non_sat_part.coefficient_sum() < captured_probability_threshold:
            iterated_part = compute_distribution(instr.body, sat_part)
            iterated_sat, iterated_non_sat, iterated_approx = iterated_part.safe_filter(instr.cond)
            non_sat_part += iterated_non_sat
            sat_part = iterated_sat
        return non_sat_part
    else:
        raise Exception(f"Input '{user_choice}' cannot be handled properly.")


def _ite_handler(instr: IfInstr,
                 input_gf: GeneratingFunction,
                 config: ForwardAnalysisConfig) -> GeneratingFunction:
    sat_part, non_sat_part, approx = input_gf.safe_filter(instr.cond)
    non_sat_part = input_gf - sat_part
    return compute_distribution(instr.true, sat_part) + compute_distribution(instr.false, non_sat_part)


def _distr_handler(instr: AsgnInstr,
                   input_gf: GeneratingFunction,
                   config: ForwardAnalysisConfig) -> GeneratingFunction:
    # rhs is a uniform distribution
    if isinstance(instr.rhs, DUniformExpr):
        variable = instr.lhs
        marginal = input_gf.linear_transformation(variable, NatLitExpr(0))  # Seems weird but think of program assignments.
        # either use the concise factorized representation of the uniform pgf ...
        start = str(instr.rhs.start.value)
        end = str(instr.rhs.end.value)
        if config.use_factorized_duniform:
            return marginal * PGFS.uniform(variable, start, end)
        # ... or use the representation as an explicit polynomial
        else:
            return marginal * GeneratingFunction(PGFS.uniform(variable, start, end)._function.expand())

    # rhs is geometric distribution
    if isinstance(instr.rhs, GeometricExpr):
        variable = instr.lhs
        marginal = input_gf.marginal([variable])
        param = instr.rhs.param
        return marginal * PGFS.geometric(variable, str(param))

    # rhs is binomial distribution
    if isinstance(instr.rhs, BinomialExpr):
        variable = instr.lhs
        marginal = input_gf.marginal([variable])
        return marginal * PGFS.binomial(variable, str(instr.rhs.n), str(instr.rhs.p))

    # rhs is poisson distribution
    if isinstance(instr.rhs, PoissonExpr):
        variable = instr.lhs
        marginal = input_gf.marginal([variable])
        return marginal * PGFS.poisson(variable, str(instr.rhs.param))

    # rhs is bernoulli distribution
    if isinstance(instr.rhs, BernoulliExpr):
        variable = instr.lhs
        marginal = input_gf.marginal([variable])
        return marginal * PGFS.bernoulli(variable, str(instr.rhs.param))

    # rhs is logarithmic distribution
    if isinstance(instr.rhs, LogDistExpr):
        variable = instr.lhs
        marginal = input_gf.marginal([variable])
        return marginal * PGFS.log(variable, str(instr.rhs.param))

    # rhs is a categorical expression (explicit finite distr)
    if isinstance(instr.rhs, CategoricalExpr):
        raise NotImplementedError


def _assignment_handler(instr: AsgnInstr,
                        input_gf: GeneratingFunction,
                        config: ForwardAnalysisConfig) -> GeneratingFunction:
    if isinstance(instr.rhs, get_args(DistrExpr)):
        return _distr_handler(instr, input_gf, config)

    # rhs is a modulo expression
    if isinstance(instr.rhs, BinopExpr) and instr.rhs.operator == Binop.MODULO:
        # currently only unnested modulo operations are supported...
        mod_expr = instr.rhs
        if isinstance(mod_expr.lhs, VarExpr) and isinstance(mod_expr.rhs, NatLitExpr):
            result = PGFS.zero(input_gf.vars())
            ap = input_gf.arithmetic_progression(str(mod_expr.lhs), str(mod_expr.rhs))
            for i in range(mod_expr.rhs.value):
                func = ap[i]
                result += func.linear_transformation(mod_expr.lhs.var, NatLitExpr(0)) * GeneratingFunction(f"{mod_expr.lhs}**{i}")
            print(result)
            return result
        else:
            raise NotImplementedError(f"Nested modulo expressions are currently not supported.")

    # rhs is a linear expression
    if check_is_linear_expr(instr.rhs) is None:
        variable = instr.lhs
        return input_gf.linear_transformation(variable, instr.rhs)

    # rhs is a non-linear expression, precf is finite
    elif input_gf.is_finite():
        result = sympy.S(0)
        for addend in input_gf.as_series():  # Take the addends of the Taylor expressions
            term = addend.as_coefficients_dict()  # Convert them into a dict separating monomials from coefficients
            new_addend = sympy.S(addend).subs(str(instr.lhs), 1)  # create the updated monomial.
            for monomial in term:  # For each of these monomial probability pairs...
                var_powers = monomial.as_powers_dict()  # check the individual powers from the variables
                new_value = sympy.S(str(instr.rhs))
                for var in input_gf.vars():  # for each variable check its current state
                    if var not in var_powers.keys():
                        new_value = new_value.subs(var, 0)
                    else:
                        new_value = new_value.subs(var, var_powers[var])
                new_addend *= sympy.S(str(instr.lhs)) ** new_value  # and update
            result += new_addend
        return GeneratingFunction(result, variables=input_gf.vars(), preciseness=input_gf.precision())

    # rhs is non-linear, precf is infinite support
    else:
        print("The assignment {} is not computable on {}".format(instr, input_gf))
        error = sympy.S(input("Continue with approximation. Enter an allowed relative error (0, 1.0):\t"))
        if 0 < error < 1:
            expanded = input_gf.expand_until((1 - error) * input_gf.coefficient_sum())
            return compute_distribution(instr, expanded)
        else:
            raise NotComputableException("The assignment {} is not computable on {}".format(instr, input_gf))


def _pchoice_handler(instr: ChoiceInstr,
                     input_gf: GeneratingFunction,
                     config: ForwardAnalysisConfig) -> GeneratingFunction:
    lhs_block = compute_distribution(instr.lhs, input_gf)
    rhs_block = compute_distribution(instr.rhs, input_gf)
    return GeneratingFunction(str(instr.prob)) * lhs_block + GeneratingFunction(f"1-{instr.prob}") * rhs_block


def _observe_handler(instr: ObserveInstr,
                     input_gf: GeneratingFunction,
                     config: ForwardAnalysisConfig) -> GeneratingFunction:
    try:
        sat_part, non_sat_part, approx = input_gf.safe_filter(instr.cond)
        normalized = sat_part.normalized()
    except ZeroDivisionError:
        raise ObserveZeroEventError(f"observed event {instr.cond} has probability 0")
    return normalized


def _expectation_handler(instr: Expr,
                         input_gf: GeneratingFunction,
                         config: ForwardAnalysisConfig) -> GeneratingFunction:
    # Expectations of Variables can just be computed by derivatives and limits.
    if isinstance(instr, VarExpr):
        return input_gf.diff(str(instr.var), 1) * GeneratingFunction(str(instr.var))

    # For constants, constant factors and addition, we make use of linearity of the expectation operator.
    elif isinstance(instr, NatLitExpr):
        return GeneratingFunction(str(instr))
    elif isinstance(instr, RealLitExpr):
        return GeneratingFunction(str(instr.to_fraction()))
    elif isinstance(instr, UnopExpr):
        if instr.operator == Unop.NEG:
            return GeneratingFunction("-1") * _expectation_handler(instr.expr, input_gf, config)
        else:
            raise SyntaxError(f"Expression {instr} is not valid.")
    elif isinstance(instr, BinopExpr):
        if instr.operator == Binop.PLUS:
            return _expectation_handler(instr.lhs, input_gf, config) + _expectation_handler(instr.rhs, input_gf, config)
        elif instr.operator == Binop.MINUS:
            return _expectation_handler(instr.lhs, input_gf, config) - _expectation_handler(instr.rhs, input_gf, config)
        elif instr.operator == Binop.TIMES:
            if isinstance(instr.lhs, (NatLitExpr, RealLitExpr)):
                return _expectation_handler(instr.lhs, input_gf, config) * _expectation_handler(instr.rhs, input_gf,
                                                                                                config)
            if isinstance(instr.rhs, (NatLitExpr, RealLitExpr)):
                return _expectation_handler(instr.lhs, input_gf, config) * _expectation_handler(instr.rhs, input_gf,
                                                                                                config)
            # This is the actual hard (non-linear) case. We can solve this by recursively compute the expected value.
            else:
                return _expectation_handler(instr.rhs, _expectation_handler(instr.lhs, input_gf, config), config)

        # Currently we dont support division in expressions.
        elif instr.operator == Binop.DIVIDE:
            raise NotImplementedError("Division currently not supported.")
    else:
        raise SyntaxError("The expression is not vaild.")


def _query_handler(instr: Queries, input_gf: GeneratingFunction, config: ForwardAnalysisConfig) -> GeneratingFunction:
    if isinstance(instr, ExpectationInstr):
        result = _expectation_handler(instr.expr, input_gf, config)
        for var in result.vars():
            result = result.linear_transformation(var, 0)
        print(f"Expected value: {result}")
        return input_gf
    elif isinstance(instr, ProbabilityQueryInstr):
        if isinstance(instr.expr, VarExpr):
            marginal = input_gf.copy()
            for var in marginal.vars().difference({sympy.S(instr.expr.var)}):
                marginal = marginal.linear_transformation(var, 0)
            print(f"Marginal distribution of {instr.expr}: {marginal}")
        else:
            sat_part, _, _ = input_gf.safe_filter(instr.expr)
            prob = sat_part.coefficient_sum() if config.show_rational_probabilities else sat_part.coefficient_sum().evalf()
            print(f"Probability of {instr.expr}: {prob}")
        return input_gf
    elif isinstance(instr, PlotInstr):
        vars = [instr.var_1.var]
        vars += [instr.var_2.var] if instr.var_2 else []

        if instr.prob:
            if instr.prob.is_infinite():
                input_gf.create_histogram(var=vars, p=str(1))
            else:
                input_gf.create_histogram(var=vars, p=str(instr.prob))
        elif instr.term_count:
            input_gf.create_histogram(var=vars, n=str(instr.term_count))
        else:
            p = .1
            inc = .1
            while True:
                input_gf.create_histogram(var=vars, p=str(p))
                p = p + inc if p + inc < 1 else 1
                cont = input(f"Continue with p={p}? [Y/n]")
                if cont.lower() == 'n':
                    break
        return input_gf
    else:
        raise SyntaxError(f"Type {type(instr)} is not known.")


def _loop_handler(instr, input_gf, config):
    for i in range(instr.iterations.value):
        logger.debug(f"Computing iteration {i+1} out of {instr.iterations.value}")
        input_gf = compute_distribution(instr.body, input_gf, config)
    return input_gf


def compute_distribution(instr: Union[Instr, Sequence[Instr]],
                         input_gf: GeneratingFunction,
                         config=ForwardAnalysisConfig()) -> GeneratingFunction:
    """
    Build the characteristic function as an expression.

    .. warning::
        Loops are not supported by this function.

    Args:
        instr: The instruction to calculate the cf for, or a list of instructions.
        input_gf: Input generating function.
        config: The configurable options.
    """

    def _show_steps(gf, instr):
        res = compute_distribution(instr, gf, config)
        print(f"Instruction: {instr}\t Result: {res.simplify()}")
        return res

    def _dont_show_steps(gf, instr):
        return compute_distribution(instr, gf, config)

    if isinstance(instr, list):
        func = _show_steps if config.show_intermediate_steps else _dont_show_steps
        return functools.reduce(func, instr, input_gf)

    # Only log instructions when its not a list
    logger.info(f"{instr} gets handled")

    if isinstance(instr, SkipInstr):
        return input_gf

    elif isinstance(instr, WhileInstr):
        return _while_handler(instr, input_gf, config)

    elif isinstance(instr, IfInstr):
        return _ite_handler(instr, input_gf, config)

    elif isinstance(instr, AsgnInstr):
        return _assignment_handler(instr, input_gf, config)

    elif isinstance(instr, ChoiceInstr):
        return _pchoice_handler(instr, input_gf, config)

    elif isinstance(instr, TickInstr):
        raise NotImplementedError("TickInstr not supported in forward analysis")

    elif isinstance(instr, ObserveInstr):
        return _observe_handler(instr, input_gf, config)

    elif isinstance(instr, get_args(Queries)):
        return _query_handler(instr, input_gf, config)

    elif isinstance(instr, LoopInstr):
        return _loop_handler(instr, input_gf, config)

    raise Exception("illegal instruction")


