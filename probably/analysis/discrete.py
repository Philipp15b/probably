r"""
------------------------
Generating Functions
------------------------

Want to calculate the generating function of a program?
You're in the right module!
"""
import functools
import sympy
from typing import Union, Sequence, Tuple, get_args

from probably.analysis.config import ForwardAnalysisConfig
from probably.analysis.exceptions import ObserveZeroEventError
from probably.analysis.generating_function import GeneratingFunction, NotComputable
from probably.analysis.pgfs import PGFS
from probably.pgcl import (Instr, SkipInstr, WhileInstr, IfInstr, AsgnInstr, GeometricExpr,
                           CategoricalExpr, ChoiceInstr, TickInstr, ObserveInstr, DUniformExpr,
                           Expr, BinomialExpr, PoissonExpr, LogDistExpr,
                           DistrExpr, Binop, BinopExpr, VarExpr, NatLitExpr)
from probably.pgcl.syntax import check_is_linear_expr


def _while_handler(instr: WhileInstr,
                   input_gf: GeneratingFunction,
                   config: ForwardAnalysisConfig) -> GeneratingFunction:
    user_choice = int(input("While Instruction has only limited support. Choose an option:\n"
                            "[1]: Solve using invariants (Checks whether the invariant over-approximates the loop)\n"
                            "[2]: Fix a maximum number of iterations (This results in an under-approximation)\n"
                            "[3]: Analyse until a certain probability mass is captured (might not terminate!)\n"))
    if user_choice == 1:
        raise NotImplementedError("Invariants not yet supported")

    elif user_choice == 2:
        max_iter = int(input("Specify a maximum iteration limit: "))
        sat_part, non_sat_part, approx = _safe_filter(input_gf, instr.cond)
        for i in range(max_iter):
            iterated_part = loopfree_gf(instr.body, sat_part)
            iterated_sat, iterated_non_sat, iterated_approx = _safe_filter(iterated_part, instr.cond)
            if iterated_non_sat == GeneratingFunction("0") and iterated_sat == sat_part:
                print(f"Terminated already after {i} step(s)!")
                break
            non_sat_part += iterated_non_sat
            sat_part = iterated_sat
        return non_sat_part
    elif user_choice == 3:
        captured_probability_threshold = sympy.S(input("Enter the probability threshold: "))
        sat_part, non_sat_part, approx = _safe_filter(input_gf, instr.cond)
        while non_sat_part.coefficient_sum() < captured_probability_threshold:
            iterated_part = loopfree_gf(instr.body, sat_part)
            iterated_sat, iterated_non_sat, iterated_approx = _safe_filter(iterated_part, instr.cond)
            non_sat_part += iterated_non_sat
            sat_part = iterated_sat
        return non_sat_part
    else:
        raise Exception(f"Input '{user_choice}' cannot be handled properly.")


def _ite_handler(instr: IfInstr,
                 input_gf: GeneratingFunction,
                 config: ForwardAnalysisConfig) -> GeneratingFunction:
    try:
        sat_part = input_gf.filter(instr.cond)
        non_sat_part = input_gf - sat_part
    except NotComputable as err:
        print(err)
        probability = input("Continue with approximation. Enter a probability (0, {}):\t"
                            .format(input_gf.coefficient_sum()))
        if probability > sympy.S(0):
            return loopfree_gf(instr, input_gf.expand_until(probability))
        else:
            raise NotComputable(str(err))
    else:
        return loopfree_gf(instr.true, sat_part) + loopfree_gf(
            instr.false, non_sat_part)


def _distr_handler(instr: AsgnInstr,
                   input_gf: GeneratingFunction,
                   config: ForwardAnalysisConfig) -> GeneratingFunction:
    # rhs is a uniform distribution
    if isinstance(instr.rhs, DUniformExpr):
        variable = instr.lhs
        marginal = input_gf.linear_transformation(variable, "0")  # Seems weird but think of program assignments.
        # either use the concise factorized representation of the uniform pgf ...
        start = instr.rhs.start.value
        end = instr.rhs.end.value
        if config.use_factorized_duniform:
            return marginal * PGFS.uniform(variable, str(start), str(end))
        # ... or use the representation as an explicit polynomial
        else:
            return marginal * GeneratingFunction(PGFS.uniform(variable, start, end)._function.expand())

    # rhs is geometric distribution
    if isinstance(instr.rhs, GeometricExpr):
        variable = instr.lhs
        marginal = input_gf.linear_transformation(variable, "0")
        param = instr.rhs.param
        return marginal * PGFS.geometric(variable, str(param))

    # rhs is binomial distribution
    if isinstance(instr.rhs, BinomialExpr):
        variable = instr.lhs
        marginal = input_gf.linear_transformation(variable, "0")
        return marginal * PGFS.binomial(variable, str(instr.rhs.n), str(instr.rhs.p))

    # rhs is poisson distribution
    if isinstance(instr.rhs, PoissonExpr):
        variable = instr.lhs
        marginal = input_gf.linear_transformation(variable, "0")
        return marginal * PGFS.poisson(variable, str(instr.rhs.param))

    # rhs is logarithmic distribution
    if isinstance(instr.rhs, LogDistExpr):
        variable = instr.lhs
        marginal = input_gf.linear_transformation(variable, "0")
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
            for i in range(mod_expr.rhs.value):
                func = input_gf.arithmetic_progression(str(mod_expr.lhs), str(mod_expr.rhs))[i]
                result += func.linear_transformation(mod_expr.lhs, 0) * GeneratingFunction(f"{mod_expr.lhs}**{i}")
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
            return loopfree_gf(instr, expanded)
        else:
            raise NotComputable("The assignment {} is not computable on {}".format(instr, input_gf))


def _pchoice_handler(instr: ChoiceInstr,
                     input_gf: GeneratingFunction,
                     config: ForwardAnalysisConfig) -> GeneratingFunction:
    lhs_block = loopfree_gf(instr.lhs, input_gf)
    rhs_block = loopfree_gf(instr.rhs, input_gf)
    return GeneratingFunction(str(instr.prob)) * lhs_block + GeneratingFunction(f"1-{instr.prob}") * rhs_block


def _observe_handler(instr: ObserveInstr,
                     input_gf: GeneratingFunction,
                     config: ForwardAnalysisConfig) -> GeneratingFunction:
    input_gf = input_gf.filter(instr.cond)
    try:
        input_gf = input_gf.normalized()
    except ZeroDivisionError:
        raise ObserveZeroEventError(f"observed event {instr.cond} has probability 0")
    return input_gf


def loopfree_gf(instr: Union[Instr, Sequence[Instr]],
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

    if isinstance(instr, list):
        return functools.reduce(lambda x, y: loopfree_gf(y, x), instr, input_gf)

    if isinstance(instr, SkipInstr):
        return input_gf

    if isinstance(instr, WhileInstr):
        return _while_handler(instr, input_gf, config)

    if isinstance(instr, IfInstr):
        return _ite_handler(instr, input_gf, config)

    if isinstance(instr, AsgnInstr):
        return _assignment_handler(instr, input_gf, config)

    if isinstance(instr, ChoiceInstr):
        return _pchoice_handler(instr, input_gf, config)

    if isinstance(instr, TickInstr):
        raise NotImplementedError("TickInstr not supported in forward analysis")

    if isinstance(instr, ObserveInstr):
        return _observe_handler(instr, input_gf, config)

    raise Exception("illegal instruction")


def _safe_filter(input_gf: GeneratingFunction, condition: Expr) -> Tuple[GeneratingFunction, GeneratingFunction, bool]:
    # TODO move into filter function in GF class
    try:
        sat_part = input_gf.filter(condition)
        non_sat_part = input_gf - sat_part
        return sat_part, non_sat_part, False
    except NotComputable as err:
        print(err)
        probability = input("Continue with approximation. Enter a probability (0, {}):\t"
                            .format(input_gf.coefficient_sum()))
        if probability > sympy.S(0):
            approx = input_gf.expand_until(probability)
            approx_sat_part = approx.filter(condition)
            approx_non_sat_part = approx - approx_sat_part
            return approx_sat_part, approx_non_sat_part, True
        else:
            raise NotComputable(str(err))
