r"""
------------------------
Characteristic Functions
------------------------

Want to calculate the characteristic function of a program?
You're in the right module!

.. autoclass:: LoopCFTransformer
.. autofunction:: one_loop_cf_transformer
.. autofunction:: general_cf_transformer

"""
import functools
from typing import Sequence, Union

from .ast import (Instr, SkipInstr, WhileInstr, IfInstr, AsgnInstr,
                  ChoiceInstr, TickInstr)

from probably.analysis.generating_function import GeneratingFunction
from .syntax import check_is_linear_expr
import sympy

def loopfree_cf(instr: Union[Instr, Sequence[Instr]],
                precf: GeneratingFunction) -> GeneratingFunction:
    """
    Build the characteristic function as an expression.

    .. warning::
        Loops are not supported by this function.

    Args:
        instr: The instruction to calculate the cf for, or a list of instructions.
        precf: Input generating function.
    """

    if isinstance(instr, list):
        return functools.reduce(lambda x, y: loopfree_cf(y, x), instr, precf)

    if isinstance(instr, SkipInstr):
        return precf

    if isinstance(instr, WhileInstr):
        raise Exception("While instruction not supported for cf generation")

    if isinstance(instr, IfInstr):
        sat_part = precf.filter(instr.cond)
        non_sat_part = precf - sat_part

        return loopfree_cf(instr.true, sat_part) + loopfree_cf(
            instr.false, non_sat_part)

    if isinstance(instr, AsgnInstr):
        if check_is_linear_expr(instr.rhs) is None:
            variable = instr.lhs
            return precf.linear_transformation(variable, instr.rhs)
        elif precf.is_finite():
            result = sympy.S(0)
            for addend in precf.as_series():  # Take the addends of the Taylor expressions
                term = addend.as_coefficients_dict()  # Convert them into a dict separating monomials from coefficients
                new_monomial = sympy.S(1)  # create the updated monomial.
                for monomial, probability in term:  # For each of these monomial probability pairs...
                    var_powers = monomial.as_powers_dict()  # check the individual powers from the variables
                    new_monomial = sympy.S(probability)  # take the corresponding probability and...
                    for var in var_powers:  # for each variable check its current state
                        new_monomial *= var ** (  # and update
                            sympy.S(str(instr.rhs))
                            .subs(var, var_powers[var])
                        )
                result += new_monomial
            return GeneratingFunction(result)
        else:
            raise NotImplementedError("Currently only supporting linear instructions on infinite support distributions")

    if isinstance(instr, ChoiceInstr):
        lhs_block = loopfree_cf(instr.lhs, precf)
        rhs_block = loopfree_cf(instr.rhs, precf)
        return GeneratingFunction(str(
            instr.prob)) * lhs_block + GeneratingFunction(
                "1-" + str(instr.prob)) * rhs_block

    if isinstance(instr, TickInstr):
        raise NotImplementedError("Dont support TickInstr in CF setting")

    raise Exception("illegal instruction")
