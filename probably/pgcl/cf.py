r"""
------------------------
Characteristic Functions
------------------------

Want to calculate the characteristic function of a program?
You're in the right module!
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
        print(precf)
        sat_part = precf.filter(instr.cond)
        non_sat_part = precf - sat_part

        return loopfree_cf(instr.true, sat_part) + loopfree_cf(
            instr.false, non_sat_part)

    if isinstance(instr, AsgnInstr):
        if check_is_linear_expr(instr.rhs) is None:
            variable = instr.lhs
            return precf.linear_transformation(variable, instr.rhs)
        elif precf.is_finite():
            print(precf)
            result = sympy.S(0)
            for addend in precf.as_series():  # Take the addends of the Taylor expressions
                term = addend.as_coefficients_dict()  # Convert them into a dict separating monomials from coefficients
                new_addend = sympy.S(addend).subs(str(instr.lhs), 1)  # create the updated monomial.
                print(new_addend)
                for monomial in term:  # For each of these monomial probability pairs...
                    var_powers = monomial.as_powers_dict()  # check the individual powers from the variables
                    new_value = sympy.S(str(instr.rhs))
                    for var in precf._variables:  # for each variable check its current state
                        if var not in var_powers.keys():
                            new_value = new_value.subs(var, 0)
                        else:
                            new_value = new_value.subs(var, var_powers[var])
                    new_addend *= sympy.S(str(instr.lhs)) ** new_value  # and update
                result += new_addend
            return GeneratingFunction(result, variables=precf._variables)
        else:
            raise NotImplementedError("Currently only supporting linear instructions on infinite support distributions")

    if isinstance(instr, ChoiceInstr):
        lhs_block = loopfree_cf(instr.lhs, precf)
        rhs_block = loopfree_cf(instr.rhs, precf)
        return GeneratingFunction(str(instr.prob), variables=precf._variables) * lhs_block + GeneratingFunction("1-" + str(instr.prob), variables=precf._variables) * rhs_block


    if isinstance(instr, TickInstr):
        raise NotImplementedError("Dont support TickInstr in CF setting")

    raise Exception("illegal instruction")
