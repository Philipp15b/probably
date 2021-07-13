r"""
------------------------
Characteristic Functions
------------------------

Want to calculate the characteristic function of a program?
You're in the right module!
"""
import functools
from probably.analysis.generating_function import *
from probably.pgcl.syntax import check_is_linear_expr
import sympy


def loopfree_gf(instr: Union[Instr, Sequence[Instr]],
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
        return functools.reduce(lambda x, y: loopfree_gf(y, x), instr, precf)

    if isinstance(instr, SkipInstr):
        return precf

    if isinstance(instr, WhileInstr):
        raise Exception("While instruction not supported for cf generation")

    if isinstance(instr, IfInstr):
        try:
            sat_part = precf.filter(instr.cond)
            non_sat_part = precf - sat_part
        except NotComputable as err:
            print(err)
            probability = input("Continue with approximation. Enter a probability (0, {}):\t"
                                .format(precf.coefficient_sum()))
            if probability > 0:
                return loopfree_gf(instr, precf.expand_until(probability))
            else:
                raise NotComputable(str(err))
        else:
            return loopfree_gf(instr.true, sat_part) + loopfree_gf(
                instr.false, non_sat_part)

    if isinstance(instr, AsgnInstr):
        if check_is_linear_expr(instr.rhs) is None:
            variable = instr.lhs
            return precf.linear_transformation(variable, instr.rhs)
        elif precf.is_finite():
            result = sympy.S(0)
            for addend in precf.as_series():  # Take the addends of the Taylor expressions
                term = addend.as_coefficients_dict()  # Convert them into a dict separating monomials from coefficients
                new_addend = sympy.S(addend).subs(str(instr.lhs), 1)  # create the updated monomial.
                for monomial in term:  # For each of these monomial probability pairs...
                    var_powers = monomial.as_powers_dict()  # check the individual powers from the variables
                    new_value = sympy.S(str(instr.rhs))
                    for var in precf.vars():  # for each variable check its current state
                        if var not in var_powers.keys():
                            new_value = new_value.subs(var, 0)
                        else:
                            new_value = new_value.subs(var, var_powers[var])
                    new_addend *= sympy.S(str(instr.lhs)) ** new_value  # and update
                result += new_addend
            return GeneratingFunction(result, variables=precf.vars(), preciseness=precf.precision())
        else:
            print("The assigntment {} is not computable on {}".format(instr, precf))
            probability = float(input("Continue with approximation. Enter a probability (0, {}):\t"
                                      .format(precf.coefficient_sum())))
            if 0 < probability < precf.coefficient_sum():
                expanded = precf.expand_until(probability)
                return loopfree_gf(instr, expanded)
            else:
                raise NotComputable("The assigntment {} is not computable on {}".format(instr, precf))

    if isinstance(instr, ChoiceInstr):
        lhs_block = loopfree_gf(instr.lhs, precf)
        rhs_block = loopfree_gf(instr.rhs, precf)
        return GeneratingFunction(str(instr.prob), variables=precf.vars(), preciseness=precf.precision()) * lhs_block +\
            GeneratingFunction("1-" + str(instr.prob), variables=precf.vars(), preciseness=precf.precision()) * rhs_block

    if isinstance(instr, TickInstr):
        raise NotImplementedError("Dont support TickInstr in CF setting")

    raise Exception("illegal instruction")
