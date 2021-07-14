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
        user_choice = int(input("While Instruction has only limited support. Choose an option:\n"
              "[1]: Enter an Invariant\n"
              "[2]: Fix a maximum number of iterations\n"
              "[3]: Analyse until a certain probability mass is captured (might not terminate!)\n"))
        if user_choice == 1:
            invariant = sympy.S(input("Enter the invariant: "))
            # TODO check the invariant here.
            pass
        elif user_choice == 2:
            max_iter = int(input("Specify a maximum iteration limit: "))
            sat_part, non_sat_part, approx = _safe_filter(precf, instr.cond)
            for i in range(max_iter):
                iterated_part = loopfree_gf(instr.body, sat_part)
                if iterated_sat + iterated_non_sat == iterated_part and not iterated_approx:
                    # reached a fixpoint
                    break
                iterated_sat, iterated_non_sat, iterated_approx = _safe_filter(iterated_part, instr.cond)
                non_sat_part += iterated_non_sat
                sat_part = iterated_sat
            return non_sat_part
        elif user_choice == 3:
            captured_probability_threshold = sympy.S(input("Enter the probability threshold: "))
            sat_part, non_sat_part, approx = _safe_filter(precf, instr.cond)
            while non_sat_part.coefficient_sum() < captured_probability_threshold:
                iterated_part = loopfree_gf(instr.body, sat_part)
                iterated_sat, iterated_non_sat, iterated_approx = _safe_filter(iterated_part, instr.cond)
                non_sat_part += iterated_non_sat
                sat_part = iterated_sat
            return non_sat_part
        else:
            raise Exception(f"Input '{user_choice}' cannot be handled properly.")

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
        if isinstance(instr.rhs, DUniformExpr):
            variable = instr.lhs
            marginal = precf.linear_transformation(variable, "0")  # Seems weird but think of program assignments.
            factors = []
            for prob, value in instr.rhs.distribution():
                factors.append(marginal * GeneratingFunction(f"{variable}**{value}*{prob}"))
            return functools.reduce(lambda x, y: x+y, factors)
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
            error = sympy.S(input("Continue with approximation. Enter an allowed relative error (0, 1.0):\t"))
            if 0 < error < 1:
                expanded = precf.expand_until((1 - error) * precf.coefficient_sum())
                return loopfree_gf(instr, expanded)
            else:
                raise NotComputable("The assigntment {} is not computable on {}".format(instr, precf))

    if isinstance(instr, ChoiceInstr):
        lhs_block = loopfree_gf(instr.lhs, precf)
        rhs_block = loopfree_gf(instr.rhs, precf)
        return GeneratingFunction(str(instr.prob), variables=precf.vars(), preciseness=precf.precision()) * lhs_block + \
               GeneratingFunction("1-" + str(instr.prob), variables=precf.vars(),
                                  preciseness=precf.precision()) * rhs_block

    if isinstance(instr, TickInstr):
        raise NotImplementedError("Dont support TickInstr in CF setting")

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
        if probability > 0:
            approx = input_gf.expand_until(probability)
            approx_sat_part = approx.filter(condition)
            approx_non_sat_part = approx - approx_sat_part
            return approx_sat_part, approx_non_sat_part, True
        else:
            raise NotComputable(str(err))
