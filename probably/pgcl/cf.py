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
from copy import deepcopy
from typing import Dict, Sequence, Union

from .ast import (AsgnInstr, Binop, BinopExpr, CategoricalExpr, ChoiceInstr,
                  Expr, FloatLitExpr, RealLitExpr, IfInstr, Instr, Program,
                  SkipInstr, SubstExpr, TickExpr, TickInstr, DUniformExpr, CUniformExpr, Unop,
                  UnopExpr, Var, VarExpr, WhileInstr)

from probably.Analysis.generating_function import GeneratingFunction
from .syntax import check_is_linear_expr



def loopfree_cf(instr: Union[Instr, Sequence[Instr]],
                precf: GeneratingFunction) -> GeneratingFunction:
    """
    Build the characteristic function as an expression. See also
    :func:`loopfree_cf_transformer`.

    .. warning::
        Loops are not supported by this function.

    .. todo::
        At the moment, the returned expression is a tree and not a DAG.
        Subexpressions that occur multiple times are *deepcopied*, even though that
        is not strictly necessary. For example, ``jump := unif(0,1); t := t + 1``
        creates an AST where the second assignment occurs twice, as different Python objects,
        even though the two substitutions generated by the *uniform* expression could reuse it.
        We do this because the :mod:`probably.pgcl.substitute` module cannot yet handle non-tree ASTs.

    .. doctest::

        >>> from .parser import parse_pgcl
        >>> from .ast import FloatLitExpr, VarExpr

        >>> program = parse_pgcl("bool a; bool x; if (a) { x := 1 } {}")
        >>> res = loopfree_cf(program.instructions, FloatLitExpr("1.0"))
        >>> str(res)
        '([a] * ((1.0)[x/1])) + ([not a] * 1.0)'

        >>> program = parse_pgcl("bool a; bool x; if (a) { { x := 1 } [0.5] {x := 2 } } {} x := x + 1")
        >>> res = loopfree_cf(program.instructions, VarExpr("x"))
        >>> str(res)
        '([a] * (((((x)[x/x + 1])[x/1]) * 0.5) + ((((x)[x/x + 1])[x/2]) * (1.0 - 0.5)))) + ([not a] * ((x)[x/x + 1]))'

        >>> program = parse_pgcl("nat x; x := unif(1, 4)")
        >>> res = loopfree_cf(program.instructions, VarExpr("x"))
        >>> str(res)
        '(((1/4 * ((x)[x/1])) + (1/4 * ((x)[x/2]))) + (1/4 * ((x)[x/3]))) + (1/4 * ((x)[x/4]))'

        >>> program = parse_pgcl("bool x; x := true : 0.5 + false : 0.5;")
        >>> res = loopfree_cf(program.instructions, VarExpr("x"))
        >>> str(res)
        '(0.5 * ((x)[x/true])) + (0.5 * ((x)[x/false]))'

    Args:
        instr: The instruction to calculate the cf for, or a list of instructions.
        precf: The precf.
    """

    if isinstance(instr, list):
        return functools.reduce(lambda x, y: loopfree_cf(y, x),
                                instr, precf)

    if isinstance(instr, SkipInstr):
        return precf

    if isinstance(instr, WhileInstr):
        raise Exception("While instruction not supported for cf generation")

    if isinstance(instr, IfInstr):
        sat_part = precf.filter(instr.cond)
        non_sat_part = precf - sat_part

        return loopfree_cf(instr.true, sat_part) + loopfree_cf(instr.false, non_sat_part)

    if isinstance(instr, AsgnInstr):
        if check_is_linear_expr(instr.rhs) is None:
            variable = instr.lhs
            return precf.linear_transformation(variable, instr.rhs)
        else:
            raise NotImplementedError("Currently only supporting linear instructions")

    if isinstance(instr, ChoiceInstr):
        lhs_block = loopfree_cf(instr.lhs, precf)
        rhs_block = loopfree_cf(instr.rhs, precf)
        return GeneratingFunction(instr.prob) * lhs_block + GeneratingFunction("1-" + str(instr.prob)) * rhs_block

    if isinstance(instr, TickInstr):
        raise NotImplementedError("Dont support TickInstr in CF setting")

    raise Exception("illegal instruction")