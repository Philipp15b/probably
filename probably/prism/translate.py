"""
--------------------
The PRISM Translator
--------------------

Internal functions for translating parts of a pCGL control graph / program to
PRISM snippets, used by probably.prism.backend.
"""

from typing import Any

from probably.pgcl.ast import *
from probably.pgcl.cfg import BasicBlock, TerminatorKind


class PrismTranslatorException(Exception):
    pass


def block_prism(block: BasicBlock, program: Program) -> str:
    """
    Translate a pGCL block to a PRISM snippet. There are multiple state
    transitions for each block, at least one for the program part and one for
    the terminator.
    """
    prism_program = ""
    # from probabilities to assignments
    # with probability 1: go to terminator
    distribution: Any = [(1.0, [("ter", BoolLitExpr(True))])]

    for assignment in block.assignments:
        var = assignment[0]
        expr = assignment[1]
        # prism does the distributions in a different order, just globally
        # outside the assignments. that's why we explicitely have to list
        # all cases.
        if isinstance(expr, CategoricalExpr):
            new_distribution = []
            for prob_old, other_assignments in distribution:
                for prob_new, value in expr.distribution():
                    new_distribution.append(
                        (prob_old * prob_new.to_fraction(),
                         other_assignments + [(var, value)]))
            distribution = new_distribution
        else:
            for _, other_assignments in distribution:
                other_assignments.append((var, expr))
    condition = f"ter=false & ppc={block.ident}"

    def make_expression(var: Var, value: Expr, program: Program):
        return f"({var}'={expression_prism(value, program)})"

    assignment_string = " + ".join([
        (f"{prob} : " if prob < 1 else "") + "&".join([
            make_expression(var, value, program) for var, value in assignments
        ]) for prob, assignments in distribution
    ])
    block_logic = f"[] ({condition}) -> {assignment_string};\n"

    # if these gotos are None, this means quit the program, which we translate
    # as ppc == -1
    if block.terminator.if_true is None:
        goto_true = -1
    else:
        goto_true = block.terminator.if_true

    if block.terminator.if_false is None:
        goto_false = -1
    else:
        goto_false = block.terminator.if_false

    if block.terminator.is_goto():
        terminator_logic = f"[] (ter=true & ppc={block.ident}) -> (ter'=false)&(ppc'={goto_true});\n"
    elif block.terminator.kind == TerminatorKind.BOOLEAN:
        cond = expression_prism(block.terminator.condition, program)
        terminator_logic = "".join([
            f"[] (ter=true & ppc={block.ident} & {cond}) -> (ter'=false)&(ppc'={goto_true});\n"
            f"[] (ter=true & ppc={block.ident} & !({cond})) -> (ter'=false)&(ppc'={goto_false});\n"
        ])
    elif block.terminator.kind == TerminatorKind.PROBABILISTIC:
        cond = expression_prism(block.terminator.condition, program)
        terminator_logic = f"[] (ter=true & ppc={block.ident}) -> {cond} : (ppc'={goto_true})&(ter'=false) + 1-({cond}) : (ppc'={goto_false})&(ter'=false);\n"
    else:
        raise RuntimeError(f"{block.terminator} not implemented")

    prism_program = block_logic + terminator_logic
    return prism_program


def type_prism(typ: Type) -> str:
    """
    Translate a pGCL type to a PRISM type.
    """
    if isinstance(typ, BoolType):
        return "bool"
    elif isinstance(typ, NatType):
        return "int"
    elif isinstance(typ, RealType):
        return "double"
    raise PrismTranslatorException("Type not implemented:", typ)


def is_int(expr: Expr, program: Program):
    """
    Whether an expression is at its core an integer (natural number).
    """
    if isinstance(expr, NatLitExpr):
        return True
    if isinstance(expr, VarExpr):
        if expr.var in program.variables and isinstance(
                program.variables[expr.var], NatType):
            return True
        if expr.var in program.parameters and isinstance(
                program.parameters[expr.var], NatType):
            return True
        if expr.var in program.constants and isinstance(
                program.constants[expr.var].value, NatLitExpr):
            return True
    if isinstance(expr, UnopExpr):
        return is_int(expr.expr, program)
    if isinstance(expr, BinopExpr):
        return is_int(expr.lhs, program) and is_int(expr.rhs, program)
    return False


def expression_prism(expr: Expr, program: Program) -> str:
    """
    Translate a pGCL expression to a PRISM expression.
    """
    if isinstance(expr, BoolLitExpr):
        return "true" if expr.value else "false"
    elif isinstance(expr, NatLitExpr):
        # PRISM understands natural numbers
        return str(expr.value)
    elif isinstance(expr, RealLitExpr):
        # PRISM understands fractions
        return str(expr.to_fraction())
    elif isinstance(expr, VarExpr):
        # Var == str
        return str(expr.var)
    elif isinstance(expr, UnopExpr):
        operand = expression_prism(expr.expr, program)
        if expr.operator == Unop.NEG:
            return f"!({operand})"
        elif expr.operator == Unop.IVERSON:
            raise PrismTranslatorException(
                "Not implemented: iverson brackets like", expr)
        raise PrismTranslatorException("Operator not implemented:", expr)
    elif isinstance(expr, BinopExpr):
        lhs = expression_prism(expr.lhs, program)
        rhs = expression_prism(expr.rhs, program)
        if expr.operator == Binop.OR:
            return f"({lhs}) | ({rhs})"
        elif expr.operator == Binop.AND:
            return f"({lhs}) & ({rhs})"
        elif expr.operator == Binop.LEQ:
            return f"({lhs}) <= ({rhs})"
        elif expr.operator == Binop.LT:
            return f"({lhs}) < ({rhs})"
        elif expr.operator == Binop.GEQ:
            return f"({lhs}) <= ({rhs})"
        elif expr.operator == Binop.GT:
            return f"({lhs}) < ({rhs})"
        elif expr.operator == Binop.EQ:
            return f"({lhs}) = ({rhs})"
        elif expr.operator == Binop.PLUS:
            return f"({lhs}) + ({rhs})"
        elif expr.operator == Binop.MINUS:
            return f"({lhs}) - ({rhs})"
        elif expr.operator == Binop.TIMES:
            return f"({lhs}) * ({rhs})"
        elif expr.operator == Binop.MODULO:
            return f"mod({lhs}, {rhs})"
        elif expr.operator == Binop.POWER:
            return f"pow({lhs}, {rhs})"
        elif expr.operator == Binop.DIVIDE:
            # PRISM doesn't have the concept of integer division, so we need to
            # cook this ourselves
            if is_int(expr.lhs, program) and is_int(expr.rhs, program):
                return f"floor({lhs} / {rhs})"
            else:
                return f"{lhs} / {rhs}"
        raise PrismTranslatorException("Operator not implemented:", expr)
    elif isinstance(expr, SubstExpr):
        raise PrismTranslatorException(
            "Substitution expression not implemented:", expr)
    raise PrismTranslatorException("Operator not implemented:", expr)
