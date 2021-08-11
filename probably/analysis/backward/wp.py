r"""
------------------------
Weakest Pre-Expectations
------------------------

Want to calculate the weakest pre-expectation of a program?
You're in the right module! You can read all about weakest pre-expectations of
probabilistic programs in [kam19]_.

The basic function is :func:`loopfree_wp`. It calculates weakest
pre-expectations of loop-free programs. :func:`one_loop_wp` can calculate
weakest pre-expectations of programs that consist of exactly one loop (see
:ref:`one_big_loop`). :func:`general_wp` is applicable to all pGCL programs, but
can produce extraordinarily ugly outputs.

.. rubric:: Expected Runtimes

All functions also support computation of *expected runtimes* (see chapter 7 of
[kam19]_), which are given by :class:`probably.pgcl.ast.TickInstr` and
represented in the computed expectations as a
:class:`probably.pgcl.ast.TickExpr`.

.. doctest::

    >>> from probably.pgcl.parser import parse_pgcl
    >>> program = parse_pgcl("x := 5; tick(5)")
    >>> transformer = loopfree_wp_transformer(program, program.instructions)
    >>> print(transformer)
    λ𝑋. ((𝑋)[x/5]) + (tick(5))
    >>> transformer
    ExpectationTransformer(variable='𝑋', expectation=BinopExpr(operator=Binop.PLUS, lhs=SubstExpr(subst={'x': NatLitExpr(5)}, expr=VarExpr('𝑋')), rhs=TickExpr(expr=NatLitExpr(5))))


It is always possible to represent expected runtimes in such a way.
Theorem 7.11 of [kam19]_ (page 173):

.. math::

    \mathrm{ert}\llbracket{}C\rrbracket(t) = \mathrm{ert}\llbracket{}C\rrbracket(0) + \mathrm{backward}\llbracket{}C\rrbracket{}(t)

Therefore the weakest pre-expectation of a program with ``tick`` instructions
can be obtained by simply ignoring all :class:`probably.pgcl.ast.TickExpr` in
the returned expectation, i.e. replacing them all by zero.

.. [kam19] `Advanced Weakest Precondition Calculi for Probabilistic Programs <https://publications.rwth-aachen.de/record/755408/files/755408.pdf>`_, Benjamin Kaminski, 2019.

Loop-Free
#########

The definition of loop-free weakest pre-expectations (and expected runtimes) is
rather straightforward.

.. autofunction:: loopfree_wp

Transformers
############

.. autoclass:: ExpectationTransformer
.. autofunction:: loopfree_wp_transformer

Loops
#####

The :math:`\mathrm{backward}` semantics of loops require a least fixed-point. It is
undecidable to compute. We do not represent least-fixed points explicitely in
our AST (see :mod:`probably.pgcl.ast`), but instead represent the weakest
pre-expectation transformer of a **single loop** (without any more nested loops)
with optional initialization by a special :class:`LoopExpectationTransformer`.
For such programs with exactly one loop, :func:`one_loop_wp` can calculate the
:class:`LoopExpectationTransformer`.

The function :func:`general_wp` is applicable to **all** pGCL programs. It uses
:func:`probably.pgcl.cfg.program_one_big_loop` to transform arbitrarily structured
programs into programs with one loop. Then it uses :func:`one_loop_wp` to
compute the :class:`LoopExpectationTransformer`.

.. autoclass:: LoopExpectationTransformer
.. autofunction:: one_loop_wp_transformer
.. autofunction:: general_wp_transformer

"""
import functools
from copy import deepcopy
from typing import Dict, Sequence, Union

import attr

from probably.pgcl.ast import *
from probably.pgcl.substitute import substitute_expr
from probably.pgcl.analyzer.syntax import check_is_one_big_loop
from probably.pgcl.ast.walk import Walk, walk_expr


def loopfree_wp(instr: Union[Instr, Sequence[Instr]],
                postexpectation: Expr) -> Expr:
    """
    Build the weakest preexpectation as an expression. See also
    :func:`loopfree_wp_transformer`.

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

        >>> from probably.pgcl.parser import parse_pgcl
        >>> from probably.pgcl.ast import RealLitExpr, VarExpr

        >>> program = parse_pgcl("bool a; bool x; if (a) { x := 1 } {}")
        >>> res = loopfree_wp(program.instructions, RealLitExpr("1.0"))
        >>> str(res)
        '([a] * ((1.0)[x/1])) + ([not a] * 1.0)'

        >>> program = parse_pgcl("bool a; bool x; if (a) { { x := 1 } [0.5] {x := 2 } } {} x := x + 1")
        >>> res = loopfree_wp(program.instructions, VarExpr("x"))
        >>> str(res)
        '([a] * (((((x)[x/x + 1])[x/1]) * 0.5) + ((((x)[x/x + 1])[x/2]) * (1.0 - 0.5)))) + ([not a] * ((x)[x/x + 1]))'

        >>> program = parse_pgcl("nat x; x := unif(1, 4)")
        >>> res = loopfree_wp(program.instructions, VarExpr("x"))
        >>> str(res)
        '(((1/4 * ((x)[x/1])) + (1/4 * ((x)[x/2]))) + (1/4 * ((x)[x/3]))) + (1/4 * ((x)[x/4]))'

        >>> program = parse_pgcl("bool x; x := true : 0.5 + false : 0.5;")
        >>> res = loopfree_wp(program.instructions, VarExpr("x"))
        >>> str(res)
        '(0.5 * ((x)[x/true])) + (0.5 * ((x)[x/false]))'

    Args:
        instr: The instruction to calculate the backward for, or a list of instructions.
        postexpectation: The postexpectation.
    """

    if isinstance(instr, list):
        return functools.reduce(lambda x, y: loopfree_wp(y, x),
                                reversed(instr), postexpectation)

    if isinstance(instr, SkipInstr):
        return postexpectation

    if isinstance(instr, WhileInstr):
        raise Exception("While instruction not supported for backward generation")

    if isinstance(instr, IfInstr):
        true_block = loopfree_wp(instr.true, postexpectation)
        false_block = loopfree_wp(instr.false, deepcopy(postexpectation))
        true = BinopExpr(Binop.TIMES, UnopExpr(Unop.IVERSON, instr.cond),
                         true_block)
        false = BinopExpr(
            Binop.TIMES, UnopExpr(Unop.IVERSON,
                                  UnopExpr(Unop.NEG, instr.cond)), false_block)
        return BinopExpr(Binop.PLUS, true, false)

    if isinstance(instr, AsgnInstr):
        if isinstance(instr.rhs, (DUniformExpr, CategoricalExpr)):
            distribution = instr.rhs.distribution()
            branches = [
                BinopExpr(
                    Binop.TIMES, prob,
                    SubstExpr({instr.lhs: expr}, deepcopy(postexpectation)))
                for prob, expr in distribution
            ]
            return functools.reduce(lambda x, y: BinopExpr(Binop.PLUS, x, y),
                                    branches)
        if isinstance(instr.rhs, CUniformExpr):
            raise Exception(
                "continuous uniform not supported for backward generation")

        subst: Dict[Var, Expr] = {instr.lhs: instr.rhs}
        return SubstExpr(subst, postexpectation)

    if isinstance(instr, ChoiceInstr):
        lhs_block = loopfree_wp(instr.lhs, postexpectation)
        rhs_block = loopfree_wp(instr.rhs, deepcopy(postexpectation))
        lhs = BinopExpr(Binop.TIMES, lhs_block, instr.prob)
        rhs = BinopExpr(Binop.TIMES, rhs_block,
                        BinopExpr(Binop.MINUS, RealLitExpr("1.0"), instr.prob))
        return BinopExpr(Binop.PLUS, lhs, rhs)

    if isinstance(instr, TickInstr):
        return BinopExpr(Binop.PLUS, postexpectation, TickExpr(instr.expr))

    raise Exception("unsupported instruction")


@attr.s
class ExpectationTransformer:
    r"""
    Wraps an expectation :math:`f` (represented by
    :class:`probably.pgcl.ast.Expr`) and holds a reference to a variable
    :math:`v`. Together they represent an *expectation transformer*, a function
    :math:`\Phi : \mathbb{E} \to \mathbb{E}` mapping expectations to
    expectations.

    An expectation is applied (:math:`\Phi(g)`) by replacing :math:`f` by
    :math:`g`.

    The :mod:`probably.pgcl.simplify` module can translate expectation
    transformers into *summation normal form*. See
    :class:`probably.pgcl.simplify.SnfExpectationTransformer` and
    :func:`probably.pgcl.simplify.normalize_expectation_transformer`.
    """

    variable: Var = attr.ib()
    """The variable occuring in :data:`expectation`. Must not occur in the
    program."""

    expectation: Expr = attr.ib()

    @property
    def _expectation_ref(self) -> Mut[Expr]:
        def write_expectation(value):
            self.expectation = value

        return Mut(lambda: self.expectation, write_expectation)

    def substitute(self) -> 'ExpectationTransformer':
        """
        Apply all :class:`probably.pgcl.ast.SubstExpr` using
        :func:`probably.pgcl.substitute.substitute_expr`, keeping
        :data:`variable` symbolic.

        Returns ``self`` for convenience.
        """
        substitute_expr(self._expectation_ref,
                        symbolic=frozenset([self.variable]))
        return self

    def apply(self, expectation: Expr, substitute: bool = True) -> Expr:
        """
        Transform the given expectation with this expectation transformer.

        The result is the modified internal expression with the variable
        replaced by the post-expectation, i.e. ``self.expectation``.

        **Calling this method will change this object!**

        .. doctest::

            >>> from probably.pgcl.parser import parse_pgcl
            >>> program = parse_pgcl("x := 3")
            >>> backward = loopfree_wp_transformer(program, program.instructions)
            >>> print(backward)
            λ𝑋. (𝑋)[x/3]
            >>> print(backward.apply(VarExpr("x")))
            3

        Args:
            substitute: Whether to apply all remaining substitutions in the expectation.
        """
        # What would be very wrong:
        #   expr: Expr = SubstExpr({self.variable: expectation}, self.expectation)
        #   expectation_ref = Mut.alloc(expr)
        #   substitute_expr(expectation_ref)
        #   return expectation_ref.val
        #
        # This would first apply inner substitutions, and then replace the
        # post-expectation. Consider the transformer "λ𝑋. (𝑋)[x/3]". Consider
        # the post-expectation "x". Applying a substitution as above would
        # result in "λ𝑋. (𝑋)[x/3][𝑋/x]" - after reduction, we'd get the
        # result "x" instead of the the correct "3"!
        #
        # Instead, simply search and replace the post-expectation in the expression.
        for expr_ref in walk_expr(Walk.DOWN, self._expectation_ref):
            expr = expr_ref.val
            if isinstance(expr, VarExpr) and expr.var == self.variable:
                expr_ref.val = expectation
        # Apply all remaining substitutions for convenience.
        if substitute:
            substitute_expr(self._expectation_ref, deepcopy=True)
        return self.expectation

    def __str__(self) -> str:
        return f"λ{self.variable}. {self.expectation}"


def loopfree_wp_transformer(program: Program,
                            instr: Union[Instr, Sequence[Instr]],
                            variable: str = '𝑋',
                            substitute: bool = True) -> ExpectationTransformer:
    """
    Generalized version of :func:`loopfree_wp` that returns an
    :class:`ExpectationTransformer`.

    Args:
        variable: Optional name for the variable to be used for the transformer. Must not occur elsewhere in the program.
        substitute: Whether to call :meth:`ExpectationTransformer.substitute` on the ``body``.
    """
    assert variable not in program.variables
    expectation = loopfree_wp(instr, VarExpr(variable))
    transformer = ExpectationTransformer(variable, expectation)
    if substitute:
        transformer.substitute()
    return transformer


@attr.s
class LoopExpectationTransformer:
    r"""
    The expectation *transformer* for a pGCL program with **exactly one loop and
    optional initialization assignments before the loop**.
    See :py:func:`one_loop_wp`.

    A loop's expectation transformer is represented by the initialization
    assignments themselves, an expectation transformer for the body, and a
    :data:`done` term which is the expectation for loop termination.

    Note that :data:`done` is not represented by an expectation transformer, but
    rather just an expectation (:class:`probably.pgcl.ast.Expr`). This is
    because the term in the weakest pre-expectation semantics for termination of
    the while loop never contains the post-expectation :math:`f` (it is just
    multiplied with it). A simpler representation allows for more convenient
    further use. The :math:`\mathrm{backward}` semantics of a ``while`` loop are shown
    below:

    .. math::

        \mathrm{backward}\llbracket{}\mathtt{while} (b) \{ C \}\rrbracket(f) = \mathrm{lfp}~X.~ \underbrace{[b] \cdot \mathrm{backward}\llbracket{}C\rrbracket{}(X)}_{\text{body}} + \underbrace{[\neg b]}_{\text{done}} \cdot f

    .. doctest::

        >>> from probably.pgcl.parser import parse_pgcl
        >>> program = parse_pgcl("bool x; x := true; while (x) { x := false }")
        >>> transformer = one_loop_wp_transformer(program, program.instructions)
        >>> print(transformer)
        x := true;
        λ𝐹. lfp 𝑋. [x] * ((𝑋)[x/false]) + [not x] * 𝐹
        >>> print(repr(transformer))
        LoopExpectationTransformer(init=[AsgnInstr(lhs='x', rhs=BoolLitExpr(True))], body=ExpectationTransformer(variable='𝑋', expectation=BinopExpr(operator=Binop.TIMES, lhs=UnopExpr(operator=Unop.IVERSON, expr=VarExpr('x')), rhs=SubstExpr(subst={'x': BoolLitExpr(False)}, expr=VarExpr('𝑋')))), done=UnopExpr(operator=Unop.IVERSON, expr=UnopExpr(operator=Unop.NEG, expr=VarExpr('x'))))
    """

    init: List[AsgnInstr] = attr.ib()
    """Initial assignments before the loop."""
    body: ExpectationTransformer = attr.ib()
    """The expectation transformer for the loop's body."""
    done: Expr = attr.ib()
    """The expectation for when the loop is done."""
    def __str__(self) -> str:
        assignments = " ".join((str(assignment) for assignment in self.init))
        if len(assignments) != 0:
            assignments += "\n"
        # possibly assert that 𝐹 is not in the program?
        return f'{assignments}λ𝐹. lfp {self.body.variable}. {self.body.expectation} + {self.done} * 𝐹'


def one_loop_wp_transformer(
        program: Program,
        instr: Union[Instr, List[Instr]],
        variable: str = '𝑋',
        substitute: bool = True) -> LoopExpectationTransformer:
    """
    Calculate the weakest pre-expectation transformer of a program that consists
    of exactly one while loop with some optional assignments at the beginning.
    That means the only supported programs are of the following form:

    .. code-block::

        x := e1
        y := e2
        while (cond) { .. }

    .. doctest::

        >>> from probably.pgcl.parser import parse_pgcl

        >>> program = parse_pgcl("bool x; bool y; while(x) { { x := true } [0.5] { x := y } }")
        >>> print(one_loop_wp_transformer(program, program.instructions))
        λ𝐹. lfp 𝑋. [x] * ((((𝑋)[x/true]) * 0.5) + (((𝑋)[x/y]) * (1.0 - 0.5))) + [not x] * 𝐹

        >>> program = parse_pgcl("bool x; while(x) { if (x) { x := false } {x := true } }")
        >>> print(one_loop_wp_transformer(program, program.instructions))
        λ𝐹. lfp 𝑋. [x] * (([x] * ((𝑋)[x/false])) + ([not x] * ((𝑋)[x/true]))) + [not x] * 𝐹

    Args:
        variable: Optional name for the variable to be used for the transformer. Must not occur elsewhere in the program.
        substitute: Whether to call :meth:`ExpectationTransformer.substitute` on the ``body``.
    """
    # avoid a cyclic import
    from .simplify import simplifying_neg  # pylint: disable=import-outside-toplevel,cyclic-import

    # extract assignments and the loop from the program
    instrs = list(instr) if isinstance(instr, list) else [instr]
    init: List[AsgnInstr] = []
    while len(instrs) > 0 and isinstance(instrs[0], AsgnInstr):
        next_init = instrs.pop(0)
        assert isinstance(next_init, AsgnInstr)  # mypy is a bit dumb here
        init.append(next_init)
    loop = instrs.pop(0)
    err = "Program must consist of only instructions and a single while loop"
    assert len(instrs) == 0, err
    assert isinstance(loop, WhileInstr), err

    # weakest pre-expectation of the loop body
    body = loopfree_wp_transformer(program,
                                   loop.body,
                                   variable=variable,
                                   substitute=substitute)
    body.expectation = BinopExpr(Binop.TIMES, UnopExpr(Unop.IVERSON,
                                                       loop.cond),
                                 body.expectation)

    # weakest pre-expectation of the done term
    done = UnopExpr(Unop.IVERSON, simplifying_neg(loop.cond))

    return LoopExpectationTransformer(init, body, done)


def general_wp_transformer(
        program: Program,
        substitute: bool = True) -> LoopExpectationTransformer:
    """
    Calculate the weakest pre-expectation transformer for any pGCL program. For
    programs that consist of one big loop (see :ref:`one_big_loop`),
    :func:`one_loop_wp_transformer` is invoked directly. All other programs are
    run through :func:`probably.pgcl.cfg.program_one_big_loop` first and then
    :func:`one_loop_wp_transformer` is used. This will introduce a new,
    additional variable for the program counter.

    .. doctest::

        >>> from probably.pgcl.parser import parse_pgcl

        >>> program = parse_pgcl("bool x; while(x) { while (y) {} }")
        >>> print(general_wp_transformer(program))
        pc := 1;
        λ𝐹. lfp 𝑋. [not (pc = 0)] * (([pc = 1] * (([x] * ((𝑋)[pc/2])) + ([not x] * ((𝑋)[pc/0])))) + ([not (pc = 1)] * (([y] * ((𝑋)[pc/2])) + ([not y] * ((𝑋)[pc/1]))))) + [pc = 0] * 𝐹

    Args:
        substitute: Whether to call :meth:`ExpectationTransformer.substitute` on the ``body``.
    """
    # avoid a cyclic import
    from probably.pgcl.cfg import \
        program_one_big_loop  # pylint: disable=import-outside-toplevel,cyclic-import

    one_big_loop_err = check_is_one_big_loop(program.instructions)
    if one_big_loop_err is None:
        return one_loop_wp_transformer(program, program.instructions)
    program = program_one_big_loop(program, pc_var='pc')
    return one_loop_wp_transformer(program,
                                   program.instructions,
                                   substitute=substitute)