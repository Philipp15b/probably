"""
Probably also offers a simple command-line interface for quick program inspection.

If you use `poetry` and do not have probably installed globally, you can use `poetry run probably INPUT`.

.. click:: probably.cmd:main
   :prog: probably
   :show-nested:
"""

from typing import IO

import logging
import click

import probably.pgcl.compiler as pgcl
from probably.analysis.forward import ForwardAnalysisConfig
from probably.analysis.forward.equivalence.equivalence_check import check_equivalence
from probably.pgcl.typechecker.check import CheckFail
import probably.analysis
from probably.analysis import GeneratingFunction


@click.group()
def cli():
    pass


@cli.command('main')
@click.argument('program_file', type=click.File('r'))
@click.argument('input_gf', type=str, required=False)
@click.option('--intermediate-results', is_flag=True, required=False, default=False)
@click.option('--no-simplification', is_flag=True, required=False, default=False)
@click.option('--use-latex', is_flag=True, required=False, default=False)
@click.option('--show-input-program', is_flag=True, required=False, default=False)
def main(program_file: IO, input_gf: str, intermediate_results: bool, no_simplification: bool, use_latex: bool,
         show_input_program: bool) -> None:
    """
    Compile the given program and print some information about it.
    """

    # Setup the logging.
    # logging.basicConfig(level=logging.INFO)
    logging.getLogger("probably.cli").info("Program started.")

    # Parse and the input and do typechecking.
    program_source = program_file.read()
    program = pgcl.compile_pgcl(program_source)
    if isinstance(program, CheckFail):
        print("Error:", program)
        return

    if show_input_program:
        print("Program source:")
        print(program_source)
        print()

    if input_gf is None:
        gf = GeneratingFunction("1", *program.variables.keys(), preciseness=1.0, closed=True, finite=True)
    else:
        gf = GeneratingFunction(input_gf, *program.variables.keys(), preciseness=1.0)
    config = ForwardAnalysisConfig(show_intermediate_steps=intermediate_results,
                                   use_latex=use_latex, use_simplification=not no_simplification,
                                   engine=ForwardAnalysisConfig.Engine.GF)
    gf = probably.analysis.compute_discrete_distribution(program.instructions, gf, config)


@cli.command('check_equality')
@click.argument('program_file', type=click.File('r'))
@click.argument('invariant_file', type=click.File('r'))
def check_equality(program_file: IO, invariant_file: IO):
    """
    Checks whether a certain loop-free program is an invariant of a specified while loop.
    :param program_file: the file containing the while-loop
    :param invariant_file: the provided invariant
    :return:
    """
    prog_src = program_file.read()
    inv_src = invariant_file.read()

    prog = pgcl.compile_pgcl(prog_src)
    if isinstance(prog, CheckFail):
        print("Error:", prog)
        return

    inv = pgcl.compile_pgcl(inv_src)
    if isinstance(inv, CheckFail):
        print("Error:", inv)
        return

    equiv = check_equivalence(prog, inv, ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.GINAC))
    print(f"Program is {'not ' if not equiv else ''}equivalent to invaraint")
    return equiv


if __name__ == "__main__":
    # execute only if run as a script
    cli()  # pylint: disable=no-value-for-parameter
