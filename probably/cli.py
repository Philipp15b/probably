# pylint: disable=W
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
from probably.analysis import ForwardAnalysisConfig
from probably.pgcl.typechecker.check import CheckFail
import probably.analysis
from probably.analysis import GeneratingFunction


@click.command()
@click.argument('program_file', type=click.File('r'))
@click.argument('input_gf', type=str, required=False)
@click.option('--intermediate-results', is_flag=True, required=False, default=False)
@click.option('--no-simplification', is_flag=True, required=False, default=False)
@click.option('--use-latex', is_flag=True, required=False, default=False)
@click.option('--show-input-program', is_flag=True, required=False, default=False)
# pylint: disable=redefined-builtin
def main(program_file: IO, input_gf: str, intermediate_results: bool, no_simplification: bool, use_latex: bool,
         show_input_program: bool) -> None:
    """
    Compile the given program and print some information about it.
    """

    # Setup the logging.
    #logging.basicConfig(level=logging.INFO)
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
                                   parameters=program.parameters,
                                   use_latex=use_latex, use_simplification=not no_simplification)
    gf = probably.analysis.compute_discrete_distribution(program.instructions, gf, config)


if __name__ == "__main__":
    # execute only if run as a script
    main()  # pylint: disable=no-value-for-parameter
