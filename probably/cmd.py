# pylint: disable=W
"""
Probably also offers a simple command-line interface for quick program inspection.

If you use `poetry` and do not have probably installed globally, you can use `poetry run probably INPUT`.

.. click:: probably.cmd:main
   :prog: probably
   :show-nested:
"""

from typing import IO

import click
import sympy

import probably.pgcl.compiler as pgcl
from probably.analysis.config import ForwardAnalysisConfig
from probably.pgcl.check import CheckFail
from probably.analysis.discrete import loopfree_gf
from probably.analysis.generating_function import GeneratingFunction


@click.command()
@click.argument('program_file', type=click.File('r'))
@click.argument('input_gf', type=str, required=False)
# pylint: disable=redefined-builtin
def main(program_file: IO, input_gf: str):
    """
    Compile the given program and print some information about it.
    """

    program_source = program_file.read()
    print("Program source:")
    print(program_source)

    program = pgcl.compile_pgcl(program_source)
    if isinstance(program, CheckFail):
        print("Error:", program)
        return

    print("\nProgram instructions:")
    for instr in program.instructions:
        print(instr)

    print()
    if input_gf is None:
        gf = GeneratingFunction("1", set(program.variables.keys()), 1.0, True, True)
        print(gf.vars())
    else:
        gf = GeneratingFunction(input_gf, set(program.variables.keys()), 1.0)
    GeneratingFunction.rational_preciseness = True
    GeneratingFunction.verbose_mode = True
    gf = loopfree_gf(program.instructions, gf, ForwardAnalysisConfig(verbose_generating_functions=True))
    print("\nGeneratingfunction\n", gf._function.refine(sympy.Q.real(sympy.S('x'))).simplify().doit())
    gf = GeneratingFunction(gf._function.refine(sympy.Q.real(sympy.S('x'))).simplify(), preciseness=gf.precision(), closed=True)
    print(f"Create Plot for {gf}")
    gf.create_histogram(var="x")
    print("Plotted")


if __name__ == "__main__":
    # execute only if run as a script
    main()  # pylint: disable=no-value-for-parameter
