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

import probably.pgcl.compiler as pgcl
from probably.pgcl.check import CheckFail
from probably.pgcl.syntax import check_is_linear_program
from probably.analysis.discrete import loopfree_gf
from probably.analysis.generating_function import *



@click.command()
@click.argument('input', type=click.File('r'))
# pylint: disable=redefined-builtin
def main(input: IO):
    """
    Compile the given program and print some information about it.
    """
    program_source = input.read()
    print("Program source:")
    print(program_source)

    program = pgcl.compile_pgcl(program_source)
    if isinstance(program, CheckFail):
        print("Error:", program)
        return

    print("\nProgram instructions:")
    with open("instr", "w") as instr_file:
        for instr in program.instructions:
            print(instr)

    print()
    res = check_is_linear_program(program)
    GeneratingFunction.rational_preciseness = True
    if res is not None:
        print("Program is NOT linear:\n")
        print(f"\t{res}")
        gf = loopfree_gf(program.instructions, GeneratingFunction("1/(2-x)"))
        print("\nCharacteristic-function\n", gf)
        gf.create_histogram()
    else:
        print("Program is linear.")
        gf = loopfree_gf(program.instructions, GeneratingFunction("1/(2-x)"))
        print("\nCharacteristic-function\n", gf)
        gf.create_histogram()
