# pylint: disable=W
"""
Probably also offers a simple command-line interface for quick program inspection.

If you use `poetry` and do not have probably installed globally, you can use `poetry run probably INPUT`.

.. click:: probably.cmd:main
   :prog: probably
   :show-nested:
"""

import pprint
import time
from typing import IO

import click

import probably.pgcl.compiler as pgcl
from probably.pgcl.check import CheckFail
from probably.pgcl.syntax import check_is_linear_program
from probably.pgcl.cf import loopfree_cf


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
            pprint.pprint(instr)
            pprint.pprint(instr,instr_file)

    print()
    res = check_is_linear_program(program)
    if res is not None:
        print("Program is NOT linear:\n")
        print(f"\t{res}")
    else:
        print("Program is linear.")
        #print("\nWeakest pre-expectation transformer:\n", str(general_wp_transformer(program)))
        print("\nCharacteristic-function transformer:\n", str(loopfree_cf(program.instructions,1)))
