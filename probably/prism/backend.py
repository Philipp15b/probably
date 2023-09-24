"""
-----------------
The PRISM Backend
-----------------

This module can translate pCGL Programs to PRISM Programs with the function
translate_to_prism.
"""

from probably.pgcl.ast.program import Program
from probably.pgcl.ast.types import NatType
from probably.pgcl.cfg import ControlFlowGraph
from probably.prism.translate import (PrismTranslatorException, block_prism,
                                      expression_prism, type_prism)


def translate_to_prism(program: Program) -> str:
    """
    Translate a pCGL program to a PRISM program. The PRISM program shares the
    same variable names as the pCGL programs, so you can ask a model checker
    like Storm for conditions on these variables:
        
    What is the probability that the boolean done is, at some point, true?
        P=? [F (done)]
    
    What is the probability that the natural number c is always zero?
        P=? [G c=0]
    
    The PRISM program has a single module named "program". The input program
    may not use the variable names "ppc" and "ter".
    """
    # Initialize variables
    prism_program = "dtmc\n"
    # pCGL constants are interpreted as PRISM constants
    for (var, value) in program.constants.items():
        prism_program += f"const {var} = {expression_prism(value, program)};\n"
    prism_program += "module program\n"
    # Parameters and variables are PRISM variables
    for (var, typ) in list(program.parameters.items()) + list(
            program.variables.items()):
        if isinstance(typ, NatType) and typ.bounds is not None:
            prism_program += f"{var} : {typ.bounds};\n"
        else:
            prism_program += f"{var} : {type_prism(typ)};\n"

    graph = ControlFlowGraph.from_instructions(program.instructions)

    # Initialize prism's program counter ppc
    if "ppc" in [x.var for x in program.declarations]:
        raise PrismTranslatorException(
            "Don't declare a variable called ppc, that's needed by the PRISM translator."
        )
    prism_program += f"ppc : int init {graph.entry_id};\n"
    # Initialize terminator execution bool
    if "ppc" in [x.var for x in program.declarations]:
        raise PrismTranslatorException(
            "Don't declare a variable called ter, that's needed by the PRISM translator."
        )
    prism_program += "ter : bool init false;\n"

    for block in graph:
        prism_program += block_prism(block, program)
    prism_program += "endmodule\n"
    return prism_program
