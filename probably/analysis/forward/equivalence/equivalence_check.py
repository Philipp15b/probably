import logging
from typing import Tuple, Optional

from probably.analysis.forward.distribution import Distribution
from probably.analysis.forward.config import ForwardAnalysisConfig
from probably.pgcl import Program, IfInstr, SkipInstr, VarExpr
from probably.pgcl.analyzer.syntax import check_is_one_big_loop
from probably.analysis.forward.instruction_handler import compute_discrete_distribution
from probably.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


def phi(program: Program, invariant: Program):
    assert check_is_one_big_loop(program.instructions) is None, "Program can only be one big loop to analyze."
    logger.debug("Create modified invariant program.")
    new_instructions = program.instructions[0].body.copy()

    for instr in invariant.instructions:
        new_instructions.append(instr)

    guarded_instr = IfInstr(cond=program.instructions[0].cond, true=new_instructions, false=[SkipInstr()])

    return Program(config=invariant.config,
                   declarations=invariant.declarations,
                   variables=invariant.variables,
                   constants=invariant.constants,
                   parameters=invariant.parameters,
                   instructions=[guarded_instr])


def generate_equivalence_test_distribution(program: Program, config: ForwardAnalysisConfig) -> Distribution:
    # TODO: give dist an initial value as this looks scary!
    logger.debug("Generating test distribution.")
    dist = config.factory.one()
    for i, variable in enumerate(program.variables):
        dist *= config.factory.geometric(variable, VarExpr(var=f"p{i}", is_parameter=True))
    return dist


def check_equivalence(program: Program, invariant: Program, config: ForwardAnalysisConfig) \
        -> Tuple[bool, Optional[Distribution]]:
    """
    This method uses the fact that we can sometimes determine program equivalence,
    by checking the equality of two parametrized infinite-state Distributions.
    :param config: The configuration.
    :param program: The While-Loop program
    :param invariant: The loop-free invariant
    :return: True, False, Unknown
    """

    logger.debug("Checking equivalence.")
    # First we create the modified input program in order to fit the premise of Park's Lemma
    modified_inv = phi(program, invariant)

    # Now we have to generate a infinite state parametrized distribution for every program variable.
    test_dist = generate_equivalence_test_distribution(program, config)

    # Compute the resulting distributions for both programs
    logger.debug("Compute the modified invariant...")
    modified_inv_result = compute_discrete_distribution(modified_inv.instructions, test_dist, config)
    logger.debug(f"modified invariant result:\t{modified_inv_result}")
    logger.debug("Compute the invariant...")
    inv_result = compute_discrete_distribution(invariant.instructions, test_dist, config)
    logger.debug(f"invariant result:\t{inv_result}")
    # Compare them and check whether they are equal.
    logger.debug("Compare results")
    if modified_inv_result == inv_result:
        logger.debug("Invariant validated.")
        return True, inv_result
    else:
        logger.debug("Invariant could not be validated.")
        return False, None
