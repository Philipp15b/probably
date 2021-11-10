from probably.analysis import ForwardAnalysisConfig, Distribution, GeneratingFunction
from probably.pgcl import Program, IfInstr, SkipInstr, VarExpr
from probably.pgcl.analyzer.syntax import check_is_one_big_loop
from probably.analysis import compute_discrete_distribution


def phi(program: Program, invariant: Program):
    assert check_is_one_big_loop(program.instructions) is None, "Program can only be one big loop to analyze."

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
    for i, variable in enumerate(program.variables):
        if i == 0:
            dist = config.factory.geometric(variable, VarExpr(var=f"p{i}", is_parameter=True))
        else:
            dist *= config.factory.geometric(variable, VarExpr(var=f"p{i}", is_parameter=True))
    return dist


def check_equivalence(program: Program, invariant: Program, config: ForwardAnalysisConfig):
    """
    This method uses the fact that we can sometimes determine program equivalence,
    by checking the equality of two parametrized infinite-state Distributions.
    :param config: The configuration.
    :param program: The While-Loop program
    :param invariant: The loop-free invariant
    :return: True, False, Unknown
    """

    # First we create the modified input program in order to fit the premise of Park's Lemma
    modified_inv = phi(program, invariant)

    # Now we have to generate a infinite state parametrized distribution for every program variable.
    test_dist = generate_equivalence_test_distribution(program, config)

    print(f"Test distribution: {test_dist}")

    # Compute the resulting distributions for both programs
    print("Compute the modified invariant...")
    modified_inv_result = compute_discrete_distribution(modified_inv.instructions, test_dist, config)
    print(f"Result (mod-inv): {modified_inv_result}")
    print("")
    print("Compute the invariant...")
    inv_result = compute_discrete_distribution(invariant.instructions, test_dist, config)
    print(f"result (inv): {inv_result}")
    # Compare them and check whether they are equal.

    if modified_inv_result == inv_result:
        return True
    else:
        return False
