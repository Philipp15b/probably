import attr
from typing import Dict
from probably.pgcl import Var, Type


@attr.s
class ForwardAnalysisConfig:
    """Global configurable options for forward analysis."""

    show_intermediate_steps: bool = attr.ib(default=False)

    # IMPORTANT: Currently has no effect and needs to be set manually in Generating Function class
    verbose_generating_functions: bool = attr.ib(default=False)

    # IMPORTANT: show_ attributes just change the string representation, not the actual computation
    show_rational_probabilities: bool = attr.ib(default=False)

    # IMPORTANT: this field is just a hack to circumvent passing the program to all instruction handlers.
    parameters: Dict[Var, Type] = attr.ib(default=[])
