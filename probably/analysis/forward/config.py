import attr
from typing import Dict
from probably.pgcl import Var, Type
from .generating_function import GeneratingFunction


@attr.s
class ForwardAnalysisConfig:
    """Global configurable options for forward analysis."""

    show_intermediate_steps: bool = attr.ib(default=False)

    # IMPORTANT: show_ attributes just change the string representation, not the actual computation
    show_rational_probabilities: bool = attr.ib(default=False)

    # IMPORTANT: this field is just a hack to circumvent passing the program to all instruction handlers.
    parameters: Dict[Var, Type] = attr.ib(default=[])

    use_simplification: bool = attr.ib(default=False)

    # Print output in Latex
    use_latex: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        GeneratingFunction.use_latex_output = self.use_latex
        GeneratingFunction.rational_preciseness = self.show_rational_probabilities
        GeneratingFunction.use_simplification = self.use_simplification

