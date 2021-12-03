from enum import Enum, auto
from typing import Type

import attr

from .distribution import CommonDistributionsFactory
from .exceptions import ConfigurationError
from .fast_generating_function import FPSFactory
from .generating_function import GeneratingFunction
from .optimization.gf_optimizer import GFOptimizer
from .optimization import Optimizer
from .pgfs import PGFS


@attr.s
class ForwardAnalysisConfig:
    """Global configurable options for forward analysis."""

    class Engine(Enum):
        GF = auto()
        GINAC = auto()

    show_intermediate_steps: bool = attr.ib(default=False)

    # IMPORTANT: show_ attributes just change the string representation, not the actual computation
    show_rational_probabilities: bool = attr.ib(default=False)

    use_simplification: bool = attr.ib(default=False)

    # Print output in Latex
    use_latex: bool = attr.ib(default=False)

    # The distribution engine (defaults to generating functions)
    engine: Engine = attr.ib(default=Engine.GF)

    @property
    def optimizer(self) -> Type[Optimizer]:
        if self.engine == ForwardAnalysisConfig.Engine.GF:
            return GFOptimizer
        else:
            raise ConfigurationError("The configured engine does not implement an optimizer.")

    @property
    def factory(self) -> Type[CommonDistributionsFactory]:
        if self.engine == self.Engine.GF:
            return PGFS
        elif self.engine == self.Engine.GINAC:
            return FPSFactory
        else:
            return CommonDistributionsFactory

    def __attrs_post_init__(self):
        GeneratingFunction.use_latex_output = self.use_latex
        GeneratingFunction.rational_preciseness = self.show_rational_probabilities
        GeneratingFunction.use_simplification = self.use_simplification
        GeneratingFunction.intermediate_results = self.show_intermediate_steps

