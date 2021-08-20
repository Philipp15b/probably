"""
=====================
``probably.analysis``
=====================
"""
from typing import Union, Sequence

from probably.analysis.forward.instruction_handler import SequenceHandler
from .backward.wp import loopfree_wp_transformer
from probably.analysis.forward import ForwardAnalysisConfig, Distribution
from probably.analysis.forward.generating_function import GeneratingFunction
from probably.pgcl.ast.instructions import Instr


def compute_discrete_distribution(instr: Union[Instr, Sequence[Instr]],
                                  dist: Distribution,
                                  config: ForwardAnalysisConfig) -> Distribution:
    result = SequenceHandler.compute(instr, dist, config)
    if SequenceHandler.normalization:
        result = result.normalize()
    return result
