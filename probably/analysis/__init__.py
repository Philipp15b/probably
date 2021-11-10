"""
=====================
``probably.analysis``
=====================
"""

from .backward.wp import loopfree_wp_transformer
from probably.analysis.forward.config import ForwardAnalysisConfig
from probably.analysis.forward.distribution import Distribution
from probably.analysis.forward.generating_function import GeneratingFunction
from probably.analysis.forward.instruction_handler import compute_discrete_distribution
