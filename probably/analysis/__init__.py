"""
=====================
``probably.analysis``
=====================
"""

from probably.analysis.forward.instruction_handler import SequenceHandler
from .backward.wp import loopfree_wp_transformer
from probably.analysis.forward.config import ForwardAnalysisConfig
from probably.analysis.forward.generating_function import GeneratingFunction

compute_discrete_distribution = SequenceHandler.compute
