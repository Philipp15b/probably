class ForwardAnalysisError(Exception):
    """Base class for forward analysis-related exceptions."""
    pass


class ObserveZeroEventError(ForwardAnalysisError):
    pass


class DistributionParameterError(ForwardAnalysisError):
    pass
