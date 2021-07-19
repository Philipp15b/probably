import attr

@attr.s
class ForwardAnalysisConfig:
    """Global configurable options for forward analysis."""

    use_factorized_duniform: bool = attr.ib(default=True)
    # add more options with their default value here

    show_intermediate_steps: bool = attr.ib(default=True)