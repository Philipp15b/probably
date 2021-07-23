import attr


@attr.s
class ForwardAnalysisConfig:
    """Global configurable options for forward analysis."""

    use_factorized_duniform: bool = attr.ib(default=True)
    # add more options with their default value here

    show_intermediate_steps: bool = attr.ib(default=False)

    # IMPORTANT: Currently has no effect and needs to be set manually in Generating Function class
    verbose_generating_functions: bool = attr.ib(default=False)

    # IMPORTANT: show_ attributes just change the string representation, not the actual computation
    show_rational_probabilities: bool = attr.ib(default=False)
