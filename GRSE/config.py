"""Options kept separate from upstream MASkills configuration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GroupConfig:
    rollouts_per_question: int = 4
    # None uses config.trajectories_per_iteration as the QUESTION count.
    questions_per_iteration: int | None = None
    credit_question_batch_size: int | None = None
    selection_mode: str = "all"  # select: metadata routing; all: full injection
    temperature: float | None = None  # None: provider default, for restricted models
    # Actor temperature for the baseline, gate and val/test reports. None
    # keeps the training temperature; upstream LOCOMO evaluates at 0.
    eval_temperature: float | None = None
    # "locomo" ports the prompt changes upstream's per-category runner applies.
    prompt_profile: str = "default"
    # Which rollout groups reach the critic. "zero_variance" is DAPO's
    # dynamic-sampling filter (drop groups whose rewards are all equal);
    # "solved" drops only groups where every rollout gets full reward.
    group_filter: str = "none"
    structured_failures: bool = False
    category_analysis: bool = False
    validation_rollouts: int = 1
    validation_mode: str = "independent"  # MASkills gate; sequential_team is an ablation
    epsilon: float = 1e-8
    seed: int = 42
    # LOCOMO conversation split. Kept apart from `seed` so that repeats with
    # another seed are tested on the same conversations.
    split_seed: int = 42
    bootstrap_empty: bool = False
    allow_no_validation: bool = False
    # Ablation: one rollout per question, so there is no group contrast (every
    # advantage is 0). Everything else, including the critic prompt, is unchanged.
    single_rollout: bool = False

    def cache_options(self):
        """Keep pre-analysis rollout fingerprints valid for the original mode."""
        options = asdict(self)
        if not self.single_rollout:  # absent in runs from before the ablation existed
            options.pop("single_rollout")
        # Runs from before split_seed existed split with `seed`, which was 42 for them.
        if self.split_seed == 42:
            options.pop("split_seed")
        if not self.structured_failures and not self.category_analysis:
            options.pop("structured_failures")
            options.pop("category_analysis")
        return options

    def __post_init__(self):
        if not isinstance(self.structured_failures, bool) or not isinstance(self.category_analysis, bool):
            raise ValueError("Analysis switches must be booleans")
        if self.category_analysis and not self.structured_failures:
            raise ValueError("category_analysis requires structured_failures")
        for name in ("rollouts_per_question", "validation_rollouts"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.single_rollout:
            if self.rollouts_per_question != 1:
                raise ValueError("single_rollout requires rollouts_per_question=1")
        elif self.rollouts_per_question < 2:
            raise ValueError("Group-relative training needs at least two rollouts")
        for name in ("questions_per_iteration", "credit_question_batch_size"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.validation_mode not in ("independent", "sequential_team"):
            raise ValueError("validation_mode must be independent or sequential_team")
        if self.selection_mode not in ("select", "all"):
            raise ValueError("selection_mode must be 'select' or 'all'")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        for name in ("temperature", "eval_temperature"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 2:
                raise ValueError(f"{name} must be in [0, 2] or None")
        if self.prompt_profile not in ("default", "locomo"):
            raise ValueError("prompt_profile must be 'default' or 'locomo'")
        if self.group_filter not in ("none", "zero_variance", "solved"):
            raise ValueError("group_filter must be 'none', 'zero_variance' or 'solved'")
