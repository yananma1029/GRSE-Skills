"""Group-relative, training-free skill evolution for MASkills.

Run from the repository root with ``python -m GRSE.train``.
This package edits skill text; it does not optimize model weights.
"""

from .config import GroupConfig
from .critic import GroupSkillCreditCritic
from .trainer import GroupSkillTrainer

__all__ = ["GroupConfig", "GroupSkillCreditCritic", "GroupSkillTrainer"]
