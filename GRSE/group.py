"""Question groups and finite, population-normalized advantages."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from statistics import fmean, pstdev

from maskills.core.base import Trajectory
from maskills.core.skill_credit import TrajectorySkillCredit


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def relative_advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    if not rewards or not all(math.isfinite(r) for r in rewards):
        raise ValueError("A group needs finite rewards")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if len(rewards) == 1:  # single-rollout ablation: no contrast to normalize against
        return [0.0]
    mean, std = fmean(rewards), pstdev(rewards)
    # No reward contrast: preserve the group for residual-based induction.
    if std <= epsilon:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + epsilon) for r in rewards]


@dataclass
class RolloutGroup:
    group_id: str
    trajectories: list[Trajectory]
    advantages: list[float]

    @classmethod
    def build(cls, group_id, trajectories, epsilon=1e-8):
        if len({fingerprint(t.task) for t in trajectories}) != 1:
            raise ValueError("Every trajectory in a group must have the same task")
        return cls(group_id, trajectories, relative_advantages(
            [t.reward for t in trajectories], epsilon))

    @property
    def mean_reward(self):
        return fmean(t.reward for t in self.trajectories)


@dataclass
class GroupFeedback:
    # One credit per (agent, skill) per QUESTION, not per rollout.
    credit: TrajectorySkillCredit
    induction_evidence: dict[str, list[str]] = field(default_factory=dict)
