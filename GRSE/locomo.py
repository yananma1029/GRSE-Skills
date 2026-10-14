"""Port of the prompt changes upstream's per-category LOCOMO runner applies.

``maskills.experiments.locomo_percat.install_runtime_patches`` swaps in
LOCOMO-specific induce/refine/consolidate/credit prompts, de-duplicates
edited skill bodies, and pins the actor to temperature 0. GRSE reuses the
same text so the comparison differs only in how credit is assigned:

- refine and the edit de-duplication: the upstream patch itself (GRSE
  inherits ``refine_skill``);
- induce and consolidate: upstream text, formatted by GroupSkillOptimizer;
- credit: upstream's field-writing guidance appended to GROUP_CREDIT;
- temperature 0: only for evaluation (``eval_temperature``); training
  rollouts must stay stochastic or every group has zero variance. The
  patched ``LLMClient`` method is shadowed by SamplingLLMClient.

Upstream also sets ``optimizer.task_hints``, but only
``PolicyGradientOptimizer._tool_library_block`` reads it and no
skill-evolution operator calls that, so the hints never reach a prompt in
upstream runs. They are deliberately not ported.
"""

from maskills.experiments import locomo_percat as upstream

# Upstream prompts end with their own SKILL.md output line; this pins the
# exact first line without the default skeleton, whose "When to use" header
# upstream's Requirement 5 bans.
FORMAT_TAIL = """

## Output formatting
The first line must be exactly `---`. Do not write a file name, a preface,
or a code fence (no ```). Shape:
---
name: <short-kebab-case-name>
description: <one sentence>
---
<markdown body>
"""

INDUCE_SKILL = upstream._NEW_INDUCE_PROMPT + FORMAT_TAIL
CONSOLIDATE_SKILLS = upstream._NEW_CONSOLIDATE_PROMPT + FORMAT_TAIL
dedup = upstream._dedup_immediate_repeats


def credit_guidance():
    """The block upstream inserts into the per-trajectory credit prompt."""
    text = upstream._NEW_SKILL_CREDIT_PROMPT
    start = text.index("## How to write the JSON fields")
    end = text.index("## Agent identifier convention")
    return text[start:end].strip()


def install(optimizer, critic):
    upstream.install_runtime_patches()
    optimizer.prompt_profile = "locomo"
    critic.extra_instructions = credit_guidance()
