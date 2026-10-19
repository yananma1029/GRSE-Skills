"""Give upstream MASkills the same output-format handling as GRSE.

GRSE runs repair malformed SKILL.md output, ask for the exact frontmatter
shape, constrain the critic to valid JSON, and allow 1200 tokens for induced
and consolidated skills. Upstream MASkills does none of this, so on LOCOMO it
dropped four induced skills as malformed. ``install()`` applies only these
format-level changes to the upstream operators, leaving every task
instruction, the credit method and the training loop untouched, so a
MASkills-versus-GRSE comparison isolates credit assignment.

Call it after any benchmark patch that replaces the upstream prompts (the
LOCOMO runner's ``install_runtime_patches``), since it appends to them.
Refinement is left as is: GRSE inherits the same upstream refine operator.
"""

import maskills.core.optimizer as optimizer_mod
import maskills.core.skill_credit as credit_mod
import maskills.core.skill_operators as operators_mod

from .locomo import FORMAT_TAIL
from .optimizer import _normalize_skill_md

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    # Same frontmatter instruction GRSE appends to these two prompts.
    operators_mod._INDUCE_PROMPT += FORMAT_TAIL
    operators_mod._CONSOLIDATE_PROMPT += FORMAT_TAIL
    # induce_skill / consolidate_skills parse parse_skill_md(_strip_fences(raw));
    # route that through GRSE's repair (it does its own fence handling).
    operators_mod._strip_fences = _normalize_skill_md

    # Upstream caps induction/consolidation at 900 tokens; GRSE uses 1200.
    # Those two calls are the only ones that request 900.
    original_call = optimizer_mod.PolicyGradientOptimizer._llm_call

    def _llm_call(self, prompt, max_tokens=400):
        return original_call(self, prompt, max_tokens=1200 if max_tokens == 900 else max_tokens)

    optimizer_mod.PolicyGradientOptimizer._llm_call = _llm_call

    # JSON mode for the per-trajectory credit critic, as for GRSE's group
    # critic. GroupSkillCreditCritic defines its own _call_llm and is unaffected.
    def _call_llm_json(self, prompt, max_tokens=1500):
        params = {"model": self._model, "messages": [{"role": "user", "content": prompt}],
                  "response_format": {"type": "json_object"}}
        limit_key = ("max_completion_tokens" if any(
            name in self._model.lower() for name in ("o1", "o3", "gpt-5")) else "max_tokens")
        params[limit_key] = max_tokens
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()

    credit_mod.SkillCreditCritic._call_llm = _call_llm_json


def announce_run_id(experiment_cls):
    """Print "MASKILLS run: <id>" when the upstream runner builds its experiment.

    Upstream runners never print their run id, and per-category LOCOMO runs
    share one name prefix, so repeated runs could not be matched to their
    directories. This mirrors the "GRPO run: <id>" line of GRSE.train.
    """
    original = experiment_cls.from_json

    def from_json(*args, **kwargs):
        experiment = original(*args, **kwargs)
        print(f"MASKILLS run: {experiment.trainer.run_id}", flush=True)
        return experiment

    experiment_cls.from_json = staticmethod(from_json)


def route_critic_to_optimizer(experiment_cls):
    """Give the upstream credit critic the optimizer's model, not the actor's.

    Upstream runners set ``llm`` to the actor and never set ``critic_llm``, so
    the critic silently follows the actor. MASkills uses its strong optimizer
    model for credit assignment, and GRSE's critic is the optimizer model too;
    with a different actor (e.g. Gemma) both methods must keep that critic. A
    no-op when actor and optimizer are the same model.
    """
    original = experiment_cls.from_json

    def from_json(config_path, overrides=None, *args, **kwargs):
        overrides = dict(overrides or {})
        if overrides.get("optimizer_llm") and not overrides.get("critic_llm"):
            overrides["critic_llm"] = overrides["optimizer_llm"]
        return original(config_path, overrides, *args, **kwargs)

    experiment_cls.from_json = staticmethod(from_json)
