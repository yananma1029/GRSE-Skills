"""One grounded feedback object per question, then reuse skill aggregation."""

import json

from maskills.core.skill_credit import _QA_BREVITY_HINT, SkillCreditCritic, _extract_json

from .group import GroupFeedback
from .prompts import GROUP_CREDIT


def _compact_steps(steps, context, seen_prompts, rollout_index):
    """Drop text the critic already has elsewhere in the payload.

    Every step repeats the task context (already in payload["task"]), the
    system prompt is identical across rollouts, and ``action`` copies
    ``output``.  Left in, 4 HotPotQA rollouts overflow a 32k context.
    """
    compact = []
    for step_index, step in enumerate(steps):
        step = dict(step)
        if step.get("action") == step.get("output"):
            step.pop("action", None)
        if context and isinstance(step.get("input"), str):
            step["input"] = step["input"].replace(context, "[see task.context]")
        prompt = step.get("system_prompt")
        if isinstance(prompt, str) and prompt:
            if prompt in seen_prompts:
                step["system_prompt"] = f"[same as {seen_prompts[prompt]}]"
            else:
                seen_prompts[prompt] = f"rollout {rollout_index} step {step_index}"
        compact.append(step)
    return compact


class GroupSkillCreditCritic(SkillCreditCritic):
    extra_instructions = ""  # benchmark-specific guidance, e.g. GRSE.locomo
    analysis_instructions = ""

    def _analysis_payload(self, group):
        return {}

    def _finish_feedback(self, feedback, data, group, policies):
        return feedback

    def _call_llm(self, prompt, max_tokens=1500):
        # JSON mode: the server constrains decoding to valid JSON. Free-form
        # output from Qwen broke on unescaped quotes in ~2% of groups.
        params = {"model": self._model, "messages": [{"role": "user", "content": prompt}],
                  "response_format": {"type": "json_object"}}
        limit_key = ("max_completion_tokens" if any(
            name in self._model.lower() for name in ("o1", "o3", "gpt-5")) else "max_tokens")
        params[limit_key] = max_tokens
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()

    def evaluate_group_skill_credit(self, group, policies):
        available = {
            agent: [{"id": s.skill_id, "name": s.name,
                     "description": s.description, "body": s.body}
                    for s in p.skill_library]
            for agent, p in policies.items()
        }
        observed = set()
        rollouts = []
        context = group.trajectories[0].task.get("context", "")
        seen_prompts = {}
        for index, (traj, advantage) in enumerate(zip(group.trajectories, group.advantages)):
            for agent, entries in traj.skill_trace.items():
                for entry in entries:
                    observed.update((agent, sid) for sid in entry.get("skills", []))
            rollouts.append({
                "rollout_index": index, "reward": traj.reward, "advantage": advantage,
                "steps": _compact_steps(traj.steps, context, seen_prompts, index),
                "skill_trace": traj.skill_trace,
                "trace_semantics": traj.metadata.get("skill_trace_semantics", "unknown"),
                "selected_skills": traj.metadata.get("selected_skills", {}),
                "final_answer": traj.metadata.get("final_answer", ""),
            })
        # LOCOMO's session_meta is the whole conversation (~36k tokens). Like
        # upstream's critic, show only what the agents retrieved (grep_log).
        task = {k: v for k, v in group.trajectories[0].task.items() if k != "session_meta"}
        payload = {"task": task, "available_library": available, "rollouts": rollouts}
        payload.update(self._analysis_payload(group))
        instructions = GROUP_CREDIT
        if self.extra_instructions:
            instructions += "\n\n" + self.extra_instructions
        if self.analysis_instructions:
            instructions += "\n\n" + self.analysis_instructions
        prompt = self._tool_library_block() + instructions + "\n\nINPUT:\n" + json.dumps(
            payload, ensure_ascii=False)
        if getattr(self, "answer_brevity_hint", False):
            prompt += "\n\n" + _QA_BREVITY_HINT
        return self._request_feedback(prompt, group, policies, observed)

    def _request_feedback(self, prompt, group, policies, observed):
        raw = self._call_llm(prompt, max_tokens=4000)
        return self._decode_feedback(raw, group, policies, observed)

    def _decode_feedback(self, raw, group, policies, observed):
        data = _extract_json(raw)
        credit = self._parse(raw, policies, group.mean_reward)
        seen, filtered = set(), []
        for c in credit.skill_credits:
            key = (c.agent, c.skill_id)
            if key in seen or key not in observed:
                continue
            if policies[c.agent].skill_library.get(c.skill_id) is None:
                continue
            seen.add(key)
            filtered.append(c)
        credit.skill_credits = filtered
        evidence = {}
        block = data.get("induction_evidence", {})
        if isinstance(block, dict):
            for agent, entries in block.items():
                if agent in policies and isinstance(entries, list):
                    evidence[agent] = [s.strip() for s in entries
                                       if isinstance(s, str) and s.strip()][:6]
        return self._finish_feedback(
            GroupFeedback(credit=credit, induction_evidence=evidence), data, group, policies)
