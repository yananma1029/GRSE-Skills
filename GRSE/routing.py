"""Task-start skill selection without changing upstream agent/tool protocols."""

import json
from copy import deepcopy

from maskills.core.policy import AgentPolicy
from maskills.core.skill_credit import _extract_json
from maskills.core.skills import SkillLibrary

from .prompts import SELECT_SKILLS


class LLMSkillSelector:
    def __init__(self, client):
        self.client = client

    def select(self, agent, policy, question):
        payload = {
            "agent": agent, "role": policy.role, "question": question,
            "skills": [{"id": s.skill_id, "name": s.name, "description": s.description}
                       for s in policy.skill_library],
        }
        raw = self.client.chat(SELECT_SKILLS, json.dumps(payload, ensure_ascii=False),
                               max_tokens=800)
        data = _extract_json(raw)
        ids = data.get("skill_ids")
        if not isinstance(ids, list) or any(not isinstance(s, str) for s in ids):
            raise ValueError("Skill selector must return a skill_ids string list")
        return ids


class SkillRoutingEnvironment:
    """Expose all metadata, load selected bodies, record disclosure explicitly.

    Selection is once per agent at TASK START, before team execution. It is not
    an in-turn read_skill tool. Only the public question goes to the selector;
    gold answers and hidden context never do. No inference of 'actually used'.
    """

    def __init__(self, env, selector=None, mode="select"):
        if mode not in ("select", "all"):
            raise ValueError("Unknown skill routing mode")
        if mode == "select" and selector is None:
            raise ValueError("select mode needs a selector")
        if getattr(env, "inject_skill_library", False):
            raise ValueError("Disable inject_skill_library; seed skills must be trainable policies")
        self.env, self.selector, self.mode = env, selector, mode

    def __getattr__(self, name):
        return getattr(self.env, name)

    def collect_trajectory(self, policies, task):
        routed, selected, available = {}, {}, {}
        question = str(task.get("question", task.get("Question", task.get("problem", ""))))
        for agent, policy in policies.items():
            ids = policy.skill_library.ids()
            available[agent] = ids
            choice = (self.selector.select(agent, policy.copy(), question)
                      if ids and self.mode == "select" else ids)
            if not isinstance(choice, list) or any(s not in ids for s in choice):
                raise ValueError(f"Selector returned an unavailable skill for {agent}")
            selected[agent] = list(dict.fromkeys(choice))
            routed[agent] = AgentPolicy(
                role=policy.role,
                skill_library=SkillLibrary([
                    policy.skill_library.get(sid).copy() for sid in selected[agent]
                ]),
            )
        traj = self.env.collect_trajectory(routed, deepcopy(task))
        trace = {}
        for index, step in enumerate(traj.steps):
            agent = step.get("agent") or step.get("agent_id")
            if agent in selected:
                trace.setdefault(agent, []).append({
                    "step": index, "skills": selected[agent], "kind": "disclosed",
                })
        traj.skill_trace = trace
        traj.metadata.update({
            "available_skills": available, "selected_skills": selected,
            "skill_trace_semantics": "selected_and_disclosed" if self.mode == "select"
                                     else "all_disclosed_not_proven_used",
        })
        return traj
