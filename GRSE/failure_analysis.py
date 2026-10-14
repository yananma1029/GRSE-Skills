"""Opt-in, grounded failure diagnoses and category summaries of TRAIN groups."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean

from maskills.core.skill_credit import ResidualCredit, TrajectorySkillCredit

from .critic import GroupSkillCreditCritic
from .group import GroupFeedback

FAILURE_TYPES = {
    "skill_not_followed", "skill_defect", "missing_capability", "handoff_error",
    "evidence_missing", "answer_format", "tool_failure", "uncertain",
}

STRUCTURED_INSTRUCTIONS = """Additional required output field: failure_analysis (a JSON array).
Keep the existing skills/residual/induction_evidence schema, and add at most 6
root-cause diagnoses, merging repeated occurrences across rollouts. An empty
array is allowed when no failure is supported. Never invent an error just
because reward is below 1: partial credit can reflect answer format.
Use task_category as supplied; never infer a category from correctness.
For each diagnosis return exactly these fields:
{
  "agent": "agent_1",
  "skill_id": null,
  "failure_type": "missing_capability",
  "failure_locations": [{"rollout": 0, "step": 0}],
  "successful_contrast": [{"rollout": 1, "step": 0}],
  "evidence": "What went wrong at the cited step and what the contrast shows",
  "proposed_fix": "A reusable correction, not a task answer",
  "action": "induce"
}
Use citation_constraints: it lists legal full/non-full reward rollout indices
and step_index/agent mappings. Do not copy the example indices blindly.
Indices are zero-based positions in the supplied rollout/steps arrays.
Cite the earliest supported error for this agent, not merely the final wrong
answer. failure_locations must cite non-full-reward rollouts (< 1).
successful_contrast must cite full-reward rollouts (>= 1), or be [] if absent;
positive advantage alone is NOT success. Both kinds of cited steps must belong
to the named agent. Without a successful contrast, describe the correction as
an unverified hypothesis. Observation is not proof of causal skill use.
failure_type must be one of: skill_not_followed, skill_defect,
missing_capability, handoff_error, evidence_missing, answer_format,
tool_failure, uncertain.
action must be refine, induce, or none. refine requires an existing skill_id
recorded in this agent's skill_trace in the cited failed rollout(s); reference
that skill in skills and make suggested_edit agree with proposed_fix.
induce requires skill_id=null and a capability not covered by existing skills.
Do NOT create a skill merely because a tool failed or attribution is uncertain;
for tool_failure/uncertain use action=none. Use agent="unknown" and action=none
if responsibility cannot be grounded; do not blame every teammate.
Prefer fixing applicability, execution checks or handoff rules over duplicating
an existing skill. Preserve behavior that works in other categories.
All-full-reward groups must return failure_analysis=[].
"""


def task_category(task):
    """Use stored dataset labels only; never the top-level 'qa' task type."""
    metadata = task.get("metadata") or {}
    label = task.get("category", metadata.get("category"))
    if label is None or label == "":
        label = metadata.get("type") or metadata.get("question_type")
    if isinstance(label, bool) or not isinstance(label, (str, int)):
        return "unknown"
    return str(label).strip() or "unknown"


@dataclass
class StructuredFeedback(GroupFeedback):
    category: str = "unknown"
    failure_analysis: list[dict] = field(default_factory=list)
    validation_audit: dict = field(default_factory=dict)


def _locations(value, group, agent, successful):
    if not isinstance(value, list) or (not successful and not value):
        raise ValueError("Diagnosis requires a list of grounded failure locations")
    result = []
    for ref in value:
        if not isinstance(ref, dict):
            raise ValueError("Location must be an object")
        rollout, step = ref.get("rollout"), ref.get("step")
        if type(rollout) is not int or not 0 <= rollout < len(group.trajectories):
            raise ValueError("Invalid diagnosis rollout index")
        trajectory = group.trajectories[rollout]
        if type(step) is not int or not 0 <= step < len(trajectory.steps):
            raise ValueError("Invalid diagnosis step index")
        if (trajectory.reward >= 1.0) != successful:
            raise ValueError("Diagnosis success/failure citation contradicts reward")
        actual_agent = trajectory.steps[step].get("agent") or trajectory.steps[step].get("agent_id")
        if agent != "unknown" and actual_agent != agent:
            raise ValueError("Diagnosis cites another agent's step")
        entry = {"rollout": rollout, "step": step}
        if entry not in result:
            result.append(entry)
    return result


def parse_diagnoses(data, group, policies):
    diagnoses = data.get("failure_analysis")
    if not isinstance(diagnoses, list):
        raise ValueError("failure_analysis is missing or is not an array")
    if len(diagnoses) > 6:
        raise ValueError("failure_analysis exceeds 6 entries")
    result = []
    for entry in diagnoses:
        if not isinstance(entry, dict):
            raise ValueError("Diagnosis must be an object")
        agent, sid = entry.get("agent"), entry.get("skill_id")
        kind, action = entry.get("failure_type"), entry.get("action")
        if not isinstance(agent, str) or (agent not in policies and agent != "unknown"):
            raise ValueError("Unknown diagnosis agent")
        if not isinstance(kind, str) or kind not in FAILURE_TYPES or action not in ("refine", "induce", "none"):
            raise ValueError("Unknown failure type or update action")
        if (agent == "unknown" or kind in ("tool_failure", "uncertain")) and action != "none":
            raise ValueError("Uncertain/tool failures cannot automatically update skills")
        if sid is not None and (not isinstance(sid, str) or agent == "unknown"
                                or policies[agent].skill_library.get(sid) is None):
            raise ValueError("Unknown diagnosis skill")
        if (action == "refine" and sid is None) or (action == "induce" and sid is not None):
            raise ValueError("Refine needs an existing skill; induce needs skill_id=null")
        locations = _locations(entry.get("failure_locations"), group, agent, False)
        contrast = _locations(entry.get("successful_contrast"), group, agent, True)
        if sid is not None:
            for ref in locations:
                trace = group.trajectories[ref["rollout"]].skill_trace.get(agent, [])
                if not any(sid in item.get("skills", []) for item in trace):
                    raise ValueError("Attributed skill absent from failed rollout trace")
        for key in ("evidence", "proposed_fix"):
            if not isinstance(entry.get(key), str) or (key == "evidence" or action != "none") and not entry[key].strip():
                raise ValueError(f"Diagnosis requires {key}")
        result.append({"agent": agent, "skill_id": sid, "failure_type": kind,
                       "action": action, "failure_locations": locations,
                       "successful_contrast": contrast, "evidence": entry["evidence"].strip(),
                       "proposed_fix": entry["proposed_fix"].strip()})
    return result


def diagnosis_text(diagnosis, category):
    return json.dumps({"category": category, **diagnosis,
                       "fix_status": "observational_contrast" if diagnosis["successful_contrast"]
                       else "unverified_hypothesis"}, ensure_ascii=False)


class StructuredSkillCreditCritic(GroupSkillCreditCritic):
    analysis_instructions = STRUCTURED_INSTRUCTIONS
    category_analysis = False

    def _analysis_payload(self, group):
        return {"task_category": task_category(group.trajectories[0].task)
                if self.category_analysis else "not_grouped",
                "citation_constraints": {
                    "non_full_reward_rollouts": [i for i, t in enumerate(group.trajectories) if t.reward < 1],
                    "full_reward_rollouts": [i for i, t in enumerate(group.trajectories) if t.reward >= 1],
                    "steps": [{"rollout": i, "step_index": j,
                               "agent": step.get("agent") or step.get("agent_id")}
                              for i, t in enumerate(group.trajectories) for j, step in enumerate(t.steps)]}}

    def _request_feedback(self, prompt, group, policies, observed):
        # All state is local: critic calls for different groups run concurrently.
        attempts, accepted, sources = [], [], []
        first_feedback = None
        for attempt in range(2):
            request = prompt
            if attempt:
                request += "\n\nREPAIR REQUEST (one attempt only):\n" + json.dumps({
                    "previous_raw_response": attempts[0]["raw_response"],
                    "validation_errors": attempts[0]["issues"],
                    "already_accepted_diagnoses": accepted,
                    "instruction": "Return the complete original JSON schema. Preserve accepted diagnoses; "
                    "repair or omit rejected ones using the actual evidence above. Never just delete an "
                    "incorrect success reference while retaining an unsupported explanation. "
                    "For uncertain/tool failures use action=none. No supported diagnosis means an empty array."
                }, ensure_ascii=False)
            raw = None
            try:
                raw = self._call_llm(request, max_tokens=4000)
            except Exception as exc:
                issues = [{"error": f"LLM call failed: {type(exc).__name__}: {exc}"}]
                attempts.append({"raw_response": raw, "issues": issues})
                break  # Transport errors are not semantic repair requests.
            try:
                fb = self._decode_feedback(raw, group, policies, observed)
                issues = list(fb.validation_audit["issues"])
                if attempt == 0:
                    first_feedback = fb
                for d in fb.failure_analysis:
                    if d not in accepted and len(accepted) < 6:
                        accepted.append(d)
                        sources.append((d, raw))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                fb = None
                issues = [{"error": f"Response invalid: {exc}"}]
            attempts.append({"raw_response": raw, "issues": issues})
            if not issues:
                break

        clean = not attempts[-1]["issues"]
        if len(attempts) == 1 and clean:
            result = first_feedback
        else:
            # Free-text skill credits have no machine-checkable citation schema.
            # After a malformed response, retain only credits linked to validated
            # refinement diagnoses; don't let rejected evidence change utility/pruning.
            credits = {}
            for d, raw in sources:
                if d["action"] != "refine":
                    continue
                key = (d["agent"], d["skill_id"])
                for c in self._parse(raw, policies, group.mean_reward).skill_credits:
                    if (c.agent, c.skill_id) == key and key not in credits:
                        credits[key] = c
            result = self._finish_feedback(
                GroupFeedback(credit=TrajectorySkillCredit(skill_credits=list(credits.values()),
                                                           reward=group.mean_reward)),
                {"failure_analysis": accepted}, group, policies)
        result.validation_audit = {
            "status": ("valid" if len(attempts) == 1 else "repaired") if clean
                      else ("partial" if accepted else "invalid"),
            "repair_attempted": len(attempts) == 2,
            "repair_succeeded": len(attempts) == 2 and clean,
            "accepted_diagnoses": len(accepted),
            "attempts": attempts,
            "standalone_skill_credits_withheld": len(attempts) > 1 or not clean,
        }
        return result

    def _finish_feedback(self, feedback, data, group, policies):
        category = self._analysis_payload(group)["task_category"]
        diagnoses, issues = [], []
        entries = data.get("failure_analysis")
        if not isinstance(entries, list):
            issues.append({"index": None, "error": "failure_analysis is missing or is not an array"})
            entries = []
        credit_keys = {(c.agent, c.skill_id) for c in feedback.credit.skill_credits}
        for index, entry in enumerate(entries):
            try:
                d = parse_diagnoses({"failure_analysis": [entry]}, group, policies)[0]
                if d["action"] == "refine" and (d["agent"], d["skill_id"]) not in credit_keys:
                    raise ValueError("Refinement diagnosis requires a matching skill credit")
                if d in diagnoses:
                    continue
                if len(diagnoses) >= 6:
                    raise ValueError("failure_analysis exceeds 6 distinct valid entries")
                diagnoses.append(d)
            except (ValueError, TypeError, KeyError) as exc:
                issues.append({"index": index, "error": str(exc), "entry": entry})
        # Only grounded diagnoses drive failure-related edits/induction in this
        # variant. Original contribution and utility scores remain unchanged.
        credits = {(c.agent, c.skill_id): c for c in feedback.credit.skill_credits}
        for c in credits.values():
            c.evidence = f"[category={category}] {c.evidence}"
            c.suggested_edit = ""
        feedback.credit.residuals = {}
        evidence = defaultdict(list)
        for d in diagnoses:
            text = diagnosis_text(d, category)
            if d["action"] == "refine":
                key = (d["agent"], d["skill_id"])
                if key not in credits:
                    continue
                credits[key].evidence += "\n" + text
                credits[key].suggested_edit += "\n" + d["proposed_fix"]
            elif d["action"] == "induce":
                evidence[d["agent"]].append(text)
        for agent, lines in evidence.items():
            feedback.credit.residuals[agent] = ResidualCredit(
                agent=agent, summary="\n".join(lines), needs_new_skill=True,
                missing_capability="See grounded diagnoses")
        feedback.credit.skill_credits = list(credits.values())
        return StructuredFeedback(credit=feedback.credit, induction_evidence=dict(evidence),
                                  category=category, failure_analysis=diagnoses,
                                  validation_audit={"issues": issues})


def summarize_failures(groups, selected, feedbacks, use_categories):
    """Count distinct questions, using ALL train groups for category scores.

    Diagnosis frequencies use successfully critiqued groups, not repeated
    rollouts. Induction evidence is ranked by repeated cross-question support.
    No validation/test feedback enters this function.
    """
    def category_of(group):
        return task_category(group.trajectories[0].task) if use_categories else "not_grouped"
    buckets = defaultdict(list)
    for group in groups:
        buckets[category_of(group)].append(group)
    report = {"categories": {}, "patterns": []}
    analyzed = {g.group_id for g in selected}
    for category, items in sorted(buckets.items()):
        report["categories"][category] = {
            "questions": len(items), "critic_questions": sum(g.group_id in analyzed for g in items),
            "mean_reward": fmean(g.mean_reward for g in items),
            "non_full_reward_questions": sum(any(t.reward < 1 for t in g.trajectories) for g in items),
        }
    patterns = defaultdict(dict)
    skill_scores = defaultdict(dict)
    for group, feedback in zip(selected, feedbacks):
        category = category_of(group)
        for c in feedback.credit.skill_credits:
            skill_scores[(category, c.agent, c.skill_id)][group.group_id] = c.utility_delta
        for d in feedback.failure_analysis:
            key = (category, d["agent"], d["failure_type"], d["skill_id"], d["action"])
            patterns[key].setdefault(group.group_id, []).append(d)
    report["skill_credits"] = [
        {"category": k[0], "agent": k[1], "skill_id": k[2], "questions": len(v),
         "mean_utility": fmean(v.values())} for k, v in sorted(skill_scores.items())]
    induction = defaultdict(list)
    for key, support in sorted(patterns.items(), key=lambda item: (-len(item[1]), str(item[0]))):
        category, agent, kind, sid, action = key
        examples = [{"group_id": gid, **d} for gid, rows in support.items() for d in rows]
        pattern = {"category": category, "agent": agent, "failure_type": kind,
                   "skill_id": sid, "action": action, "questions": len(support),
                   "category_questions": report["categories"][category]["questions"],
                   "examples": examples}
        report["patterns"].append(pattern)
        if action == "induce":
            # One evidence slot per pattern: repeated rollouts never crowd out
            # other categories in the optimizer's 12-item evidence budget.
            induction[agent].append(json.dumps({**pattern, "examples": examples[:2],
                                                "note": "Grouped by failure type; examples may have different causes. "
                                                "Preserve applicability; no contrast means an unverified fix."},
                                               ensure_ascii=False))
    return report, dict(induction)
