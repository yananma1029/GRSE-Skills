import json
import logging
from copy import deepcopy
from pathlib import Path

import pytest

from GRSE.config import GroupConfig
from GRSE.demo import DemoEnvironment, DemoOptimizer
from GRSE.failure_analysis import (
    StructuredSkillCreditCritic,
    parse_diagnoses,
    summarize_failures,
    task_category,
)
from GRSE.group import RolloutGroup
from GRSE.store import GroupRunStore
from GRSE.trainer import GroupSkillTrainer
from maskills.config.base import LanguageTaskConfig
from maskills.core.base import Trajectory
from maskills.core.policy import AgentPolicy
from maskills.core.skills import Skill, SkillLibrary


def policies(with_skill=False):
    return {agent: AgentPolicy(role="Check evidence", skill_library=SkillLibrary(
        [Skill(name="verify", description="Check facts", body="Verify evidence")] if with_skill else []))
        for agent in ("agent_1", "agent_2")}


def group(gid="q", category="bridge", rewards=(0, 1)):
    task = {"question": gid, "metadata": {"type": category}}
    return RolloutGroup.build(gid, [Trajectory(
        task, [{"agent": "agent_1", "output": "claim"}, {"agent": "agent_2", "output": "answer"}],
        reward, skill_trace={"agent_1": [{"skills": ["verify"]}]}) for reward in rewards])


def diagnosis(**overrides):
    return {"agent": "agent_1", "skill_id": None, "failure_type": "missing_capability",
            "failure_locations": [{"rollout": 0, "step": 0}],
            "successful_contrast": [{"rollout": 1, "step": 0}],
            "evidence": "Failure handed off unchecked evidence; success verified it.",
            "proposed_fix": "Verify evidence before handoff.", "action": "induce", **overrides}


def critic(response, categories=True):
    obj = StructuredSkillCreditCritic.__new__(StructuredSkillCreditCritic)
    obj.tool_library = ""
    obj.logger = logging.getLogger("structured-test")
    obj.category_analysis = categories
    obj._call_llm = lambda *args, **kwargs: json.dumps(response)
    return obj


def feedback(g, entries, p=None):
    credits = [{"agent": d["agent"], "skill_id": d["skill_id"],
                "contribution": "neutral", "utility_delta": 0.2}
               for d in entries if d["action"] == "refine"]
    response = {"skills": credits, "failure_analysis": entries,
                "residual": {"agent_2": {"needs_new_skill": True, "summary": "invented"}},
                "induction_evidence": {"agent_2": ["ungrounded suggestion"]}}
    return critic(response).evaluate_group_skill_credit(g, p or policies())


@pytest.mark.parametrize("task, expected", [
    ({"type": "qa", "metadata": {"type": "bridge"}}, "bridge"),
    ({"category": 2}, "2"), ({"metadata": {"category": 3}}, "3"),
    ({"type": "qa"}, "unknown"), ({"category": None}, "unknown"),
])
def test_dataset_labels_only(task, expected):
    assert task_category(task) == expected


@pytest.mark.parametrize("change", [
    {"agent": "invented"}, {"failure_type": "made_up"}, {"skill_id": "missing"},
    {"failure_locations": [{"rollout": 4, "step": 0}]},
    {"failure_locations": [{"rollout": 0, "step": 1}]},
    {"failure_locations": [{"rollout": True, "step": 0}]},
    {"failure_locations": [{"rollout": 0, "step": 20}]},
    {"failure_locations": [{"rollout": 1, "step": 0}]},
    {"successful_contrast": [{"rollout": 0, "step": 0}]},
    {"action": "refine"}, {"failure_type": "tool_failure"},
    {"failure_type": "uncertain"}, {"evidence": ""},
])
def test_reject_ungrounded_diagnoses(change):
    with pytest.raises(ValueError):
        parse_diagnoses({"failure_analysis": [diagnosis(**change)]}, group(), policies())


def test_positive_advantage_is_not_success():
    with pytest.raises(ValueError, match="contradicts reward"):
        parse_diagnoses({"failure_analysis": [diagnosis()]}, group(rewards=(0, 0.8)), policies())


def test_all_failed_retains_hypothesis_and_only_grounded_induction():
    fb = feedback(group(rewards=(0, 0)), [diagnosis(successful_contrast=[])])
    assert "unverified_hypothesis" in fb.induction_evidence["agent_1"][0]
    assert "agent_2" not in fb.induction_evidence
    assert "agent_2" not in fb.credit.residuals


def test_full_success_and_tool_failures_do_not_induce():
    assert feedback(group(rewards=(1, 1)), []).induction_evidence == {}
    fb = feedback(group(), [diagnosis(failure_type="tool_failure", action="none", proposed_fix="")])
    assert fb.induction_evidence == {} and fb.credit.residuals == {}


def test_refine_preserves_citation_and_no_new_skill():
    fb = feedback(group(), [diagnosis(action="refine", skill_id="verify", failure_type="skill_defect")], policies(True))
    assert not fb.induction_evidence
    assert len(fb.credit.skill_credits) == 1
    assert fb.credit.skill_credits[0].utility_delta == 0.2
    assert fb.credit.skill_credits[0].suggested_edit.strip() == "Verify evidence before handoff."
    assert '"category": "bridge"' in fb.credit.skill_credits[0].evidence
    missing = group()
    missing.trajectories[0].skill_trace = {}
    rejected = feedback(missing, [diagnosis(action="refine", skill_id="verify")], policies(True))
    assert rejected.validation_audit["status"] == "invalid"
    assert not rejected.credit.skill_credits


def test_missing_schema_is_not_silently_accepted():
    fb = critic({"skills": []}).evaluate_group_skill_credit(group(), policies())
    assert fb.validation_audit["status"] == "invalid"
    assert len(fb.validation_audit["attempts"]) == 2
    assert "missing" in fb.validation_audit["attempts"][0]["issues"][0]["error"]


def test_distinct_questions_full_denominator_and_category_separation():
    groups = [group("a"), group("b"), group("c", "comparison"), group("d", rewards=(1, 1))]
    feedbacks = [feedback(g, [diagnosis(), diagnosis()]) for g in groups[:3]]
    report, evidence = summarize_failures(groups, groups[:3], feedbacks, True)
    assert report["categories"]["bridge"]["questions"] == 3
    assert report["categories"]["bridge"]["critic_questions"] == 2
    assert report["categories"]["bridge"]["mean_reward"] == pytest.approx(2 / 3)
    assert [r["questions"] for r in report["patterns"]] == [2, 1]
    assert len(evidence["agent_1"]) == 2
    assert json.loads(evidence["agent_1"][0])["category"] == "bridge"
    ablation, _ = summarize_failures(groups, groups[:3], feedbacks, False)
    assert list(ablation["categories"]) == ["not_grouped"]
    assert ablation["patterns"][0]["questions"] == 3


def test_variant_configs_only_change_analysis_and_name():
    root = Path(__file__).resolve().parents[2] / "configs" / "grse"
    original = json.loads((root / "hotpotqa" / "grse_wo_structure.json").read_text())
    for name, categories in [("hotpotqa/grse", False), ("ablation/hotpotqa_with_label", True)]:
        new = json.loads((root / (name + ".json")).read_text())
        assert new["grpo"].pop("structured_failures") is True
        assert new["grpo"].pop("category_analysis") is categories
        new["base"]["exp_name"] = original["base"]["exp_name"]
        assert new == original
    with pytest.raises(ValueError, match="requires"):
        GroupConfig(category_analysis=True)


def test_training_passes_categorized_evidence_to_optimizer(tmp_path):
    class Env(DemoEnvironment):
        def sample_tasks(self, n, seed=None, split="train"):
            tasks = super().sample_tasks(n, seed, split)
            for task in tasks:
                task["metadata"] = {"type": "bridge" if split == "train" else "held_out"}
            return tasks

    c = critic({})
    def respond(prompt, **kwargs):
        payload = json.loads(prompt.split("\n\nINPUT:\n")[1])
        rows = payload["rollouts"]
        failures = [r["rollout_index"] for r in rows if r["reward"] < 1]
        successes = [r["rollout_index"] for r in rows if r["reward"] >= 1]
        d = diagnosis(failure_locations=[{"rollout": failures[0], "step": 0}],
                      successful_contrast=[{"rollout": successes[0], "step": 0}] if successes else [])
        return json.dumps({"skills": [], "failure_analysis": [d]})
    c._call_llm = respond
    optimizer = DemoOptimizer()
    seen = []
    original = optimizer.induce_skill
    def induce(evidence, library, residual=""):
        seen.append(deepcopy(evidence))
        return original(evidence, library, residual)
    optimizer.induce_skill = induce
    config = LanguageTaskConfig(exp_name="structured_test", experiment_dir=str(tmp_path),
                                num_iterations=1, trajectories_per_iteration=2,
                                n_val=2, n_test=0, max_workers=1, optimizer_workers=1,
                                induct_every=1, consolidate_every=0, prune_every=0)
    t = GroupSkillTrainer(config=config, env=Env(), critic=c, optimizer=optimizer,
                          group_config=GroupConfig(structured_failures=True, category_analysis=True),
                          store=GroupRunStore(str(tmp_path)))
    t.train_one_iteration(1)
    report = json.loads((t.artifact_root / "iter_1/failure_analysis.json").read_text())
    assert list(report["categories"]) == ["bridge"]
    assert report["patterns"][0]["questions"] == 2
    assert len(seen) == 1  # No unsupported induction for agent_2.
    assert json.loads(seen[0][0])["questions"] == 2
    saved = json.loads(next((t.artifact_root / "iter_1").glob("credit_*.json")).read_text())
    assert saved["credit"]["failure_analysis"][0]["agent"] == "agent_1"
    assert t.store.latest_checkpoint(t.run_id) == 1


def test_original_run_resume_and_cached_rollouts_remain_compatible(tmp_path):
    from GRSE.demo import DemoCritic
    config = LanguageTaskConfig(exp_name="resume", experiment_dir=str(tmp_path), n_val=1,
                                n_test=0, trajectories_per_iteration=1, max_workers=1)
    options = GroupConfig()
    env = DemoEnvironment()
    t = GroupSkillTrainer(config=config, env=env, critic=DemoCritic(), optimizer=DemoOptimizer(),
                          group_config=options, store=GroupRunStore(str(tmp_path)))
    p = t.checkpoint.get_policies()
    t.checkpoint.save_policies(0, p, {})
    groups = t._collect_groups(p, 1)
    option_path = t.artifact_root / "options.json"
    old = json.loads(option_path.read_text())
    del old["structured_failures"], old["category_analysis"]
    option_path.write_text(json.dumps(old))
    # cache_options also leaves out split_seed and single_rollout at their defaults
    assert options.cache_options() == {k: v for k, v in old.items() if k not in ("split_seed", "single_rollout")}
    resumed = GroupSkillTrainer(config=config, env=env, critic=DemoCritic(), optimizer=DemoOptimizer(),
                                group_config=options, store=GroupRunStore(str(tmp_path), t.run_id))
    before = dict(env.calls)
    assert resumed._collect_groups(p, 1)[0].group_id == groups[0].group_id
    assert dict(env.calls) == before
    with pytest.raises(ValueError, match="unchanged"):
        GroupSkillTrainer(config=config, env=env, critic=DemoCritic(), optimizer=DemoOptimizer(),
                          group_config=GroupConfig(structured_failures=True),
                          store=GroupRunStore(str(tmp_path), t.run_id))


@pytest.mark.parametrize("categories", [False, True])
def test_cli_selects_structured_critic(tmp_path, monkeypatch, categories):
    from GRSE import failure_analysis, train
    from maskills.envs.language import metrics

    c = critic({"skills": [], "failure_analysis": []})
    monkeypatch.setattr(failure_analysis, "StructuredSkillCreditCritic", lambda *a, **kw: c)
    monkeypatch.setattr(train, "make_env", lambda *a: DemoEnvironment())
    monkeypatch.setattr(train, "GroupSkillOptimizer", lambda *a, **kw: DemoOptimizer())
    monkeypatch.setattr(metrics, "compute_qa_metrics", lambda *a: {"f1": 1})
    class Client:
        def __init__(self, *args):
            self.temperature = None
    monkeypatch.setattr(train, "SamplingLLMClient", Client)
    config = {"env": "language", "base": {"llm": "gpt-4o-mini", "num_iterations": 1,
              "trajectories_per_iteration": 1, "n_val": 1, "n_test": 0,
              "max_workers": 1, "optimizer_workers": 1},
              "grpo": {"structured_failures": True, "category_analysis": categories}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    rows = train.main(["--config", str(path), "--output", str(tmp_path / "out")])
    assert c.category_analysis is categories
    assert rows[0]["type"] == "baseline"
    report_path = next((tmp_path / "out").glob("runs/*/grpo/iter_1/failure_analysis.json"))
    report = json.loads(report_path.read_text())
    assert list(report["categories"]) == ["unknown" if categories else "not_grouped"]


def run_responses(responses, g=None, p=None):
    c, calls = critic({}), []
    def call(prompt, **kwargs):
        calls.append(prompt)
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)
    c._call_llm = call
    fb = c.evaluate_group_skill_credit(g or group(), p or policies())
    return fb, calls


def test_valid_response_needs_no_repair_and_explicit_constraints():
    fb, calls = run_responses([{"skills": [], "failure_analysis": [diagnosis()]}])
    assert len(calls) == 1 and fb.validation_audit["status"] == "valid"
    payload = json.loads(calls[0].split("\n\nINPUT:\n")[1])
    constraints = payload["citation_constraints"]
    assert constraints["full_reward_rollouts"] == [1]
    assert constraints["non_full_reward_rollouts"] == [0]
    assert constraints["steps"][1] == {"rollout": 0, "step_index": 1, "agent": "agent_2"}


def test_reward_conflict_repaired_once_with_original_evidence():
    bad = diagnosis(successful_contrast=[{"rollout": 0, "step": 0}])
    fb, calls = run_responses([{"skills": [], "failure_analysis": [bad]},
                              {"skills": [], "failure_analysis": [diagnosis()]}])
    assert fb.validation_audit["status"] == "repaired"
    assert len(calls) == 2 and fb.failure_analysis == [diagnosis()]
    assert "previous_raw_response" in calls[1] and "contradicts reward" in calls[1]
    assert '"output": "claim"' in calls[1]  # Actual trajectory evidence is resent.
    assert fb.validation_audit["attempts"][0]["raw_response"]


@pytest.mark.parametrize("retry", ["not JSON", RuntimeError("timeout")])
def test_partial_survives_failed_repair_without_unsafe_skill_scores(retry):
    bad = diagnosis(successful_contrast=[{"rollout": 0, "step": 0}])
    response = {"skills": [{"agent": "agent_1", "skill_id": "verify", "contribution": "harmful",
                            "utility_delta": -1, "evidence": "Unverified blame"}],
                "failure_analysis": [diagnosis(), bad]}
    fb, calls = run_responses([response, retry], p=policies(True))
    assert len(calls) == 2 and fb.validation_audit["status"] == "partial"
    assert fb.failure_analysis == [diagnosis()]
    assert fb.induction_evidence["agent_1"]
    assert not fb.credit.skill_credits


def test_repair_cannot_overwrite_already_accepted_diagnosis():
    bad = diagnosis(failure_type="tool_failure")
    fb, _ = run_responses([{"skills": [], "failure_analysis": [diagnosis(), bad]},
                           {"skills": [], "failure_analysis": []}])
    assert fb.failure_analysis == [diagnosis()]


def test_tool_failure_can_be_repaired_to_observation_only():
    fb, _ = run_responses([{"skills": [], "failure_analysis": [diagnosis(failure_type="tool_failure")]},
                           {"skills": [], "failure_analysis": [diagnosis(failure_type="tool_failure", action="none")]}])
    assert fb.validation_audit["status"] == "repaired"
    assert fb.failure_analysis[0]["action"] == "none"
    assert not fb.induction_evidence


def test_empty_valid_diagnosis_is_distinct_from_invalid_output():
    valid, _ = run_responses([{"skills": [], "failure_analysis": []}])
    invalid, _ = run_responses([{"skills": []}, {"skills": []}])
    assert not valid.failure_analysis and valid.validation_audit["status"] == "valid"
    assert not invalid.failure_analysis and invalid.validation_audit["status"] == "invalid"


def test_over_limit_keeps_valid_entries_and_logs_specific_error():
    entries = [diagnosis(evidence=f"Evidence {i}") for i in range(7)]
    fb, _ = run_responses([{"skills": [], "failure_analysis": entries}, "invalid JSON"])
    assert len(fb.failure_analysis) == 6 and fb.validation_audit["status"] == "partial"
    assert "exceeds" in fb.validation_audit["attempts"][0]["issues"][0]["error"]


def test_initial_transport_failure_is_not_semantically_retried():
    fb, calls = run_responses([RuntimeError("API unavailable")])
    assert len(calls) == 1 and fb.validation_audit["status"] == "invalid"
    assert fb.validation_audit["attempts"][0]["raw_response"] is None


def test_all_invalid_groups_persist_audit_and_coverage_before_stopping(tmp_path):
    c = critic({"skills": []})
    config = LanguageTaskConfig(exp_name="invalid", experiment_dir=str(tmp_path),
                                trajectories_per_iteration=2, n_val=1, n_test=0,
                                max_workers=1, optimizer_workers=1)
    t = GroupSkillTrainer(config=config, env=DemoEnvironment(), critic=c, optimizer=DemoOptimizer(),
                          group_config=GroupConfig(structured_failures=True),
                          store=GroupRunStore(str(tmp_path)))
    with pytest.raises(ValueError, match="No valid structured"):
        t.train_one_iteration(1)
    root = t.artifact_root / "iter_1"
    coverage = json.loads((root / "critic_coverage.json").read_text())
    assert coverage["critic_requested_groups"] == coverage["critic_invalid_groups"] == 2
    assert coverage["critic_repair_attempts"] == 2
    assert coverage["critic_no_diagnosis_groups"] == 0
    assert t.store.latest_checkpoint(t.run_id) is None
    audit = json.loads(next(root.glob("credit_*.json")).read_text())["credit"]["validation_audit"]
    assert len(audit["attempts"]) == 2 and audit["attempts"][0]["raw_response"]


def test_valid_refinement_credit_survives_unrelated_failed_repair():
    good = diagnosis(action="refine", skill_id="verify", failure_type="skill_defect")
    bad = diagnosis(successful_contrast=[{"rollout": 0, "step": 0}])
    fb, _ = run_responses([
        {"skills": [{"agent": "agent_1", "skill_id": "verify", "contribution": "neutral",
                     "utility_delta": 0.2, "evidence": "Grounded verification issue"}],
         "failure_analysis": [good, bad]}, RuntimeError("timeout")], p=policies(True))
    assert fb.validation_audit["status"] == "partial"
    assert fb.failure_analysis == [good]
    assert len(fb.credit.skill_credits) == 1
    assert fb.credit.skill_credits[0].utility_delta == 0.2
    assert fb.credit.skill_credits[0].suggested_edit.strip() == good["proposed_fix"]
    assert not fb.induction_evidence
