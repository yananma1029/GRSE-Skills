import json
from dataclasses import replace
from pathlib import Path

import pytest

from GRSE.config import GroupConfig
from GRSE.demo import DemoCritic, DemoEnvironment, DemoOptimizer, DemoSelector
from GRSE.group import RolloutGroup, relative_advantages
from GRSE.routing import LLMSkillSelector, SkillRoutingEnvironment
from GRSE.store import GroupRunStore
from GRSE.trainer import GroupSkillTrainer
from maskills.config.base import LanguageTaskConfig
from maskills.core.base import Trajectory
from maskills.core.policy import AgentPolicy
from maskills.core.skills import Skill, SkillLibrary


def policy(*ids):
    return AgentPolicy(role="Check evidence", skill_library=SkillLibrary([
        Skill(name=sid, description="Evidence procedure", body=f"Instructions for {sid}")
        for sid in ids
    ]))


def trainer(tmp_path, env=None, options=None, **overrides):
    config = LanguageTaskConfig(
        exp_name="test_grpo", experiment_dir=str(tmp_path), skill_evolution=True,
        num_iterations=2, trajectories_per_iteration=2, n_val=2, n_test=0,
        max_workers=1, optimizer_workers=1, induct_every=2,
        consolidate_every=0, prune_every=0, **overrides,
    )
    return GroupSkillTrainer(
        config=config,
        env=SkillRoutingEnvironment(env or DemoEnvironment(), DemoSelector()),
        critic=DemoCritic(), optimizer=DemoOptimizer(),
        group_config=options or GroupConfig(validation_rollouts=4),
        store=GroupRunStore(str(tmp_path)),
    )


def test_relative_advantages_and_zero_variance():
    assert relative_advantages([1, 0, 1, 0]) == pytest.approx([1, -1, 1, -1])
    assert relative_advantages([0, 0, 0, 0]) == [0] * 4
    assert relative_advantages([1, 1, 1, 1]) == [0] * 4
    # Relative success is not absolute correctness.
    assert relative_advantages([0.1, 0.2])[1] > 0


@pytest.mark.parametrize("rewards", [[], [float("nan"), 1], [1, float("inf")]])
def test_bad_rewards_rejected(rewards):
    with pytest.raises(ValueError):
        relative_advantages(rewards)


def test_single_rollout_has_no_contrast():
    # G=1 ablation: one rollout per question gives a zero advantage
    assert relative_advantages([1.0]) == [0.0]


def test_group_rejects_different_questions():
    with pytest.raises(ValueError, match="same task"):
        RolloutGroup.build("q", [Trajectory({"q": 1}, [], 0), Trajectory({"q": 2}, [], 1)])


def test_selector_only_sees_index_and_question():
    class Client:
        def chat(self, system, payload, max_tokens):
            data = json.loads(payload)
            assert data["question"] == "Which entity?"
            assert "body" not in data["skills"][0]
            assert "ground_truth" not in data
            return '{"skill_ids": ["lookup"]}'
    env = SkillRoutingEnvironment(DemoEnvironment(), LLMSkillSelector(Client()))
    p = {"agent_1": policy("lookup", "format")}
    traj = env.collect_trajectory(p, {"question": "Which entity?", "task_id": "q", "ground_truth": "SECRET"})
    assert traj.metadata["selected_skills"] == {"agent_1": ["lookup"]}
    assert traj.skill_trace["agent_1"][0]["skills"] == ["lookup"]
    assert p["agent_1"].skill_library.ids() == ["lookup", "format"]


def test_selection_loads_only_selected_bodies_and_empty_means_none():
    class Selector:
        def select(self, agent, policy, question):
            return ["keep"] if agent == "agent_1" else []
    class Env(DemoEnvironment):
        def collect_trajectory(self, policies, task):
            assert policies["agent_1"].skill_library.ids() == ["keep"]
            assert "secret body" not in policies["agent_1"].combined
            assert not policies["agent_2"].skill_library
            return super().collect_trajectory(policies, task)
    p = {"agent_1": policy("keep", "secret body"), "agent_2": policy("keep")}
    traj = SkillRoutingEnvironment(Env(), Selector()).collect_trajectory(p, {"task_id": "q"})
    assert traj.skill_trace["agent_2"][0]["skills"] == []


def test_selector_unknown_skill_fails_instead_of_exposing_all():
    class Selector:
        def select(self, *args):
            return ["invented"]
    with pytest.raises(ValueError, match="unavailable"):
        SkillRoutingEnvironment(DemoEnvironment(), Selector()).collect_trajectory(
            {"agent_1": policy("real")}, {"task_id": "q"})


def test_group_critic_filters_unobserved_unknown_and_duplicate_skills():
    critic = DemoCritic()
    raw = {"skills": [
        {"agent": "agent_1", "skill_id": sid, "utility_delta": 0.5}
        for sid in ("used", "used", "unselected", "invented")
    ], "residual": {}, "induction_evidence": {"unknown_agent": ["ignore"]}}
    critic._call_llm = lambda *args, **kwargs: json.dumps(raw)
    trajectories = [Trajectory({"question": "q"}, [], r, skill_trace={
        "agent_1": [{"step": 0, "skills": ["used"]}]
    }) for r in (0, 1)]
    feedback = critic.evaluate_group_skill_credit(
        RolloutGroup.build("q", trajectories), {"agent_1": policy("used", "unselected")})
    assert [c.skill_id for c in feedback.credit.skill_credits] == ["used"]
    assert feedback.induction_evidence == {}


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_empty_library_keeps_induction_signal_when_all_rewards_equal(reward):
    trajectories = [Trajectory({"question": "q"}, [], reward) for _ in range(4)]
    group = RolloutGroup.build("q", trajectories)
    feedback = DemoCritic().evaluate_group_skill_credit(group, {"agent_1": policy()})
    assert group.advantages == [0] * 4
    assert feedback.credit.skill_credits == []
    assert feedback.induction_evidence["agent_1"]


def test_team_groups_preserve_task_and_agent_execution(tmp_path):
    t = trainer(tmp_path)
    groups = t._collect_groups({"agent_1": policy(), "agent_2": policy()}, 1)
    assert len(groups) == 2
    for group in groups:
        assert len(group.trajectories) == 4
        assert [tr.reward for tr in group.trajectories] == [1, 0, 0, 0]
        assert all([s["agent"] for s in tr.steps] == ["agent_1", "agent_2"] for tr in group.trajectories)


def test_partial_cache_reuses_completed_rollouts_and_rejects_changed_policy(tmp_path):
    t = trainer(tmp_path)
    policies = {"agent_1": policy(), "agent_2": policy()}
    groups = t._collect_groups(policies, 1)
    total_calls = sum(t.env.env.calls.values())
    t._collect_groups(policies, 1)
    assert sum(t.env.env.calls.values()) == total_calls
    run_dir = Path(t.store.get_run_dir(t.run_id))
    files = list(run_dir.rglob("episode_000.json"))
    assert len(files) == 1
    files[0].unlink()
    recovered = t._collect_groups(policies, 1)
    assert sum(t.env.env.calls.values()) == total_calls + 1
    assert len(recovered) == len(groups)
    with pytest.raises(ValueError, match="different policies"):
        t._collect_groups({"agent_1": policy("changed"), "agent_2": policy()}, 1)


def test_end_to_end_empty_bootstrap_and_one_credit_vote_per_question(tmp_path):
    t = trainer(tmp_path, options=GroupConfig(bootstrap_empty=True, validation_rollouts=4))
    first = t.train_one_iteration(1)
    assert first["skill_induced"] == 2
    assert first["skill_library_size"] == {"agent_1": 1, "agent_2": 1}
    second = t.train_one_iteration(2)
    assert second["train_avg_reward"] == 1
    policies = t.checkpoint.get_policies()
    for p in policies.values():
        s = p.skill_library.get("verify-evidence")
        assert s.invocations == 8
        assert s.utility == pytest.approx(0.32)
    update = json.loads((t.artifact_root / "iter_2" / "updates.json").read_text())
    assert update["gradients"]["agent_1"]["verify-evidence"]["n_credits"] == 2
    _, meta = t.store.load_checkpoint(t.run_id, 2)
    assert "val_avg_reward" in meta["training_stats"]


def test_question_minibatch_keeps_whole_groups(tmp_path):
    t = trainer(tmp_path, options=GroupConfig(credit_question_batch_size=1, validation_rollouts=4))
    stats = t.train_one_iteration(1)
    assert stats["train_num_episodes"] == 8
    assert stats["grpo_credit_questions"] == 1
    credits = list((t.artifact_root / "iter_1").glob("credit_*.json"))
    assert len(credits) == 1
    assert len(json.loads(credits[0].read_text())["advantages"]) == 4


def test_sequential_gate_rejects_bad_joint_interaction(tmp_path):
    class InteractionEnv(DemoEnvironment):
        def collect_trajectory(self, policies, task):
            count = sum(bool(p.skill_library) for p in policies.values())
            return Trajectory(task, [], {0: 0.5, 1: 1.0, 2: 0.0}[count])
    t = trainer(tmp_path, env=InteractionEnv(),
                options=GroupConfig(validation_mode="sequential_team", validation_rollouts=4))
    base = {"agent_1": policy(), "agent_2": policy()}
    candidates = {a: policy("new") for a in base}
    accepted, gate = t._validation_gate(base, candidates)
    assert accepted["agent_1"].skill_library
    assert not accepted["agent_2"].skill_library
    assert gate["final_score"] == 1
    assert not gate["per_agent"]["agent_2"]["accepted"]


def test_no_validation_fails_closed(tmp_path):
    t = trainer(tmp_path)
    t.config.n_val = 0
    with pytest.raises(ValueError, match="No validation"):
        t._validation_gate({"agent_1": policy()}, {"agent_1": policy("new")})


def test_targeted_mask_isolated_and_does_not_mutate_original(tmp_path):
    t = trainer(tmp_path)
    policies = {"agent_1": policy("verify-evidence"), "agent_2": policy("verify-evidence")}
    report = t.evaluate_skill_ablation(policies, [{"task_id": "ablation"}],
                                     "agent_1", "verify-evidence", repeats=4)
    assert report["mean_delta"] > 0
    assert all(p.skill_library.get("verify-evidence") for p in policies.values())
    assert len(report["tasks"][0]["full_rewards"]) == 4


def test_resume_attaches_to_existing_run_and_rejects_option_changes(tmp_path):
    t = trainer(tmp_path)
    t.checkpoint.save_policies(0, t.checkpoint.get_policies(), {})
    t._collect_groups(t.checkpoint.get_policies(), 1)
    resumed = GroupSkillTrainer(
        config=t.config, env=t.env, critic=t.critic, optimizer=t.optimizer,
        group_config=t.group_config, store=GroupRunStore(str(tmp_path), t.run_id),
    )
    before = sum(t.env.env.calls.values())
    resumed._collect_groups(resumed.checkpoint.get_policies(), 1)
    assert sum(t.env.env.calls.values()) == before
    assert len(list((tmp_path / "runs").iterdir())) == 1
    with pytest.raises(ValueError, match="unchanged GRPO"):
        GroupSkillTrainer(config=t.config, env=t.env, critic=t.critic, optimizer=t.optimizer,
                          group_config=replace(t.group_config, rollouts_per_question=8),
                          store=GroupRunStore(str(tmp_path), t.run_id))


def test_critic_failure_does_not_commit_candidate(tmp_path):
    t = trainer(tmp_path)
    def fail(*args, **kwargs):
        raise ValueError("broken JSON")
    t.critic._call_llm = fail
    with pytest.raises(ValueError, match="broken JSON"):
        t.train_one_iteration(1)
    assert t.store.latest_checkpoint(t.run_id) is None
    assert t.trajectory_store.count(1) == 8


def test_one_failed_critic_group_is_skipped_not_fatal(tmp_path):
    t = trainer(tmp_path)
    original, calls = t.critic._call_llm, []
    def fail_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("context length exceeded")
        return original(*args, **kwargs)
    t.critic._call_llm = fail_first
    stats = t.train_one_iteration(1)
    assert stats["grpo_questions"] == 2
    assert stats["grpo_credit_questions"] == 1


def test_induced_skill_md_format_slips_are_repaired_or_skipped():
    from GRSE.optimizer import GroupSkillOptimizer, _normalize_skill_md
    from maskills.core.skills import parse_skill_md
    fenced = "```yaml\nname: Precise answers\ndescription: Keep it short\n---\n\n## Steps\nDo it."
    skill = parse_skill_md(_normalize_skill_md(fenced))
    assert skill.name == "Precise answers" and "Do it." in skill.body
    wrapped = "```markdown\n---\nname: A\ndescription: B\n---\nBody\n```"
    assert parse_skill_md(_normalize_skill_md(wrapped)).body == "Body"
    unclosed = "---\nname: A\ndescription: B\n## When to use\nAlways."
    assert parse_skill_md(_normalize_skill_md(unclosed)).body == "## When to use\nAlways."
    spaced = "---\nname: A\ndescription: B\n\n---\nBody"
    assert _normalize_skill_md(spaced) == spaced

    optimizer = GroupSkillOptimizer.__new__(GroupSkillOptimizer)
    optimizer.logger = __import__("logging").getLogger("test")
    optimizer._llm_call = lambda *args, **kwargs: "no frontmatter at all"
    assert optimizer.induce_skill(["evidence"], SkillLibrary([])) is None


def test_legacy_trajectory_minibatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="never split"):
        trainer(tmp_path, mini_batch_size=1)


def test_custom_reward_used_in_rollout_gate_and_final_report(tmp_path):
    class Reward:
        def compute(self, trajectory):
            return 0.375
    t = trainer(tmp_path)
    t.reward_fn = Reward()
    stats = t.train_one_iteration(1)
    assert stats["train_avg_reward"] == stats["val_avg_reward"] == 0.375
    updates = json.loads((t.artifact_root / "iter_1" / "updates.json").read_text())
    assert updates["gate"]["base_score"] == 0.375


def test_sampling_temperature_is_forwarded(monkeypatch):
    from types import SimpleNamespace

    from GRSE import client as module
    from maskills.config.llm import LLMConfig
    seen = []
    def request(fn, **params):
        seen.append(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"skill_ids": []}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
    monkeypatch.setattr(module, "_create_with_retry", request)
    client = module.SamplingLLMClient(
        LLMConfig(name="test", model_string="test", api_key="offline-test-key"), 0.8)
    client.chat("role", "question")
    assert seen[0]["temperature"] == 0.8
    assert client.usage == {"input": 10, "output": 5}
    client.temperature = None
    client.chat("role", "question")
    assert "temperature" not in seen[1]


def test_real_cli_wiring_with_offline_components(tmp_path, monkeypatch):
    from GRSE import train as entry
    from maskills.envs.language import metrics
    class Client:
        def __init__(self, config, temperature):
            self.temperature = temperature
            self.usage = {"input": 0, "output": 0}
        def chat(self, system, payload, max_tokens):
            data = json.loads(payload)
            return json.dumps({"skill_ids": [s["id"] for s in data["skills"]]})
    monkeypatch.setattr(entry, "make_env", lambda *args: DemoEnvironment())
    monkeypatch.setattr(entry, "SamplingLLMClient", Client)
    monkeypatch.setattr(entry, "LLMSkillSelector", lambda *args: pytest.fail("all mode must bypass routing"))
    monkeypatch.setattr(entry, "GroupSkillCreditCritic", lambda *args, **kwargs: DemoCritic())
    monkeypatch.setattr(entry, "GroupSkillOptimizer", lambda *args, **kwargs: DemoOptimizer())
    monkeypatch.setattr(metrics, "compute_qa_metrics", lambda *args: {"f1": 1.0})
    config = {"env": "language", "base": {
        "exp_name": "cli_test", "llm": "gpt-4o-mini", "num_iterations": 1,
        "induct_every": 1,
        "trajectories_per_iteration": 2, "n_val": 2, "n_test": 0,
        "max_workers": 1, "optimizer_workers": 1,
    }, "grpo": {"validation_rollouts": 4}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = ["--config", str(path), "--output", str(tmp_path / "output")]
    metrics = entry.main(args)
    assert metrics[0]["type"] == "baseline"
    assert metrics[-1]["skill_induced"] == 2
    run = next((tmp_path / "output" / "runs").iterdir())
    resumed_metrics = entry.main(args + ["--resume-run-id", run.name])
    assert len(resumed_metrics) == len(metrics)


@pytest.mark.parametrize("second_reward, expected", [(1.0, True), (0.48, True), (0.46, False)])
def test_independent_gate_matches_upstream(tmp_path, second_reward, expected):
    from maskills.trainer.skill_evolution import SkillEvolutionTrainer

    class InteractionEnv(DemoEnvironment):
        def collect_trajectory(self, policies, task):
            first = bool(policies["agent_1"].skill_library)
            second = bool(policies["agent_2"].skill_library)
            reward = 0.0 if first and second else (1.0 if first else second_reward if second else 0.5)
            return Trajectory(task, [], reward)

    t = trainer(tmp_path, env=InteractionEnv(), options=GroupConfig(), skill_eval_delta=0.03)
    base = {"agent_1": policy(), "agent_2": policy()}
    candidates = {agent: policy("new") for agent in base}
    accepted, gate = t._validation_gate(base, candidates)
    upstream, original_gate = SkillEvolutionTrainer._validation_gate(t, base, candidates)
    assert {a: p.to_dict() for a, p in accepted.items()} == {a: p.to_dict() for a, p in upstream.items()}
    assert gate["mode"] == "independent"
    assert "final_score" not in gate
    assert gate["per_agent"]["agent_2"]["base_score"] == 0.5
    assert gate["per_agent"]["agent_2"]["accepted"] is expected
    assert original_gate["per_agent"]["agent_2"]["accepted"] is expected
    assert not any(p.skill_library for p in base.values())


def test_aligned_gate_evaluates_once_per_task(tmp_path):
    class CountingEnv(DemoEnvironment):
        evaluation_count = 0
        def collect_trajectory(self, policies, task):
            self.evaluation_count += 1
            return Trajectory(task, [], 0.5)
    env = CountingEnv()
    t = trainer(tmp_path, env=env, options=GroupConfig())
    base = {"agent_1": policy(), "agent_2": policy()}
    candidates = {agent: policy("new") for agent in base}
    t._validation_gate(base, candidates)
    assert env.evaluation_count == 3 * t.config.n_val


def test_hotpotqa_aligned_configs():
    root = Path(__file__).resolve().parents[2] / "configs" / "grse"
    for name, sizes in [("hotpotqa/grse_wo_structure", (100, 50, 100, 5))]:
        data = json.loads((root / (name + ".json")).read_text())
        base, group = data["base"], GroupConfig(**data["grpo"])
        assert tuple(base[k] for k in ("n_train", "n_val", "n_test", "num_iterations")) == sizes
        assert base["trajectories_per_iteration"] == group.questions_per_iteration == base["n_train"]
        assert (base["refine_every"], base["induct_every"], base["consolidate_every"], base["prune_every"]) == (1, 1, 2, 2)
        assert base["skill_eval_delta"] == 0.03
        assert base["max_search_calls"] == 3
        assert group.validation_mode == "independent" and group.validation_rollouts == 1
        assert group.selection_mode == "all" and group.temperature is None
        assert not group.bootstrap_empty


def test_eval_temperature_applies_only_inside_eval_phases(tmp_path):
    t = trainer(tmp_path, options=GroupConfig(validation_rollouts=4, eval_temperature=0.0))
    client = type("Client", (), {"temperature": None})()
    t.env.env.llm_client = client
    with t._eval_sampling():
        assert client.temperature == 0.0
    assert client.temperature is None
    with pytest.raises(ValueError):
        GroupConfig(eval_temperature=3.0)


def test_locomo_profile_uses_upstream_prompts_and_credit_guidance():
    from GRSE import locomo
    from GRSE.critic import GroupSkillCreditCritic
    from GRSE.optimizer import GroupSkillOptimizer
    optimizer = GroupSkillOptimizer.__new__(GroupSkillOptimizer)
    critic = GroupSkillCreditCritic.__new__(GroupSkillCreditCritic)
    locomo.install(optimizer, critic)
    prompt = optimizer._induce_prompt(["hard case"], SkillLibrary([]), "gap")
    assert "SPECIFICITY CONTRACT" in prompt and "1. hard case" in prompt
    assert "Residual failure pattern: gap" in prompt
    assert critic.extra_instructions.startswith("## How to write the JSON fields")
    assert GroupSkillOptimizer.prompt_profile == "default"


def test_group_critic_omits_locomo_full_conversation():
    critic, prompts = DemoCritic(), []
    critic._call_llm = lambda prompt, **kwargs: prompts.append(prompt) or json.dumps(
        {"skills": [], "residual": {}, "induction_evidence": {}})
    task = {"question": "q", "session_meta": "WHOLE CONVERSATION"}
    trajectories = [Trajectory(task, [], r) for r in (0, 1)]
    critic.evaluate_group_skill_credit(RolloutGroup.build("q", trajectories), {"agent_1": policy()})
    assert "WHOLE CONVERSATION" not in prompts[0] and '"question": "q"' in prompts[0]


@pytest.mark.parametrize("mode, kept", [
    ("none", [True, True, True, True]),
    ("zero_variance", [True, False, False, False]),
    ("solved", [True, False, True, True]),
])
def test_group_filter_modes(tmp_path, mode, kept):
    t = trainer(tmp_path, options=GroupConfig(validation_rollouts=4, group_filter=mode))
    def group(rewards):
        return RolloutGroup.build("q", [Trajectory({"q": 1}, [], r) for r in rewards])
    groups = [group([1, 0, 1, 0]), group([1, 1, 1, 1]), group([0, 0, 0, 0]), group([0.5] * 4)]
    assert [t._keep_for_credit(g) for g in groups] == kept


def test_zero_variance_filter_skips_critic_for_flat_groups(tmp_path):
    t = trainer(tmp_path, options=GroupConfig(validation_rollouts=4, group_filter="zero_variance"))
    stats = t.train_one_iteration(1)
    # Demo groups score [1, 0, 0, 0]: every group has contrast, none filtered.
    assert stats["grpo_filtered_groups"] == 0
    assert stats["grpo_credit_questions"] == stats["grpo_questions"]
