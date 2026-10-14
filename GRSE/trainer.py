"""Question-group training with cache validation and configurable acceptance."""

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from maskills.core.base import Trajectory
from maskills.core.policy import AgentPolicy
from maskills.trainer.skill_evolution import SkillEvolutionTrainer

from .config import GroupConfig
from .group import RolloutGroup, fingerprint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


class GroupSkillTrainer(SkillEvolutionTrainer):
    def __init__(self, *args, group_config=None, **kwargs):
        self.group_config = group_config or GroupConfig()
        super().__init__(*args, **kwargs)
        if self.config.mini_batch_size:
            raise ValueError("Use credit_question_batch_size; never split a rollout group")
        if not self.group_config.allow_no_validation and not (getattr(self.config, "n_val", 0) or 0):
            raise ValueError("Group training requires n_val > 0 unless allow_no_validation is explicit")
        self.artifact_root = Path(self.store.get_run_dir(self.run_id)) / "grpo"
        options_path = self.artifact_root / "options.json"
        options = asdict(self.group_config)
        if options_path.exists():
            previous = json.loads(options_path.read_text())
            for name in ("structured_failures", "category_analysis"):
                previous.setdefault(name, False)
            if previous != options:
                raise ValueError("Resume requires unchanged GRPO options")
        write_json(options_path, options)

    @contextmanager
    def _eval_sampling(self):
        """Switch the actor to eval_temperature for baseline/gate/val/test.

        Phases run one after another, so no training rollout is in flight.
        """
        temperature = self.group_config.eval_temperature
        env = getattr(self.env, "env", self.env)  # unwrap SkillRoutingEnvironment
        client = getattr(env, "llm_client", None)
        if temperature is None or not hasattr(client, "temperature"):
            yield
            return
        previous, client.temperature = client.temperature, temperature
        try:
            yield
        finally:
            client.temperature = previous

    def _run_baseline_round(self):
        with self._eval_sampling():
            return super()._run_baseline_round()

    def _run_scored(self, policies, task):
        traj = self.env.collect_trajectory(
            {a: p.copy() for a, p in policies.items()}, deepcopy(task))
        if self.reward_fn:
            traj.reward = self.reward_fn.compute(traj)
        if not math.isfinite(traj.reward):
            raise ValueError("Non-finite trajectory reward")
        return traj

    def _run_eval(self, policies, task):
        """_run_scored for gate/val/test. A question the actor never answers
        (e.g. a greedy repetition loop that ends at the token limit with no
        content) scores 0 instead of aborting the whole run."""
        try:
            return self._run_scored(policies, task)
        except TypeError as exc:
            if "no usable content" not in str(exc):
                raise
            self.run_logger.warning(f"No answer, scored 0: {task.get('task_id') or task.get('question')}")
            return Trajectory(task=task, steps=[], reward=0.0,
                              metadata={"final_answer": "", "error": str(exc)[:300],
                                        "qa_metrics": {k: 0.0 for k in ("f1", "precision", "recall", "em", "bleu")}})

    def _collect_groups(self, policies, iteration):
        opts = self.group_config
        count = opts.questions_per_iteration or self.config.trajectories_per_iteration
        signature = fingerprint({"policies": {a: p.to_dict() for a, p in policies.items()},
                                 "options": opts.cache_options(), "base_config": asdict(self.config),
                                 "question_count": count})
        manifest_path = self.artifact_root / f"iter_{iteration}" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["signature"] != signature:
                raise ValueError("Cached rollouts belong to different policies/options")
            tasks = manifest["tasks"]
        else:
            tasks = self.env.sample_tasks(count, seed=opts.seed + iteration, split="train")
            # Duplicate tasks must not become separately weighted groups.
            tasks = list({fingerprint(t): t for t in tasks}.values())
            if not tasks:
                raise ValueError("No training questions available")
            write_json(manifest_path, {"signature": signature, "tasks": tasks})
        width = opts.rollouts_per_question
        expected = {(fingerprint(task), r): (i * width + r, task)
                    for i, task in enumerate(tasks) for r in range(width)}
        cached = {}
        for traj in self.trajectory_store.load(iteration):
            meta = traj.metadata.get("grpo", {})
            key = (meta.get("group_id"), meta.get("rollout_index"))
            if meta.get("signature") != signature or key not in expected or key in cached:
                raise ValueError("Invalid, duplicate, or stale cached rollout")
            if fingerprint(traj.task) != key[0]:
                raise ValueError("Cached rollout task does not match its group")
            cached[key] = traj

        def run(item):
            key, (episode, task) = item
            if key in cached:
                return key, cached[key]
            traj = self._run_scored(policies, task)
            traj.metadata["grpo"] = {"signature": signature, "group_id": key[0],
                                     "rollout_index": key[1]}
            self.trajectory_store.save(iteration, episode, traj)
            return key, traj

        self.run_logger.info(f"GRPO: {len(tasks)} questions x {width} team rollouts; "
                             f"reusing {len(cached)} cached trajectories")
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = dict(executor.map(run, expected.items()))
        groups = []
        for task in tasks:
            gid = fingerprint(task)
            groups.append(RolloutGroup.build(gid, [results[(gid, r)] for r in range(width)],
                                             opts.epsilon))
        return groups

    def _keep_for_credit(self, group):
        mode = self.group_config.group_filter
        if mode == "zero_variance":
            return any(group.advantages)
        if mode == "solved":
            return not all(t.reward >= 1.0 for t in group.trajectories)
        return True

    def train_one_iteration(self, iteration):
        base = {a: p.copy() for a, p in self.checkpoint.get_policies().items()}
        self.run_logger.iteration_start(iteration, {a: str(p) for a, p in base.items()})
        groups = self._collect_groups(base, iteration)
        informative = [g for g in groups if self._keep_for_credit(g)]
        if len(informative) < len(groups):
            self.run_logger.info(f"GRPO filter ({self.group_config.group_filter}): "
                                 f"{len(informative)}/{len(groups)} groups go to the critic")
        selected = informative
        batch = self.group_config.credit_question_batch_size
        if batch and batch < len(selected):
            selected = random.Random(self.group_config.seed + iteration).sample(selected, batch)

        errors = []
        requested_groups = len(selected)
        audits = []

        def critique(group):
            # Skip (never invent) feedback for a group whose critic call fails;
            # if every group fails, the iteration still fails below.
            try:
                feedback = self.critic.evaluate_group_skill_credit(group, base)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                if self.group_config.structured_failures:
                    write_json(self.artifact_root / f"iter_{iteration}" / f"credit_error_{group.group_id}.json", {
                        "group_id": group.group_id, "error": f"{type(exc).__name__}: {exc}",
                        "rewards": [t.reward for t in group.trajectories]})
                self.run_logger.warning(f"critic failed for group {group.group_id[:12]}; "
                                        f"skipped: {exc}")
                return None
            write_json(self.artifact_root / f"iter_{iteration}" / f"credit_{group.group_id}.json", {
                "group_id": group.group_id,
                "rewards": [t.reward for t in group.trajectories],
                "advantages": group.advantages,
                "credit": asdict(feedback),
            })
            audit = getattr(feedback, "validation_audit", {})
            if audit:
                audits.append(audit)
                if audit["status"] != "valid":
                    self.run_logger.warning(
                        f"critic group {group.group_id[:12]}: {audit['status']}; "
                        f"kept {audit['accepted_diagnoses']} diagnoses; "
                        f"repair_attempted={audit['repair_attempted']}"
                        + (f"; last issue: {str(audit['attempts'][-1]['issues'][:1])[:300]}"
                           if audit["status"] == "invalid" and audit.get("attempts") else ""))
                if audit["status"] == "invalid":
                    errors.append(ValueError("No valid structured feedback after repair"))
                    return None
            return feedback

        workers = getattr(self.config, "optimizer_workers", 1)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            feedbacks = list(executor.map(critique, selected))
        coverage = {}
        if self.group_config.structured_failures:
            coverage = {"critic_requested_groups": requested_groups,
                        "critic_valid_groups": sum(a["status"] == "valid" for a in audits),
                        "critic_repaired_groups": sum(a["status"] == "repaired" for a in audits),
                        "critic_partial_groups": sum(a["status"] == "partial" for a in audits),
                        "critic_invalid_groups": sum(f is None for f in feedbacks),
                        "critic_repair_attempts": sum(a["repair_attempted"] for a in audits),
                        "critic_repair_successes": sum(a["repair_succeeded"] for a in audits),
                        "critic_no_diagnosis_groups": sum(a["accepted_diagnoses"] == 0
                                                          and a["status"] != "invalid" for a in audits)}
            write_json(self.artifact_root / f"iter_{iteration}" / "critic_coverage.json", coverage)
            self.run_logger.info(f"Structured critic coverage: {coverage}")
        if errors and all(f is None for f in feedbacks):
            raise errors[0]
        selected =[g for g, f in zip(selected, feedbacks) if f is not None]
        feedbacks = [f for f in feedbacks if f is not None]
        credits = [f.credit for f in feedbacks]
        used_trajectories = [t for g in selected for t in g.trajectories]
        # One utility vote per question, but count disclosure events per rollout.
        self._update_skill_stats(base, used_trajectories, credits)
        gradients, residuals, evidence = self._aggregate(credits, base, used_trajectories)
        for feedback in feedbacks:
            for agent, lines in feedback.induction_evidence.items():
                if agent in base:
                    evidence.setdefault(agent, []).extend(lines)
        if self.group_config.structured_failures:
            from .failure_analysis import summarize_failures
            analysis, evidence = summarize_failures(
                groups, selected, feedbacks, self.group_config.category_analysis)
            # Replace generic low-reward induction with grounded missing-capability
            # diagnoses. Tool failures/uncertain attribution alone create no skill.
            residuals = {agent: "\n".join(lines[:5]) for agent, lines in evidence.items()}
            for row in analysis["skill_credits"]:
                gradient = gradients.get(row["agent"], {}).get(row["skill_id"])
                if gradient is not None:
                    gradient.summary += "\nCategory evidence (do not equate team reward with skill value): " + json.dumps(row)
            write_json(self.artifact_root / f"iter_{iteration}" / "failure_analysis.json", analysis)
        schedule = self.schedule.active(iteration)
        candidates, op_stats = {}, {}
        for agent, policy in base.items():
            agent_schedule = dict(schedule)
            if self.group_config.bootstrap_empty and not policy.skill_library:
                agent_schedule["induct"] = True
            library, log = self._evolve_library(
                agent, policy, gradients.get(agent, {}), residuals.get(agent, ""),
                evidence.get(agent, []), agent_schedule)
            role = policy.role
            if getattr(self.config, "evolve_role", False):
                new_role = self.optimizer.refine_role(
                    role, residuals.get(agent, ""), evidence.get(agent, []), agent_name=agent)
                if new_role:
                    role, log["role_edited"] = new_role, True
            candidates[agent] = AgentPolicy(role=role, skill_library=library)
            op_stats[agent] = log

        with self._eval_sampling():
            accepted, gate = self._validation_gate(base, candidates)
        trajectories = [t for g in groups for t in g.trajectories]
        stats = self._collect_stats(trajectories, prefix="train")
        stats.update(self._operator_metrics(op_stats, accepted, base, gate))
        stats.update(coverage)
        stats.update({"grpo_questions": len(groups), "grpo_credit_questions": len(selected),
                      "grpo_filtered_groups": len(groups) - len(informative),
                      "grpo_rollouts_per_question": self.group_config.rollouts_per_question,
                      "grpo_zero_variance_groups": sum(not any(g.advantages) for g in groups)})
        with self._eval_sampling():
            stats.update(self._evaluate_final(accepted, iteration, "val"))
            if getattr(self.config, "eval_test_every_iter", False) or iteration >= self.config.num_iterations:
                stats.update(self._evaluate_final(accepted, iteration, "test"))
        # Save AFTER final evaluation, so checkpoint metrics are complete.
        write_json(self.artifact_root / f"iter_{iteration}" / "updates.json", {
            "gradients": {a: {sid: asdict(g) for sid, g in gs.items()}
                          for a, gs in gradients.items()},
            "residuals": residuals, "induction_evidence": evidence,
            "candidate_policies": {a: p.to_dict() for a, p in candidates.items()},
            "operations": op_stats, "gate": gate,
        })
        stats["avg_reward"] = stats["train_avg_reward"]
        # Upstream token tracker does not observe all API clients; don't report
        # misleading zero costs as measured totals.
        for key in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
            stats.pop(key, None)
        client = getattr(self.env, "llm_client", None)
        if hasattr(client, "usage"):
            stats["actor_and_selector_tokens_cumulative"] = dict(client.usage)
        self.checkpoint.save_policies(iteration, accepted, stats)
        self._export_autogen_skills(iteration, accepted)
        self._log_operator_summary(iteration, op_stats, gate, accepted)
        self.run_logger.info(f"Iteration {iteration}: train={stats['train_avg_reward']:.4f}, "
                             f"val={stats.get('val_avg_reward', 'disabled')}; "
                             "total API cost not measured")
        self.store.append_metrics(self.run_id, {"type": "iteration", "iteration": iteration, **stats})
        return stats

    def _evaluate_final(self, policies, iteration, split):
        """Use the SAME reward function in training, gate, and final reports."""
        count = getattr(self.config, f"n_{split}", 0) or 0
        if not count:
            return {}
        tasks = self.env.sample_tasks(count, seed=self.group_config.seed, split=split)
        if not tasks:
            return {}
        def run(item):
            index, task = item
            traj = self._run_eval(policies, task)
            self.trajectory_store.save(iteration, index, traj, split=split)
            return traj
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            trajectories = list(executor.map(run, enumerate(tasks)))
        return self._collect_stats(trajectories, prefix=split)

    def _val_score(self, policies, tasks):
        repeated = [task for task in tasks for _ in range(self.group_config.validation_rollouts)]
        if not repeated:
            raise ValueError("Cannot score an empty validation set")
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            trajectories = list(executor.map(lambda task: self._run_eval(policies, task), repeated))
        return fmean(t.reward for t in trajectories)

    def _validation_gate(self, base_policies, candidate_policies):
        n_val = getattr(self.config, "n_val", 0) or 0
        tasks = self.env.sample_tasks(n_val, seed=self.group_config.seed, split="val") if n_val else []
        if not tasks:
            if not self.group_config.allow_no_validation:
                raise ValueError("No validation tasks; refusing an unvalidated update")
            return dict(candidate_policies), {"status": "disabled"}
        baseline = current = self._val_score(base_policies, tasks)
        accepted, per_agent = dict(base_policies), {}
        independent = self.group_config.validation_mode == "independent"
        # MASkills tests every candidate against the same unchanged team.
        for agent, candidate in candidate_policies.items():
            base = base_policies[agent]
            if candidate.skill_library.to_dict() == base.skill_library.to_dict() and candidate.role == base.role:
                per_agent[agent] = {"changed": False, "accepted": True}
                continue
            trial = dict(base_policies if independent else accepted)
            trial[agent] = candidate
            score = self._val_score(trial, tasks)
            reference = baseline if independent else current
            floor = (baseline if independent else max(baseline, current)) - self.delta
            ok = score >= floor
            per_agent[agent] = {"changed": True, "accepted": ok, "val_score": score,
                                "base_score": reference, "acceptance_floor": floor}
            if ok:
                accepted[agent] = candidate
                current = score
        gate = {"status": "active", "mode": self.group_config.validation_mode,
                "base_score": baseline, "per_agent": per_agent}
        # Independent candidate scores do not measure the combined team.
        if not independent:
            gate["final_score"] = current
        return accepted, gate

    def evaluate_skill_ablation(self, policies, tasks, agent, skill_id, repeats=4):
        """Optional full-team with/without-skill experiment; never auto-prunes.

        Both conditions rerun selection from fresh contexts. The estimate is
        a marginal effect under current routing, not proof of causal use.
        """
        if repeats < 1 or not tasks:
            raise ValueError("Ablation requires tasks and positive repeats")
        masked = {a: p.copy() for a, p in policies.items()}
        if agent not in masked or not masked[agent].skill_library.remove(skill_id):
            raise ValueError("Unknown agent/skill for ablation")
        results = []
        for task in tasks:
            # Interleave conditions to reduce provider/time drift.
            full, without = [], []
            for _ in range(repeats):
                full.append(self._run_scored(policies, task).reward)
                without.append(self._run_scored(masked, task).reward)
            results.append({"task_id": fingerprint(task), "full_rewards": full,
                            "masked_rewards": without, "delta": fmean(full) - fmean(without)})
        return {"agent": agent, "skill_id": skill_id, "repeats": repeats,
                "mean_delta": fmean(r["delta"] for r in results), "tasks": results}
