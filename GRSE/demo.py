"""Synthetic offline integration demo. No claim of benchmark effectiveness."""

import json
import logging
from collections import defaultdict
from threading import Lock

from maskills.config.base import LanguageTaskConfig
from maskills.core.base import Trajectory

from .config import GroupConfig
from .critic import GroupSkillCreditCritic
from .optimizer import GroupSkillOptimizer
from .routing import SkillRoutingEnvironment
from .store import GroupRunStore
from .trainer import GroupSkillTrainer


class DemoEnvironment:
    def __init__(self):
        self.calls = defaultdict(int)
        self.lock = Lock()

    def sample_tasks(self, n, seed=None, split="train"):
        return [{"task_id": f"{split}-{i}", "question": f"Check evidence for case {i}",
                 "ground_truth": "verified", "split": split} for i in range(n)]

    def collect_trajectory(self, policies, task):
        with self.lock:
            index = self.calls[task["task_id"]]
            self.calls[task["task_id"]] += 1
        steps, outcomes = [], []
        for agent, policy in policies.items():
            good = bool(policy.skill_library.get("verify-evidence")) or index % 4 == 0
            outcomes.append(good)
            steps.append({"agent": agent, "output": "verified evidence" if good else "unchecked claim"})
        return Trajectory(task=task, steps=steps, reward=float(all(outcomes)),
                          metadata={"final_answer": "verified" if all(outcomes) else "guess"})


class DemoSelector:
    def select(self, agent, policy, question):
        return policy.skill_library.ids()


class DemoCritic(GroupSkillCreditCritic):
    def __init__(self):
        self.tool_library = ""
        self.logger = logging.getLogger(__name__)

    def _call_llm(self, prompt, max_tokens=None):
        data = json.loads(prompt.split("\n\nINPUT:\n", 1)[1])
        skills, residual, evidence = [], {}, {}
        for agent, library in data["available_library"].items():
            if not library:
                residual[agent] = {"summary": "Unchecked claims fail; verify evidence before handoff.",
                                   "needs_new_skill": True, "missing_capability": "evidence verification"}
                evidence[agent] = ["Successful rollout checks evidence; failed rollouts pass unchecked claims."]
            for skill in library:
                skills.append({"agent": agent, "skill_id": skill["id"], "contribution": "helped",
                               "evidence": "The disclosed procedure produces verified handoffs.",
                               "suggested_edit": "", "utility_delta": 0.8})
        return json.dumps({"skills": skills, "residual": residual, "induction_evidence": evidence})


class DemoOptimizer(GroupSkillOptimizer):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prune_utility_threshold = -0.3

    def _llm_call(self, prompt, max_tokens=None):
        if "Create at most ONE reusable skill" in prompt:
            return ("---\nname: verify-evidence\ndescription: Verify evidence before handoff.\n---\n"
                    "Check that evidence supports the claim before passing it to a teammate.")
        if "You are aggregating" in prompt:
            return json.dumps({"verdict": "keep", "summary": "Consistently helpful",
                               "utility_score": 0.8, "redundant_with": []})
        return "[]"


def run_demo(output):
    config = LanguageTaskConfig(
        exp_name="grpo_demo", experiment_dir=output, skill_evolution=True,
        num_iterations=2, trajectories_per_iteration=2, n_val=2, n_test=0,
        max_workers=1, optimizer_workers=1, induct_every=2,
        consolidate_every=0, prune_every=0,
    )
    trainer = GroupSkillTrainer(
        config=config,
        env=SkillRoutingEnvironment(DemoEnvironment(), DemoSelector()),
        critic=DemoCritic(), optimizer=DemoOptimizer(),
        group_config=GroupConfig(bootstrap_empty=True, rollouts_per_question=4, validation_rollouts=4),
        store=GroupRunStore(output),
    )
    # Use the same initial-state convention as the real CLI.
    trainer.checkpoint.save_policies(0, trainer.checkpoint.get_policies(), {})
    stats = trainer.train()
    print(f"Synthetic demo complete (no LLM calls): {trainer.store.get_run_dir(trainer.run_id)}")
    return stats
