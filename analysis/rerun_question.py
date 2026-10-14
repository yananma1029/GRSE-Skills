"""Answer selected test questions with a saved library and keep every agent's output.

Evaluation files store only final answers; this script shows the agent outputs for a library
without saved trajectories (an empty or seed library, or a HotpotQA library scored by
GRSE.evaluate).

    python analysis/rerun_question.py --config configs/runtime/test500_qwen_s42.json \
        --library experiments/grpo/runs/<run>/checkpoints/iter_5 --questions qa_00150 --out case.json
    python analysis/rerun_question.py --config configs/runtime/cat4_grse_qwen_s42.json \
        --library configs/grse/robustness/seeds/perturbed/grpo --questions conv-47__qa_0149 --out case.json

--config is a run config with the server URLs filled in (configs/runtime/, written by the
scripts/run_*.sh scripts). Decoding is greedy unless --temperature is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from GRSE.evaluate import build_env, load_policies, task_key  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--library", required=True, help="checkpoint directory with agent_*/role.md and skill_library.json")
    p.add_argument("--questions", required=True, help="comma-separated test question ids")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env, cfg = build_env(args.config, args.temperature)
    tasks = {task_key(t): t for t in env.sample_tasks(cfg.n_test, split="test")}
    policies = load_policies(args.library)
    out = {}
    for q in args.questions.split(","):
        traj = env.collect_trajectory({a: p.copy() for a, p in policies.items()}, deepcopy(tasks[q]))
        out[q] = {"question": tasks[q].get("question"), "gold": tasks[q].get("ground_truth"),
                  "reward": traj.reward, "answer": traj.metadata.get("final_answer", ""),
                  "steps": [{"agent": s.get("agent"), "output": s.get("output")} for s in traj.steps]}
        print(q, round(traj.reward, 2), out[q]["answer"][:80], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
