"""Shared helpers for the analysis scripts: find runs from the logs of scripts/run_*.sh and
read their saved test answers, skill libraries and operator logs.

Every script takes --root (the repository root, default: current directory); runs are found
from <root>/logs/<log name> through the "GRPO run: <id>" / "MASKILLS run: <id>" line.
"""

from __future__ import annotations

import glob
import json
import random
import re
import sys
from pathlib import Path
from statistics import fmean

FINAL_ITER = 5
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
RUN_RE = re.compile(r"(GRPO|MASKILLS) run: (\S+)")
# A declined answer: empty, or a "not mentioned"-style refusal.
ABSTAIN_RE = re.compile(r"not (mentioned|specified|available|found|stated)|no information|unknown"
                        r"|cannot be determined")
CITATION_RE = re.compile(r"\[D\d+:\d+\]")
OPS_RE = re.compile(r"\[(agent_\d+)\] refine=(\d+) induct=(\d+) consolidate=(\d+) prune=(\d+) .*?"
                    r"(ACCEPTED|ROLLED-BACK)")


def add_repo_to_path(root):
    """Make GRSE/ and maskills/ importable when a script is run from analysis/."""
    sys.path.insert(0, str(Path(root).resolve()))


def run_dir(root, log):
    """Run directory named by logs/<log>; exits with a hint if the run has not been made."""
    path = Path(root) / "logs" / log
    match = RUN_RE.search(path.read_text(errors="replace")) if path.exists() else None
    if not match:
        sys.exit(f"no run id in {path}; run the matching scripts/run_*.sh first")
    sub = "experiments/grpo/runs" if match.group(1) == "GRPO" else "experiments/runs"
    return Path(root) / sub / match.group(2)


def abstains(answer):
    """True for an empty answer or one that declines to answer ("not mentioned", ...)."""
    answer = (answer or "").strip().lower()
    return not answer or bool(ABSTAIN_RE.search(answer))


def task_key(task):
    return str(task.get("task_id") or task.get("question"))


def test_answers(run, iteration):
    """{question id: answer record} from a run's saved test trajectories of one iteration.

    Record: reward, f1, bleu (official LoCoMo metrics; None on HotpotQA), answer, gold,
    question, category, abstain, evidence (retriever output cites a dialogue turn), greps."""
    out = {}
    for f in glob.glob(str(Path(run) / "trajectories_test" / f"iter_{iteration}" / "*.json")):
        d = json.loads(Path(f).read_text())
        task, meta = d["task"], d["metadata"]
        metrics = meta.get("qa_metrics", {})
        retriever = d["transitions"][0] if d["transitions"] else {}
        answer = meta.get("final_answer")
        out[task_key(task)] = {
            "reward": d["reward"], "f1": metrics.get("f1"), "bleu": metrics.get("bleu"),
            "answer": answer, "gold": task.get("ground_truth") or task.get("answer"),
            "question": task.get("question"), "category": task.get("category"),
            "abstain": abstains(answer),
            "evidence": bool(CITATION_RE.search(str(retriever.get("output", "")))),
            "greps": len(retriever.get("grep_log") or [])}
    return out


def eval_scores(path):
    """{library: {question id: score record}} from a GRSE.evaluate output file, failed questions dropped."""
    data = json.loads(Path(path).read_text())
    return {name: {q: s for q, s in lib["scores"].items() if "error" not in s} for name, lib in data.items()}


def libraries(run, iteration):
    """{agent: {skill id: skill}} of a run's checkpoint."""
    libs = {}
    for f in sorted(glob.glob(str(Path(run) / "checkpoints" / f"iter_{iteration}" / "agent_*" / "skill_library.json"))):
        libs[Path(f).parent.name] = {s["skill_id"]: s for s in json.loads(Path(f).read_text())["skills"]}
    return libs


def accepted_operators(run):
    """{(iteration, agent): {"refine", "induct", "consolidate", "prune": count}} of accepted updates,
    from the run's logs/train.log (GRSE and MASkills write the same line)."""
    ops, iteration = {}, 0
    for line in (Path(run) / "logs" / "train.log").read_text(errors="replace").splitlines():
        started = re.search(r"Iteration (\d+) started", line)
        if started:
            iteration = int(started.group(1))
        match = OPS_RE.search(line)
        if match and match.group(6) == "ACCEPTED":
            ops[iteration, match.group(1)] = dict(zip(("refine", "induct", "consolidate", "prune"),
                                                      map(int, match.group(2, 3, 4, 5))))
    return ops


def paired_bootstrap(diffs, resamples=5000, seed=0):
    """Mean of paired differences and its 95% bootstrap interval."""
    rng = random.Random(seed)
    boots = sorted(fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return fmean(diffs), boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]


def mean_sd(values):
    values = list(values)
    if len(values) < 2:
        return f"{fmean(values):.1f}"
    m = fmean(values)
    sd = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return f"{m:.1f}±{sd:.1f}"
