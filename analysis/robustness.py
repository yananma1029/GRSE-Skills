"""Robustness analyses (LoCoMo single-hop, Qwen3.8 actor), from the runs of scripts/run_robustness.sh.

    python analysis/robustness.py [--root .] [--seed 42] [--methods grse maskills]

1. Repairing a perturbed skill library: test F1 and BLEU (official LoCoMo metrics, x100) of
   the perturbed seed library, of its helpful skills alone, and of each method's final
   library; change in F1 from the perturbed library; share of declined answers.
2. Credit assigned to the planted skills: u_1, the utility each method stores for every
   planted skill after the first iteration (checkpoints/iter_1), whether its sign is correct
   (positive for helpful, negative for harmful), the mean u_1 of helpful minus harmful skills,
   and the skill's fate by iteration 5 (kept, refined_k, merged_k or pruned_k; k = iteration).
3. Utility of every planted skill after each iteration, and for GRSE the labels its critic gave
   the skill (from grpo/iter_*/credit_*.json), as discussed in the robustness cases.

Fate rule: a skill that leaves the library in iteration k is "merged" if consolidation was
accepted for its agent in that iteration and "pruned" otherwise; a skill that stays but whose
text changed is "refined" at the first change.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from common import FINAL_ITER, accepted_operators, add_repo_to_path, eval_scores, libraries, run_dir, \
    test_answers, abstains

LABELS = {"grse": "GRSE", "grse_wo": "GRSE w/o structure", "maskills": "MASkills"}
SEEDS_DIR = "configs/grse/robustness/seeds/perturbed"


def fate(run, agent, skill_id):
    libs = [libraries(run, it).get(agent, {}) for it in range(FINAL_ITER + 1)]
    seed = libs[0][skill_id]
    gone = next((it for it in range(1, FINAL_ITER + 1) if skill_id not in libs[it]), None)
    if gone:
        ops = accepted_operators(run).get((gone, agent), {})
        return f"{'merged' if ops.get('consolidate') else 'pruned'}_{gone}"
    changed = next((it for it in range(1, FINAL_ITER + 1)
                    if (libs[it][skill_id].get("body"), libs[it][skill_id].get("description"))
                    != (seed.get("body"), seed.get("description"))), None)
    return f"refined_{changed}" if changed else "kept"


def critic_labels(run, agent, skill_id):
    """{iteration: Counter(contribution label)} over the GRSE critic's credits for one skill."""
    out = {}
    for it in range(1, FINAL_ITER + 1):
        labels = Counter()
        for f in glob.glob(str(Path(run) / "grpo" / f"iter_{it}" / "credit_*.json")):
            for c in json.loads(Path(f).read_text())["credit"]["credit"].get("skill_credits", []):
                if c.get("agent") == agent and c.get("skill_id") == skill_id:
                    labels[c.get("contribution")] += 1
        if labels:
            out[it] = dict(labels)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--seed", default="42")
    p.add_argument("--methods", nargs="+", default=["grse", "maskills"], choices=list(LABELS))
    args = p.parse_args()
    root = Path(args.root)
    add_repo_to_path(root)
    from maskills.envs.locomo.metrics import compute_locomo_metrics

    runs = {m: run_dir(root, f"robust_{m}_s{args.seed}.out") for m in args.methods}

    # 1. repairing the perturbed library
    rows = {}
    for name, scores in eval_scores(root / "experiments/eval/locomo_cat4_robust_temp0.json").items():
        rows[name] = {}
        for q, s in scores.items():
            m = compute_locomo_metrics(s.get("answer") or "", s.get("gold") or "", 4)
            rows[name][q] = (m["f1"], m["bleu"], abstains(s.get("answer")))
    for m, run in runs.items():
        rows[m] = {q: (a["f1"], a["bleu"], a["abstain"]) for q, a in test_answers(run, FINAL_ITER).items()}
    common = set.intersection(*(set(v) for v in rows.values()))
    print(f"1. Repairing a perturbed library ({len(common)} test questions)")
    print(f"{'Library':24s} {'F1':>6s} {'BLEU':>6s} {'dF1':>6s} {'Abst%':>6s}")
    base = 100 * fmean(rows["perturbed"][q][0] for q in common)
    for name, r in rows.items():
        f1 = 100 * fmean(r[q][0] for q in common)
        print(f"{LABELS.get(name, name):24s} {f1:6.1f} {100 * fmean(r[q][1] for q in common):6.1f} "
              f"{'--' if name == 'perturbed' else f'{f1 - base:+.1f}':>6s} {100 * fmean(r[q][2] for q in common):6.1f}")

    # 2. credit for the planted skills, 3. utility through training
    manifest = json.loads((root / SEEDS_DIR / "manifest.json").read_text())
    planted = [(a, sid, kind) for a in ("agent_1", "agent_2") for sid, kind in manifest[a].items()]
    print("\n2. Credit for the planted skills (u_1: utility after iteration 1)")
    print(f"{'Agent':8s} {'Skill':40s} {'Kind':10s}" + "".join(f"{LABELS[m]:>28s}" for m in runs))
    u1 = {m: {} for m in runs}
    for agent, sid, kind in planted:
        cells = []
        for m, run in runs.items():
            u = libraries(run, 1).get(agent, {}).get(sid, {}).get("utility")
            u1[m][agent, sid] = u
            cells.append(f"{'--' if u is None else f'{u:+.2f}'} {fate(run, agent, sid)}")
        print(f"{agent:8s} {sid:40s} {kind:10s}" + "".join(f"{c:>28s}" for c in cells))
    for m in runs:
        signed = [(u1[m][a, s], k) for a, s, k in planted if k in ("helpful", "harmful") and u1[m][a, s] is not None]
        correct = sum((u > 0) if k == "helpful" else (u < 0) for u, k in signed)
        gap = fmean(u for u, k in signed if k == "helpful") - fmean(u for u, k in signed if k == "harmful")
        print(f"  {LABELS[m]}: correct sign {correct} of {len(signed)}; helpful - harmful mean u_1 {gap:+.2f}")

    print("\n3. Utility after each iteration (- : not in the library)")
    for m, run in runs.items():
        print(f"  {LABELS[m]}")
        for agent, sid, kind in planted:
            utils = []
            for it in range(FINAL_ITER + 1):
                skill = libraries(run, it).get(agent, {}).get(sid)
                utils.append("-" if skill is None else f"{skill.get('utility', 0):+.2f}")
            line = f"    {agent} {sid:40s} {kind:10s} {' '.join(f'{u:>6s}' for u in utils)}"
            if (Path(run) / "grpo").is_dir():
                line += f"   critic labels {critic_labels(run, agent, sid)}"
            print(line)


if __name__ == "__main__":
    main()
