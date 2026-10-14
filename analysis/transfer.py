"""Cross-task transfer of skill libraries (test F1 x100, Qwen3.8 actor).

    python analysis/transfer.py [--root .] [--seeds 42 43 44]

Rows: the empty library (re-evaluated next to the transferred ones), and per method the
in-domain libraries (learned on the target, from the runs of scripts/run_main.sh) and the
transferred ones (from scripts/run_transfer.sh). Columns: HotpotQA libraries on each LoCoMo
category, and LoCoMo multi-hop libraries on HotpotQA. Mean +- SD over the seeds; for each
transferred library, also the paired difference to the empty library with a 95% bootstrap
interval (questions as units, each question's score averaged over the seeds).

The evaluation files store each answer; LoCoMo answers are scored here with the official
category-specific F1 (the files' "reward" is the training reward, the mean of F1 and BLEU).
HotpotQA scores are token F1 already.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from common import CATEGORIES, FINAL_ITER, add_repo_to_path, eval_scores, mean_sd, paired_bootstrap, \
    run_dir, test_answers

METHODS = {"grse": "GRSE", "maskills": "MASkills"}


def locomo_f1(scores, category):
    from maskills.envs.locomo.metrics import f1_score
    return {q: f1_score(s.get("answer") or "", s.get("gold") or "", category) for q, s in scores.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    args = p.parse_args()
    root = Path(args.root)
    add_repo_to_path(root)

    columns = [(f"cat{c}", CATEGORIES[c]) for c in CATEGORIES] + [("hotpotqa", "HotpotQA")]
    cells = {}       # (row, column) -> text
    tests = []       # paired comparisons with the empty library
    for col, _ in columns:
        if col == "hotpotqa":
            evals = eval_scores(root / "experiments/eval/transfer_l2h.json")
            per_lib = {name: {q: s["reward"] for q, s in lib.items()} for name, lib in evals.items()}
            for m in METHODS:
                per_seed = [eval_scores(root / f"experiments/eval/hotpotqa_test500_qwen_s{s}.json")[m]
                            for s in args.seeds]
                cells[m, "in-domain", col] = mean_sd(100 * fmean(x["reward"] for x in d.values()) for d in per_seed)
        else:
            c = int(col[3:])
            evals = eval_scores(root / f"experiments/eval/transfer_h2l_cat{c}.json")
            per_lib = {name: locomo_f1(lib, c) for name, lib in evals.items()}
            for m in METHODS:
                finals = [test_answers(run_dir(root, f"locomo_cat{c}_{m}_qwen_s{s}.out"), FINAL_ITER) for s in args.seeds]
                cells[m, "in-domain", col] = mean_sd(100 * fmean(x["f1"] for x in d.values()) for d in finals)
        empty = per_lib["empty"]
        cells["empty", "", col] = f"{100 * fmean(empty.values()):.1f}"
        for m in METHODS:
            libs = [per_lib[f"{m}_s{s}"] for s in args.seeds]
            cells[m, "transferred", col] = mean_sd(100 * fmean(lib.values()) for lib in libs)
            keys = sorted(set(empty).intersection(*libs))
            diff, lo, hi = paired_bootstrap([fmean(lib[q] for lib in libs) - empty[q] for q in keys])
            tests.append((col, m, len(keys), 100 * diff, 100 * lo, 100 * hi))

    names = [name for _, name in columns]
    print(f"{'':24s}" + "".join(f"{n:>14s}" for n in names))
    rows = [("empty", "")] + [(m, kind) for m in METHODS for kind in ("in-domain", "transferred")]
    for m, kind in rows:
        label = "Empty" if m == "empty" else f"{METHODS[m]} {kind}"
        print(f"{label:24s}" + "".join(f"{cells[m, kind, col]:>14s}" for col, _ in columns))
    print("\nTransferred minus empty library (paired over questions, 95% bootstrap CI):")
    for col, m, n, diff, lo, hi in tests:
        print(f"  {dict(columns)[col]:12s} {METHODS[m]:9s} n={n:<4d} {diff:+5.1f}  [{lo:+.1f}, {hi:+.1f}]")


if __name__ == "__main__":
    main()
