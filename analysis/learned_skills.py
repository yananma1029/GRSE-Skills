"""Analysis of the learned skills (first runs, seed 42), from the runs of scripts/run_main.sh.

    python analysis/learned_skills.py transitions [--root .] [--actors qwen gemma]
    python analysis/learned_skills.py skill-types [--root .] [--actors qwen gemma]   # needs QWEN_URL

transitions: F1 change from the empty to the final library. Under each library, a test
    question is declined (empty or "not mentioned"-style answer), correct (F1 >= 0.5) or wrong.
    Delta is the change in mean F1 (x100); D>C, W>C and C>W are the contributions to Delta of
    questions moving from declined to correct, wrong to correct and correct to wrong.
    Decl.: share of declined answers before -> after. HotpotQA uses the greedy evaluation file
    of scripts/run_main.sh (its "empty" library); on LoCoMo, each category has its own library
    and the empty-library answers are iteration 0 of the GRSE run of that category. For LoCoMo,
    the evidence statistics are printed as well: share of questions whose retriever output cites
    no dialogue turn, and how many declined->correct questions were answered without evidence.
skill-types: every final skill is labelled by Qwen3.8-27B (temperature 0) with one primary type
    (format, retrieval, verification, reasoning, abstention, generic) and a specificity score;
    labels are cached in outputs/analysis/skill_labels.json, keyed by the skill text, so reruns
    are stable. Prints the share of each type and the mean length (words of description and body);
    LoCoMo pools the four per-category libraries..
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from common import CATEGORIES, FINAL_ITER, abstains, eval_scores, libraries, run_dir, test_answers

CORRECT = 0.5
LABELS = {"maskills": "MASkills", "grse": "GRSE", "grse_wo": "GRSE w/o struct."}
SKILL_TYPES = {
    "format": "rules about the form of the final answer (length, wording, normalization, units, dates)",
    "retrieval": "how to search for or expand evidence (queries, keywords, adjacent turns, sessions, passages)",
    "verification": "checking that an answer is grounded in or consistent with the evidence",
    "reasoning": "a procedure for inferring the answer (temporal arithmetic, aggregation, multi-hop chaining)",
    "abstention": "when to answer versus decline, including rules that forbid declining",
    "generic": "general advice without a concrete, checkable procedure",
}
LABEL_PROMPT = """You label skills that a multi-agent question-answering system learned.
Classify the skill by its PRIMARY purpose into exactly one type:
{types}
Also rate its specificity: 1 = vague advice, 2 = concrete guidance, 3 = a precise procedure with
explicit conditions, steps or examples.
Answer with JSON only: {{"type": "<one of: {names}>", "specificity": <1|2|3>, "reason": "<one sentence>"}}

Skill name: {name}
Description: {description}
Body:
{body}"""


# ── transitions ─────────────────────────────────────────────────────────
def state(score, abstain):
    if abstain:
        return "declined"
    return "correct" if score >= CORRECT else "wrong"


def transitions(before, after, keys):
    """before/after: {question: (F1, abstain, evidence)}."""
    contrib = Counter()
    for q in keys:
        contrib[state(*before[q][:2]), state(*after[q][:2])] += (after[q][0] - before[q][0]) / len(keys)
    fixed = [q for q in keys if before[q][1] and state(*after[q][:2]) == "correct"]
    return {"n": len(keys), "delta": 100 * sum(contrib.values()),
            "D>C": 100 * contrib["declined", "correct"], "W>C": 100 * contrib["wrong", "correct"],
            "C>W": 100 * contrib["correct", "wrong"],
            "declined": (100 * fmean(before[q][1] for q in keys), 100 * fmean(after[q][1] for q in keys)),
            "no_evidence": (100 * fmean(not before[q][2] for q in keys if before[q][2] is not None),
                            100 * fmean(not after[q][2] for q in keys if after[q][2] is not None))
            if before[next(iter(keys))][2] is not None else None,
            "declined_to_correct": (len(fixed), sum(not after[q][2] for q in fixed))}


def hotpotqa_rows(root, actor, methods):
    evals = eval_scores(root / f"experiments/eval/hotpotqa_test500_{actor}_s42.json")
    def as_tuple(lib):
        return {q: (s["reward"], abstains(s.get("answer")), None) for q, s in lib.items()}
    empty = as_tuple(evals["empty"])
    for m in methods:
        if m in evals:
            after = as_tuple(evals[m])
            yield m, transitions(empty, after, sorted(set(empty) & set(after)))


def locomo_rows(root, actor, methods, grse):
    def as_tuple(answers):
        return {q: (a["f1"], a["abstain"], a["evidence"]) for q, a in answers.items()}
    empty, keys = {}, []
    finals = {m: {} for m in methods}
    for c in CATEGORIES:
        base = as_tuple(test_answers(run_dir(root, f"locomo_cat{c}_{grse}_{actor}_s42.out"), 0))
        empty.update(base)
        cat_keys = set(base)
        for m in methods:
            final = as_tuple(test_answers(run_dir(root, f"locomo_cat{c}_{m}_{actor}_s42.out"), FINAL_ITER))
            finals[m].update(final)
            cat_keys &= set(final)
        keys += sorted(cat_keys)
    for m in methods:
        yield m, transitions(empty, finals[m], keys)


def print_transitions(args, root):
    methods = ["maskills", args.grse]
    print(f"{'':28s} {'n':>4s} {'Delta':>6s} {'D>C':>6s} {'W>C':>6s} {'C>W':>6s}  Decl.(%)   No evid.(%)  D>C (no evid.)")
    for bench in ("hotpotqa", "locomo"):
        for actor in args.actors:
            rows = hotpotqa_rows(root, actor, methods) if bench == "hotpotqa" else locomo_rows(root, actor, methods, args.grse)
            print(f"{bench}, {actor}")
            for m, r in rows:
                line = (f"  {'GRSE' if m == args.grse else LABELS[m]:26s} {r['n']:4d} {r['delta']:+6.1f} {r['D>C']:+6.1f} {r['W>C']:+6.1f} "
                        f"{r['C>W']:+6.1f}  {r['declined'][0]:3.0f} -> {r['declined'][1]:3.0f}")
                if r["no_evidence"]:
                    line += (f"   {r['no_evidence'][0]:3.0f} -> {r['no_evidence'][1]:3.0f}   "
                             f"{r['declined_to_correct'][0]} ({r['declined_to_correct'][1]})")
                print(line)


# ── skill types ─────────────────────────────────────────────────────────
def skill_key(skill):
    text = f"{skill.get('name')}\n{skill.get('description')}\n{skill.get('body')}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def label_skills(skills, cache_path, url):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = {skill_key(s): s for s in skills if skill_key(s) not in cache}
    if todo:
        from openai import OpenAI
        client = OpenAI(base_url=url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=600)
        types = "\n".join(f"- {k}: {v}" for k, v in SKILL_TYPES.items())
        for i, (key, skill) in enumerate(todo.items(), 1):
            prompt = LABEL_PROMPT.format(types=types, names=", ".join(SKILL_TYPES), name=skill.get("name"),
                                         description=skill.get("description"), body=skill.get("body", "")[:6000])
            for _ in range(3):
                reply = client.chat.completions.create(
                    model="Qwen/Qwen3.8-27B", temperature=0, max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}).choices[0].message.content or ""
                try:
                    label = json.loads(reply)
                    if label.get("type") in SKILL_TYPES and label.get("specificity") in (1, 2, 3):
                        cache[key] = {"name": skill.get("name"), **label}
                        break
                except json.JSONDecodeError:
                    pass
            print(f"  labelled {i}/{len(todo)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache


def print_skill_types(args, root):
    groups = defaultdict(list)
    for actor in args.actors:
        for m in LABELS:
            logs = [f"hotpotqa_{m}_{actor}_s42.out"] + [f"locomo_cat{c}_{m}_{actor}_s42.out" for c in CATEGORIES]
            for log in logs:
                if not (root / "logs" / log).exists():
                    continue
                bench = log.split("_")[0]
                for skills in libraries(run_dir(root, log), FINAL_ITER).values():
                    groups[bench, actor, m] += skills.values()
    cache = label_skills([s for g in groups.values() for s in g], root / "outputs/analysis/skill_labels.json",
                         os.environ.get("QWEN_URL", "http://localhost:18001/v1"))
    print(f"{'':36s} {'n':>3s} " + " ".join(f"{t[:6]:>6s}" for t in SKILL_TYPES) + "  Words")
    for (bench, actor, m), skills in sorted(groups.items()):
        types = Counter(cache[skill_key(s)]["type"] for s in skills if skill_key(s) in cache)
        n = sum(types.values())
        words = fmean(len(f"{s.get('description', '')} {s.get('body', '')}".split()) for s in skills)
        print(f"{bench + ', ' + actor + ', ' + LABELS[m]:36s} {len(skills):3d} " +
              " ".join(f"{100 * types[t] / n if n else 0:6.0f}" for t in SKILL_TYPES) + f"  {words:5.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis", choices=["transitions", "skill-types"])
    p.add_argument("--root", default=".")
    p.add_argument("--actors", nargs="+", default=["qwen", "gemma"])
    p.add_argument("--grse", default="grse", choices=["grse_wo", "grse"],
                   help="GRSE variant for the transitions")
    args = p.parse_args()
    root = Path(args.root)
    (print_transitions if args.analysis == "transitions" else print_skill_types)(args, root)


if __name__ == "__main__":
    main()
