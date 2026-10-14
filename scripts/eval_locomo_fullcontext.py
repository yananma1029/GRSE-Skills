#!/usr/bin/env python3
"""LoCoMo's own full-context baseline on our per-category test splits.

Reproduces the benchmark authors' QA protocol (env/locomo/task_eval/gpt_utils.py,
batch size 1): the whole conversation, oldest session first, rendered as
"DATE: ... CONVERSATION: <speaker> said, "<text>"", preceded by
CONV_START_PROMPT and followed by QA_PROMPT; temporal (category 2) questions
get the authors' "Use DATE of CONVERSATION ..." suffix; 32 answer tokens,
temperature 0. No skills and no agents: one model call per question.

Questions come from the same GRSE config (category filter, seed-42 2/2/6
conversation split) and answers are scored by the same LocomoRewardGenerator
as every skill run, so results are directly comparable. Output follows
GRSE.evaluate's results format under the given --name.

    python scripts/eval_locomo_fullcontext.py --categories 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from GRSE.evaluate import build_env, task_key  # noqa: E402
from maskills.core.base import Trajectory  # noqa: E402

# Verbatim from env/locomo/task_eval/gpt_utils.py (including its typo).
CONV_START_PROMPT = ("Below is a conversation between two people: {} and {}. The conversation takes place "
                     "over multiple days and the date of each conversation is wriiten at the beginning of "
                     "the conversation.\n\n")
QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""
TEMPORAL_SUFFIX = " Use DATE of CONVERSATION to answer with an approximate date."
ANSWER_TOKENS = 32


def render_conversation(task):
    """Full conversation in the authors' format, oldest session first."""
    sessions = []
    for session in task["session_meta"]:
        turns = ""
        for dialog in session["turns"]:
            turn = f"{dialog['speaker']} said, \"{dialog['text']}\"\n"
            if dialog.get("blip_caption"):
                turn += f" and shared {dialog['blip_caption']}."
            turns += turn + "\n"
        sessions.append(f"DATE: {session['date']}\nCONVERSATION:\n{turns}")
    speakers = (list(task.get("speakers") or []) + ["", ""])[:2]
    return CONV_START_PROMPT.format(*speakers) + "\n\n".join(sessions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--config-template", default="configs/grse/locomo/cat{}_grse_wo_structure.json",
                   help="per-category GRSE config that defines the test split")
    p.add_argument("--config", default="",
                   help="one config for all categories (e.g. the shared-library split "
                        "configs/grse/ablation/mix_wo_structure.json); its test questions are filtered by --categories")
    p.add_argument("--name", default="fullcontext")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="first N questions per category (smoke test)")
    p.add_argument("--out", default="experiments/eval/locomo_percat_temp0.json")
    args = p.parse_args(argv)

    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    entry = results.setdefault(args.name, {"protocol": "LoCoMo full-context, temperature 0",
                                           "temperature": 0.0, "scores": {}})
    for cat in args.categories:
        env, config = build_env(args.config or args.config_template.format(cat), temperature=0.0)
        tasks = env.sample_tasks(config.n_test, split="test")
        if args.config:
            tasks = [t for t in tasks if int(t["category"]) == cat]
        if args.limit:
            tasks = tasks[: args.limit]
        todo = [t for t in tasks if task_key(t) not in entry["scores"]]
        print(f"category {cat}: {len(tasks)} test questions, {len(todo)} to run", flush=True)

        def run(task):
            question = task["question"] + (TEMPORAL_SUFFIX if int(task["category"]) == 2 else "")
            prompt = render_conversation(task) + "\n\n" + QA_PROMPT.format(question)
            try:
                answer, _ = env.llm_client.chat_messages_with_usage(
                    [{"role": "user", "content": prompt}], max_tokens=ANSWER_TOKENS)
            except Exception as exc:  # noqa: BLE001
                print(f"  {task_key(task)} failed: {type(exc).__name__}: {exc}", flush=True)
                return task_key(task), {"error": f"{type(exc).__name__}: {exc}"[:300]}
            traj = Trajectory(task=task, steps=[], reward=0.0, metadata={"final_answer": answer.strip()})
            reward = env.reward_gen.compute(traj)
            metrics = traj.metadata.get("qa_metrics", {})
            return task_key(task), {"reward": reward, "f1": metrics.get("f1"), "bleu": metrics.get("bleu"),
                                    "category": int(task["category"]), "answer": answer.strip(),
                                    "gold": task.get("ground_truth", "")}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for key, score in executor.map(run, todo):
                entry["scores"][key] = score
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        done = [entry["scores"][task_key(t)] for t in tasks if "error" not in entry["scores"].get(task_key(t), {"error": 1})]
        if done:
            print(f"category {cat}: Score {100 * fmean(s['reward'] for s in done):.1f}  "
                  f"F1 {100 * fmean(s['f1'] for s in done):.1f}  BLEU {100 * fmean(s['bleu'] for s in done):.1f}  "
                  f"(n={len(done)})", flush=True)


if __name__ == "__main__":
    main()
