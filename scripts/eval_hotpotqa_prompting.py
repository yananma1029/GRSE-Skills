#!/usr/bin/env python3
"""HotpotQA prompting baselines (IO, CoT, CoT-SC, Self-Refine) on our test split.

MASkills copies these baselines from AFlow (GPT-4o-mini, 800 random
questions). Here they run with our model on our greedy-evaluation questions,
so they are comparable with the skill libraries:

- same questions as GRSE.evaluate (configs/grse/hotpotqa/test500.json),
- same passages in the prompt (distractor setting, as the agents see them),
- same answer-format instruction as the agents' role (shortest faithful form),
- same metric (upstream HotpotQA token F1 via compute_qa_metrics).

Method settings follow AFlow's description: CoT-SC samples 5 answers and takes
the majority; Self-Refine runs at most 3 feedback/refine rounds. Everything is
greedy (temperature 0) except the CoT-SC samples (--sc-temperature).

    python scripts/eval_hotpotqa_prompting.py --methods io cot cot_sc self_refine
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from GRSE.client import SamplingLLMClient  # noqa: E402
from GRSE.evaluate import build_env, task_key  # noqa: E402
from maskills.envs.language.metrics import compute_qa_metrics  # noqa: E402

# The answer contract the agents receive (maskills/core/policy.py).
FORMAT = ("Your FINAL answer must be the shortest faithful form of the answer (an entity name, "
          "a year, or a short phrase), with no explanation or surrounding sentence.")
TASK = "Context:\n{context}\n\nQuestion: {question}\n\n"
IO = TASK + FORMAT + " Output ONLY the answer."
COT = TASK + ("Think step by step about which passages answer the question, then write the answer "
              "on the last line as 'FINAL ANSWER: <answer>'. " + FORMAT)
FEEDBACK = TASK + ("A proposed solution:\n{solution}\n\nCheck it against the context: is the final "
                   "answer correct, supported by the passages, and in the shortest faithful form? "
                   "If it is, reply with exactly 'CORRECT'. Otherwise explain concisely what is wrong.")
REFINE = TASK + ("Previous solution:\n{solution}\n\nFeedback:\n{feedback}\n\nWrite an improved "
                 "solution: reason step by step, then put the answer on the last line as "
                 "'FINAL ANSWER: <answer>'. " + FORMAT)
SC_SAMPLES, REFINE_ROUNDS = 5, 3


def final_answer(text):
    found = re.findall(r"FINAL ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if found:
        return found[-1].strip().strip("*").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize(answer):
    return " ".join(re.sub(r"[^\w\s]", " ", answer.lower()).split())


class Runner:
    def __init__(self, greedy, sampler):
        self.greedy, self.sampler = greedy, sampler

    def ask(self, client, prompt, max_tokens):
        text, tokens = client.chat_messages_with_usage([{"role": "user", "content": prompt}],
                                                      max_tokens=max_tokens)
        return text, tokens.get("input", 0) + tokens.get("output", 0)

    def io(self, task):
        text, tok = self.ask(self.greedy, IO.format(**task), 64)
        return text.strip(), 1, tok

    def cot(self, task, client=None):
        text, tok = self.ask(client or self.greedy, COT.format(**task), 1024)
        return final_answer(text), 1, tok, text

    def cot_sc(self, task):
        votes, tokens, first = [], 0, {}
        for _ in range(SC_SAMPLES):
            answer, _, tok, _ = self.cot(task, self.sampler)
            tokens += tok
            votes.append(normalize(answer))
            first.setdefault(normalize(answer), answer)
        best = Counter(votes).most_common(1)[0][0]  # ties: first answer to reach the top count
        return first[best], SC_SAMPLES, tokens

    def self_refine(self, task):
        answer, calls, tokens, solution = self.cot(task)
        for _ in range(REFINE_ROUNDS):
            feedback, tok = self.ask(self.greedy, FEEDBACK.format(solution=solution, **task), 512)
            calls, tokens = calls + 1, tokens + tok
            if feedback.strip().upper().startswith("CORRECT"):
                break
            solution, tok = self.ask(self.greedy, REFINE.format(solution=solution, feedback=feedback, **task), 1024)
            calls, tokens = calls + 1, tokens + tok
            answer = final_answer(solution)
        return answer, calls, tokens


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["io", "cot", "cot_sc", "self_refine"],
                   choices=["io", "cot", "cot_sc", "self_refine"])
    p.add_argument("--config", default="configs/grse/hotpotqa/test500.json")
    p.add_argument("--sc-temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="first N questions (smoke test)")
    p.add_argument("--out", default="experiments/eval/hotpotqa_prompting_temp0.json")
    args = p.parse_args(argv)

    env, config = build_env(args.config, temperature=0.0)
    runner = Runner(env.llm_client, SamplingLLMClient(config.get_actor_llm(), args.sc_temperature))
    tasks = env.sample_tasks(config.n_test, split="test")
    if args.limit:
        tasks = tasks[: args.limit]
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}

    for method in args.methods:
        entry = results.setdefault(method, {"protocol": f"{method}, temperature 0"
                                            + (f", samples at {args.sc_temperature}" if method == "cot_sc" else ""),
                                            "scores": {}, "elapsed_sec": 0.0})
        todo = [t for t in tasks if task_key(t) not in entry["scores"]]
        print(f"{method}: {len(tasks)} questions, {len(todo)} to run", flush=True)
        fn = getattr(runner, method)

        def run(task):
            try:
                answer, calls, tokens = fn(task)[:3]
            except Exception as exc:  # noqa: BLE001
                print(f"  {task_key(task)} failed: {type(exc).__name__}: {exc}", flush=True)
                return task_key(task), {"error": f"{type(exc).__name__}: {exc}"[:300]}
            metrics = compute_qa_metrics(answer, task.get("ground_truth", ""))
            return task_key(task), {"reward": metrics["f1"], "em": metrics["em"], "answer": answer,
                                    "gold": task.get("ground_truth", ""), "calls": calls, "tokens": tokens}

        started = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for key, score in executor.map(run, todo):
                entry["scores"][key] = score
        entry["elapsed_sec"] += time.time() - started
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        done = [entry["scores"][task_key(t)] for t in tasks if "error" not in entry["scores"].get(task_key(t), {"error": 1})]
        if done:
            print(f"{method}: F1 {100 * fmean(s['reward'] for s in done):.1f}  EM {100 * fmean(s['em'] for s in done):.1f}  "
                  f"calls/question {fmean(s['calls'] for s in done):.1f}  tokens/question {fmean(s['tokens'] for s in done):.0f}  "
                  f"time {entry['elapsed_sec'] / 60:.1f} min (n={len(done)})", flush=True)


if __name__ == "__main__":
    main()
