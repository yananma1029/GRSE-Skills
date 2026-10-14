"""Re-score saved skill libraries on one split with a fixed actor temperature.

Every library runs in the same environment on the same questions, so score
differences come from the libraries alone. At temperature 0 the actor is
greedy, which removes the sampling noise that made repeated empty-library
baselines differ by ~0.03 on HotpotQA.

    python -m GRSE.evaluate --config configs/grse/hotpotqa/test500.json \
        --library empty=experiments/grpo/runs/<run>/checkpoints/iter_0 \
        --library dapo=experiments/grpo/runs/<run>/checkpoints/iter_5 \
        --out experiments/eval/hotpotqa_temp0.json

The first library is the reference for the paired comparisons. Results are
saved after each library; rerunning with the same --out skips finished ones.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from statistics import fmean

from maskills.config.base import LanguageTaskConfig, LocomoConfig
from maskills.core.policy import AgentPolicy
from maskills.core.skills import SkillLibrary
from maskills.envs import make_env

from .client import SamplingLLMClient


def build_env(config_path, temperature):
    data = json.loads(Path(config_path).read_text())
    env_name = data.get("env", "language")
    classes = {"language": LanguageTaskConfig, "locomo": LocomoConfig}
    base = dict(data["base"])
    base.update(skill_evolution=True, inject_skill_library=False)
    config = classes[env_name].from_dict(base)
    env = make_env(env_name, config)
    if env_name == "locomo":
        counts = data.get("conversation_split", [2, 2, 6])
        env.task_loader.split_by_conversation(*counts, seed=data.get("grpo", {}).get("split_seed", 42))
    env.llm_client = SamplingLLMClient(config.get_actor_llm(), temperature)
    if hasattr(env, "reward_gen"):
        env.reward_gen.set_client(env.llm_client.raw_client)
    return env, config


def load_policies(checkpoint_dir):
    """Policies from a run checkpoint (agent_N/role.md + skill_library.json), or
    from a seed library (agent_N/<skill>/SKILL.md, as passed to --init-skills).

    Seed libraries carry no role: the policy gets an empty role, which the
    LOCOMO environment fills with its default retriever/reasoner prompt, as it
    does for the upstream per-category runner.
    """
    policies = {}
    for agent_dir in sorted(p for p in Path(checkpoint_dir).iterdir() if p.is_dir()):
        if (agent_dir / "role.md").exists():
            library = SkillLibrary.from_dict(json.loads((agent_dir / "skill_library.json").read_text()))
            policies[agent_dir.name] = AgentPolicy(role=(agent_dir / "role.md").read_text(),
                                                   skill_library=library)
        elif any(agent_dir.glob("*/SKILL.md")):
            policies[agent_dir.name] = AgentPolicy(role="", skill_library=SkillLibrary.from_skill_md_dir(agent_dir))
    if not policies:
        raise ValueError(f"No agent_N/role.md or agent_N/<skill>/SKILL.md under {checkpoint_dir}")
    return policies


def task_key(task):
    return str(task.get("task_id") or task.get("question"))


def evaluate(env, policies, tasks, workers):
    def run(task):
        # One failed question (e.g. a greedy repetition loop outliving the
        # client timeout after retries) must not discard the whole library.
        try:
            traj = env.collect_trajectory({a: p.copy() for a, p in policies.items()}, deepcopy(task))
        except Exception as exc:  # noqa: BLE001
            print(f"  {task_key(task)} failed: {type(exc).__name__}: {exc}", flush=True)
            return task_key(task), {"error": f"{type(exc).__name__}: {exc}"[:300]}
        return task_key(task), {"reward": traj.reward,
                                "answer": traj.metadata.get("final_answer", ""),
                                "gold": task.get("ground_truth", "")}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(run, tasks))


def paired(reference, other, resamples=5000):
    keys = sorted(set(reference) & set(other))
    diffs = [other[k]["reward"] - reference[k]["reward"] for k in keys]
    rng = random.Random(0)
    boots = sorted(fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return {"n": len(keys), "diff": fmean(diffs),
            "ci95": [boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]],
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="GRSE config; supplies env, split and actor")
    parser.add_argument("--library", action="append", required=True, metavar="NAME=CHECKPOINT_DIR")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    libraries = dict(item.split("=", 1) for item in args.library)
    env, config = build_env(args.config, args.temperature)
    count = getattr(config, f"n_{args.split}", 0)
    tasks = env.sample_tasks(count, split=args.split)
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    for name, checkpoint in libraries.items():
        if name in results:
            print(f"{name}: cached")
            continue
        policies = load_policies(checkpoint)
        sizes = {a: len(p.skill_library) for a, p in policies.items()}
        print(f"{name}: {len(tasks)} {args.split} tasks, skills {sizes}", flush=True)
        scores = evaluate(env, policies, tasks, args.workers)
        results[name] = {"checkpoint": checkpoint, "temperature": args.temperature,
                         "split": args.split, "skills": sizes, "scores": scores}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1))

    names = list(libraries)
    # Compare only questions every library answered, so means stay paired.
    valid = set.intersection(*({k for k, s in results[n]["scores"].items() if "error" not in s}
                               for n in names))
    for name in names:
        failed = len(results[name]["scores"]) - len(valid & set(results[name]["scores"]))
        if failed:
            print(f"{name}: {failed} question(s) excluded (failed here or in another library)")
    reference = {k: results[names[0]]["scores"][k] for k in valid}
    print(f"\n{len(valid)} questions compared")
    print(f"{'library':16s} {'mean':>6s}   vs {names[0]}: diff [95% CI]  better/worse")
    for name in names:
        scores = {k: results[name]["scores"][k] for k in valid}
        line = f"{name:16s} {fmean(s['reward'] for s in scores.values()):6.3f}"
        if name != names[0]:
            p = paired(reference, scores)
            line += (f"   {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
                     f"  {p['better']}/{p['worse']}")
        print(line)


if __name__ == "__main__":
    main()
