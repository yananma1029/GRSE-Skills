"""CLI: python -m GRSE.train --config configs/grse/hotpotqa/grse.json --init-skills empty."""

import argparse
import json
from pathlib import Path

from maskills.config.base import LanguageTaskConfig, LocomoConfig
from maskills.core.skills import SkillLibrary
from maskills.envs import make_env

from .client import SamplingLLMClient
from .config import GroupConfig
from .critic import GroupSkillCreditCritic
from .optimizer import GroupSkillOptimizer
from .routing import LLMSkillSelector, SkillRoutingEnvironment
from .store import GroupRunStore
from .trainer import GroupSkillTrainer


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grse/hotpotqa/grse_wo_structure.json")
    parser.add_argument("--init-skills", default="empty", help="empty or root/agent_N/skill_id/SKILL.md")
    parser.add_argument("--output", help="Override experiment_dir")
    parser.add_argument("--resume-run-id", help="Continue a run under output/runs, with identical config")
    parser.add_argument("--demo", action="store_true", help="Synthetic offline demo; no model calls")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.demo:
        if args.resume_run_id or args.init_skills != "empty":
            raise ValueError("Demo starts fresh; use the real runner for seeded/resumed runs")
        from .demo import run_demo
        return run_demo(args.output or "experiments/grpo_demo")
    data = json.loads(Path(args.config).read_text())
    env_name = data.get("env", "language")
    classes = {"language": LanguageTaskConfig, "locomo": LocomoConfig}
    if env_name not in classes:
        raise ValueError("This runner supports language and locomo environments")
    base_data = dict(data["base"])
    base_data.update(skill_evolution=True, inject_skill_library=False)
    if args.output:
        base_data["experiment_dir"] = args.output
    config = classes[env_name].from_dict(base_data)
    options = GroupConfig(**data.get("grpo", {}))
    if config.get_actor_llm() is None:
        raise ValueError("Configure an actor LLM")
    env = make_env(env_name, config)
    # Upstream metric wrappers swallow missing evaluator errors. Preflight
    # here so missing benchmark dependencies cannot silently train on zeros.
    if env_name == "language" and config.task_type == "qa":
        from maskills.envs.language.metrics import compute_qa_metrics
        compute_qa_metrics("test", "test")
    elif env_name == "locomo":
        from maskills.envs.locomo.metrics import compute_locomo_metrics
        compute_locomo_metrics("test", "test", 4)
        counts = data.get("conversation_split", [2, 2, 6])
        env.task_loader.split_by_conversation(*counts, seed=options.split_seed)
    # SamplingLLMClient forwards temperature (upstream client ignores extras).
    env.llm_client = SamplingLLMClient(config.get_actor_llm(), options.temperature)
    if hasattr(env, "reward_gen"):
        env.reward_gen.set_client(env.llm_client.raw_client)
    # Full injection uses upstream execution and its original skill trace.
    routed = env if options.selection_mode == "all" else SkillRoutingEnvironment(
        env, LLMSkillSelector(env.llm_client), options.selection_mode)
    tools = env.get_tool_library() if hasattr(env, "get_tool_library") else ""
    critic_cls = GroupSkillCreditCritic
    if options.structured_failures:
        from .failure_analysis import StructuredSkillCreditCritic
        critic_cls = StructuredSkillCreditCritic
    critic = critic_cls(config, tool_library=tools)
    if options.structured_failures:
        critic.category_analysis = options.category_analysis
    trainer = GroupSkillTrainer(
        config=config, env=routed,
        critic=critic,
        optimizer=GroupSkillOptimizer(config.get_optimizer_llm(), tool_library=tools),
        group_config=options,
        store=GroupRunStore(config.experiment_dir, args.resume_run_id),
    )
    if options.prompt_profile == "locomo":
        from . import locomo
        locomo.install(trainer.optimizer, trainer.critic)
    if not args.resume_run_id:
        policies = trainer.checkpoint.get_policies()
        if env_name == "language" and config.architecture in ("centralized", "hybrid"):
            if config.num_agents != 2 or config.main_agent != "agent_1":
                raise ValueError("Built-in centralized language roles require two agents and main_agent=agent_1")
            from maskills.experiments.hotpotqa import _CENTRALIZED_ROLES
            for agent, policy in policies.items():
                policy.role = _CENTRALIZED_ROLES[agent]
        elif env_name == "locomo":
            for index, (agent, policy) in enumerate(policies.items()):
                policy.role = env._role_for(agent, index)
        if args.init_skills != "empty":
            for agent, policy in policies.items():
                directory = Path(args.init_skills) / agent
                policy.skill_library = SkillLibrary.from_skill_md_dir(directory)
                if not policy.skill_library:
                    raise ValueError(f"No seed skills in {directory}")
        # Preserve topology roles and evaluate the empty baseline as upstream does.
        trainer.checkpoint.save_policies(0, policies, {"initialization": args.init_skills})
    if trainer.store.latest_checkpoint(trainer.run_id) == 0:
        _, initial_meta = trainer.store.load_checkpoint(trainer.run_id, 0)
        if (initial_meta.get("training_stats") or {}).get("initialization") == "empty":
            if not any(m.get("type") == "baseline" for m in trainer.store.load_metrics(trainer.run_id)):
                trainer._run_baseline_round()
    print(f"GRPO run: {trainer.run_id}")
    print(f"Artifacts: {trainer.store.get_run_dir(trainer.run_id)}")
    return trainer.train()


if __name__ == "__main__":
    main()
