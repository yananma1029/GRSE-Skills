<div align="center">

# GRSE-Skills

**Which Agent's Skill Should Change? Group-Relative Credit Assignment for Evolving Multi-Agent Teams**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[**Quick Start**](#quick-start) · [**Key Components**](#key-components) · [**Experiments**](#experiments) · [**Configuration**](#configuration) · [**Project Structure**](#project-structure)

</div>

When a team of agents answers a question wrongly, the failure can start with any agent, and a
single trajectory rarely shows which agent's skill was at fault. GRSE executes each question
several times under the same skill libraries, compares the executions within the group, assigns
credit to individual agent–skill pairs with a structured failure analysis, and turns the
aggregated credit into validated skill updates. Language-model weights stay fixed.

---

## Overview

| MASkills | GRSE |
|---|---|
| One trajectory per question | A group of `G` trajectories per question under the same libraries |
| Each trajectory is credited on its own | Executions are compared within the group (group-relative advantage) |
| Free-form credit from the critic | Structured failure analysis: the failing step, the responsible agent and skill, checked against the trajectory |
| A skill present in every trajectory can collect credit for successes | Credit follows the behavior that separates better from worse executions |

Skill updates use the same four operators (induct, refine, consolidate, prune) and the same
validation gate as MASkills, so the comparison isolates how credit is assigned.

---

## Key Components

1. **Grouped rollouts** -- every training question is executed `G` times (default 4) by the whole
   team with the same skill libraries; each execution receives a group-relative advantage.
2. **Group-relative critic** -- the critic sees the whole group and attributes the outcome to
   individual agent–skill pairs (helped, harmful, neutral, redundant).
3. **Structured failure analysis** -- for each low-advantage execution, the critic names the
   failing step, the responsible agent and skill; references to steps, agents and skills are
   checked against the trajectory and repaired once if invalid.
4. **Credit aggregation and validated updates** -- credits are aggregated per skill across the
   iteration's groups into utilities and edit directions, the scheduled operators produce a
   candidate library, and a held-out validation gate commits it or rolls it back.

One iteration:

```
G rollouts per question under K_t -> group-relative advantages -> critic: credit per agent-skill pair
  -> aggregate per skill -> operators -> candidate K_t+1 -> validation gate -> commit or roll back
```

---

## Quick Start

### Installation

```bash
pip install -e ".[language,analysis]"
```

### Models

All models are served locally with vLLM. Qwen3.8-27B is the critic and optimizer in every run and
one of the two actors; Gemma-4-26B-A4B is the other actor.

```bash
vllm serve Qwen/Qwen3.8-27B --port 18001 --tensor-parallel-size 2 --max-model-len 65536 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
vllm serve google/gemma-4-26B-A4B-it --port 18002 --tensor-parallel-size 2 --max-model-len 65536 \
  --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking": false}'
```

If the servers run elsewhere, set `QWEN_URL` and `GEMMA_URL` (default `http://localhost:18001/v1`
and `http://localhost:18002/v1`).

### Benchmark data

```bash
python scripts/fetch_hotpotqa.py                                # HotpotQA (distractor, dev set)
git clone https://github.com/snap-research/locomo env/locomo    # LoCoMo
```

### Train and evaluate one configuration

```bash
python -m GRSE.train --config configs/grse/hotpotqa/grse.json --init-skills empty
python -m GRSE.evaluate --config configs/grse/hotpotqa/test500.json \
  --library grse=experiments/grpo/runs/<run>/checkpoints/iter_5 --out eval.json
```

---

## Experiments

| Experiment | Script | What it runs |
|---|---|---|
| Main results | `scripts/run_main.sh` | GRSE, GRSE w/o structure, MASkills on HotpotQA and each LoCoMo category |
| Single-model baselines | `scripts/run_baselines.sh` | IO, CoT, CoT-SC, Self-Refine (HotpotQA); full context (LoCoMo) |
| Ablations | `scripts/run_ablation.sh` | G=1, DAPO filter, w/o structure, w/ label |
| Robustness | `scripts/run_robustness.sh` + `analysis/robustness.py` | repair of a perturbed library; credit for the planted skills |
| Cross-task transfer | `scripts/run_transfer.sh` + `analysis/transfer.py` | HotpotQA libraries on LoCoMo, LoCoMo multi-hop libraries on HotpotQA |
| Learned skills | `analysis/learned_skills.py` | answer changes from the empty to the final library; skill types |

`--actor qwen|gemma` selects the actor; `--smoke` runs a quick check (one iteration, a few
questions) for the main, baseline, ablation and robustness scripts.

```bash
# main results, three seeds per actor
bash scripts/run_main.sh --actor qwen  --methods "grse grse_wo maskills" --seeds "42 43 44"
bash scripts/run_main.sh --actor gemma --methods "grse grse_wo maskills" --seeds "42 43 44"
bash scripts/run_baselines.sh --actor qwen

# ablations (LoCoMo with one shared library, and HotpotQA)
bash scripts/run_ablation.sh

# robustness: both methods start from a perturbed library (configs/grse/robustness/seeds/)
bash scripts/run_robustness.sh
python analysis/robustness.py

# cross-task transfer (needs the Qwen runs of run_main.sh)
bash scripts/run_transfer.sh
python analysis/transfer.py

# learned skills
python analysis/learned_skills.py transitions
python analysis/learned_skills.py skill-types      # labels each final skill with Qwen3.8
```

Per-iteration LoCoMo curves: `notebooks/locomo_iteration_curves.ipynb`. The trajectories behind
the case studies are in `case_studies/`.

---

## Configuration

A GRSE config is a MASkills config (`base`, `llm`, ...) plus a `grpo` block:

| Field | Default | Meaning |
|---|---|---|
| `rollouts_per_question` | `4` | Group size `G` |
| `questions_per_iteration` | -- | Training questions per iteration |
| `structured_failures` | `false` | Structured failure analysis (`true` in the GRSE configs) |
| `category_analysis` | `false` | Condition the critic on the question category (w/ label ablation) |
| `group_filter` | `none` | `zero_variance` drops groups whose rewards are all equal (DAPO filter ablation) |
| `single_rollout` | `false` | One rollout per question, no group contrast (G=1 ablation) |
| `validation_rollouts` | `1` | Executions per validation question in the gate |
| `seed` / `split_seed` | `42` | Question sampling / LoCoMo conversation split |

Skill-library settings come from MASkills (`base` block): `max_skills_per_agent`,
`skill_eval_delta` (gate tolerance) and `refine_every` / `induct_every` / `consolidate_every` /
`prune_every` (operator schedule).

---

## Project Structure

```
GRSE/                    GRSE implementation
├── train.py             entry point (python -m GRSE.train)
├── evaluate.py          greedy evaluation of saved libraries (python -m GRSE.evaluate)
├── trainer.py           training loop: grouped rollouts, credit, aggregation, updates, gate
├── group.py             rollout groups and group-relative advantages
├── critic.py            group-relative critic
├── failure_analysis.py  structured failure analysis and its checks
├── optimizer.py         skill-update operators
└── config.py            the grpo block
maskills/                multi-agent framework, HotpotQA and LoCoMo environments (MASkills)
configs/grse/            experiment configs: hotpotqa/, locomo/, ablation/, robustness/
configs/llm/             model endpoints
scripts/                 experiment scripts (run_*.sh), baselines, MASkills baseline runners
analysis/                analyses of finished runs
case_studies/            trajectories of the case studies
notebooks/               per-iteration curves
```

---

## Acknowledgements

We acknowledge [MASkills](https://github.com/DaRL-GenAI/MASkills), on which this code builds:
`maskills/` provides the multi-agent framework, the HotpotQA and LoCoMo environments and the
MASkills baseline. Our main changes are the GRSE training method in `GRSE/` (grouped rollouts,
group-relative critic, structured failure analysis, credit aggregation and validated skill
updates) and the same skill-file format handling for both methods.

---

## License

Released under the [MIT License](LICENSE).
