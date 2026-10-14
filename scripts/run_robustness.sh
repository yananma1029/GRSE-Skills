#!/usr/bin/env bash
# Robustness to a perturbed skill library (LoCoMo single-hop, Qwen3.8 actor):
# every method starts from the same perturbed seed library, in which each agent keeps one helpful
# skill and receives two harmful or irrelevant ones (configs/grse/robustness/seeds/perturbed;
# the kind of every seed skill is listed in its manifest.json).
#   eval_seeds  greedy test score of the perturbed library and of its helpful skills alone (run once)
#   grse        GRSE (config cat4_grse.json)
#   grse_wo     GRSE w/o structured failure analysis (config cat4_grse_wo_structure.json; not run by default)
#   maskills    MASkills
# One run per method by default (--seeds 42); --seeds "42 43 44" repeats them.
#
#   bash scripts/run_robustness.sh --jobs "eval_seeds grse maskills" --seeds 42
#
# GRSE runs: experiments/grpo/runs/; MASkills runs: experiments/runs/; reference scores:
# experiments/eval/locomo_cat4_robust_temp0.json. Per-iteration skill utilities are stored in
# each run's checkpoints/iter_*/agent_*/skill_library.json.

source "$(dirname "$0")/common.sh"
JOBS="eval_seeds grse maskills"; SEEDS=42; SMOKE=0
SEED_ROOT=configs/grse/robustness/seeds
while [ $# -gt 0 ]; do
  case $1 in
    --jobs) JOBS=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
check_server "$QWEN_URL"
QWEN=$(llm_config qwen)
SMK=$([ "$SMOKE" = 1 ] && echo _smoke)

case " $JOBS " in *" eval_seeds "*)
  run "robust_eval_seeds$SMK.out" \
    python -m GRSE.evaluate --config "$(run_config configs/grse/locomo/cat4_grse.json 42 qwen "$SMOKE")" \
      --temperature 0 --library "perturbed=$SEED_ROOT/perturbed/grpo" --library "helpful=$SEED_ROOT/helpful/grpo" \
      --out "experiments/eval/locomo_cat4_robust_temp0$SMK.json" ;;
esac

for seed in $SEEDS; do
  for job in $JOBS; do
    case $job in
      grse|grse_wo)
        template=configs/grse/robustness/cat4_$([ $job = grse ] && echo grse || echo grse_wo_structure).json
        run "robust_${job}_s$seed$SMK.out" \
          python -m GRSE.train --config "$(run_config "$template" "$seed" qwen "$SMOKE")" --init-skills "$SEED_ROOT/perturbed/grpo" ;;
      maskills)
        size=(--iters 5)
        [ "$SMOKE" = 1 ] && size=(--iters 1 --train-per-iter 2 --n-val 2 --n-test 3)
        run "robust_maskills_s$seed$SMK.out" \
          python scripts/train_locomo_format_repair.py --category 4 "${size[@]}" \
            --init-skills "$SEED_ROOT/perturbed/maskills" \
            --actor-model "$QWEN" --optimizer-model "$QWEN" --gate-tolerance 0.02 ;;
      eval_seeds) ;;
      *) echo "unknown job $job" ;;
    esac
  done
done
echo "$(date '+%F %T') all done"
