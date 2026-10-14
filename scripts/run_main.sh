#!/usr/bin/env bash
# Main results: GRSE, GRSE w/o structured
# failure analysis, and MASkills on HotpotQA and on each LoCoMo category, three runs each.
# Switch the actor with --actor qwen|gemma and the runs with --seeds (default 42 43 44).
#
#   bash scripts/run_main.sh --actor qwen --bench "hotpotqa locomo" --methods "grse grse_wo maskills" --seeds "42 43 44"
#   bash scripts/run_main.sh --actor gemma --bench locomo --methods grse --seeds 42 --cats "1 4"
#   bash scripts/run_main.sh --smoke --bench hotpotqa --methods grse --seeds 42      # 1 iteration, a few questions
#
# GRSE runs: experiments/grpo/runs/<exp_name>_<time>/; MASkills runs: experiments/runs/.
# LoCoMo runs test every iteration's library on the full test split themselves. HotpotQA
# libraries are then scored greedily on the 500 test questions into
# experiments/eval/hotpotqa_test500_<actor>_s<seed>.json (seed 42 also scores the empty library).
# MASkills has no seed option; --seeds only numbers its repeated runs.

source "$(dirname "$0")/common.sh"
ACTOR=qwen; BENCH="hotpotqa locomo"; METHODS="grse grse_wo maskills"; SEEDS="42 43 44"; CATS="1 2 3 4"; SMOKE=0
while [ $# -gt 0 ]; do
  case $1 in
    --actor) ACTOR=$2; shift 2 ;;
    --bench) BENCH=$2; shift 2 ;;
    --methods) METHODS=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --cats) CATS=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
check_server "$QWEN_URL"; [ "$ACTOR" = gemma ] && check_server "$GEMMA_URL"
QWEN=$(llm_config qwen); ACTOR_LLM=$(llm_config "$ACTOR")
# The Qwen MASkills runs take the optimizer from --actor-model; the Gemma runs keep Qwen as optimizer.
OPTIMIZER_ARG=(); [ "$ACTOR" = gemma ] && OPTIMIZER_ARG=(--optimizer-model "$QWEN")
SMK=$([ "$SMOKE" = 1 ] && echo _smoke)

# eval_hotpotqa <name> <runs root> <seed>: greedy test-set score of the run just trained
eval_hotpotqa() {
  [ -z "$RUN_ID" ] && { echo "no run id for $1; skipping evaluation"; return; }
  local libs=(--library "$1=$2/$RUN_ID/checkpoints/iter_5")
  [ "$SMOKE" = 1 ] && libs=(--library "$1=$2/$RUN_ID/checkpoints/iter_1")
  [ "$3" = 42 ] && libs=(--library "empty=$2/$RUN_ID/checkpoints/iter_0" "${libs[@]}")
  run "hotpotqa_${1}_${ACTOR}_s$3_eval$SMK.out" \
    python -m GRSE.evaluate --config "$(run_config configs/grse/hotpotqa/test500.json "$3" "$ACTOR" "$SMOKE")" \
      --temperature 0 "${libs[@]}" --out "experiments/eval/hotpotqa_test500_${ACTOR}_s$3$SMK.json"
}

for seed in $SEEDS; do
  for bench in $BENCH; do
    for method in $METHODS; do
      case $bench/$method in
        hotpotqa/grse|hotpotqa/grse_wo)
          template=configs/grse/hotpotqa/$([ $method = grse ] && echo grse || echo grse_wo_structure).json
          run "hotpotqa_${method}_${ACTOR}_s$seed$SMK.out" \
            python -m GRSE.train --config "$(run_config "$template" "$seed" "$ACTOR" "$SMOKE")" --init-skills empty \
          && eval_hotpotqa "$method" experiments/grpo/runs "$seed" ;;
        hotpotqa/maskills)
          size=(--n-train 100 --n-val 50 --n-test 100 --iters 5)
          [ "$SMOKE" = 1 ] && size=(--n-train 2 --n-val 2 --n-test 3 --iters 1)
          run "hotpotqa_maskills_${ACTOR}_s$seed$SMK.out" \
            python scripts/train_hotpotqa_format_repair.py "${size[@]}" \
              --actor-model "$ACTOR_LLM" "${OPTIMIZER_ARG[@]}" --tag "${ACTOR}_empty_fmt_s$seed$SMK" \
          && eval_hotpotqa maskills experiments/runs "$seed" ;;
        locomo/grse|locomo/grse_wo)
          for c in $CATS; do
            template=configs/grse/locomo/cat${c}_$([ $method = grse ] && echo grse || echo grse_wo_structure).json
            run "locomo_cat${c}_${method}_${ACTOR}_s$seed$SMK.out" \
              python -m GRSE.train --config "$(run_config "$template" "$seed" "$ACTOR" "$SMOKE")" --init-skills empty
          done ;;
        locomo/maskills)
          for c in $CATS; do
            size=(--iters 5)
            [ "$SMOKE" = 1 ] && size=(--iters 1 --train-per-iter 2 --n-val 2 --n-test 3)
            run "locomo_cat${c}_maskills_${ACTOR}_s$seed$SMK.out" \
              python scripts/train_locomo_format_repair.py --category "$c" "${size[@]}" \
                --actor-model "$ACTOR_LLM" --optimizer-model "$QWEN" --gate-tolerance 0.02
          done ;;
        *) echo "unknown benchmark/method $bench/$method" ;;
      esac
    done
  done
done
echo "$(date '+%F %T') all done"
