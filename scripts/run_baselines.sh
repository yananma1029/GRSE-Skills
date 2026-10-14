#!/usr/bin/env bash
# Single-model baselines (no agents, no skills):
#   hotpotqa_prompting   IO, CoT, CoT-SC (5 samples) and Self-Refine on the 500 HotpotQA test questions
#   locomo_fullcontext   LoCoMo full-context baseline, every category
# Switch the actor with --actor qwen|gemma.
#
#   bash scripts/run_baselines.sh --actor qwen --jobs "hotpotqa_prompting locomo_fullcontext"
#   bash scripts/run_baselines.sh --smoke                  # 3 questions per setting
#
# Results: experiments/eval/hotpotqa_prompting_<actor>.json, experiments/eval/locomo_fullcontext_<actor>.json.

source "$(dirname "$0")/common.sh"
ACTOR=qwen; JOBS="hotpotqa_prompting locomo_fullcontext"; SMOKE=0
while [ $# -gt 0 ]; do
  case $1 in
    --actor) ACTOR=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
check_server "$QWEN_URL"; [ "$ACTOR" = gemma ] && check_server "$GEMMA_URL"
LIMIT=(); [ "$SMOKE" = 1 ] && LIMIT=(--limit 3)
SMK=$([ "$SMOKE" = 1 ] && echo _smoke)

for job in $JOBS; do
  case $job in
    hotpotqa_prompting)
      run "hotpotqa_prompting_$ACTOR$SMK.out" \
        python scripts/eval_hotpotqa_prompting.py --config "$(run_config configs/grse/hotpotqa/test500.json 42 "$ACTOR" 0)" \
          "${LIMIT[@]}" --out "experiments/eval/hotpotqa_prompting_$ACTOR$SMK.json" ;;
    locomo_fullcontext)
      # one config per category; the script fills {} with the category number
      for c in 1 2 3 4; do run_config "configs/grse/locomo/cat${c}_grse_wo_structure.json" 42 "$ACTOR" 0 > /dev/null; done
      run "locomo_fullcontext_$ACTOR$SMK.out" \
        python scripts/eval_locomo_fullcontext.py --categories 1 2 3 4 \
          --config-template "configs/runtime/cat{}_grse_wo_structure_${ACTOR}_s42.json" \
          "${LIMIT[@]}" --out "experiments/eval/locomo_fullcontext_$ACTOR$SMK.json" ;;
    *) echo "unknown job $job" ;;
  esac
done
echo "$(date '+%F %T') all done"
