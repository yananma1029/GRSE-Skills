#!/usr/bin/env bash
# Ablations (Qwen3.8 actor, one run each, seed 42):
#   LoCoMo Mix, one library shared by the four categories:
#     mix_grse  mix_wo_structure  mix_g1  mix_dapo_filter  mix_with_label
#   HotpotQA:
#     hotpotqa_g1  hotpotqa_with_label  hotpotqa_dapo_filter_wo_structure
# G=1 samples one rollout per question; the DAPO filter sends only groups whose rewards
# differ to the critic; w/ label conditions the critic on the question category.
#
#   bash scripts/run_ablation.sh --variants "mix_grse mix_wo_structure mix_g1 mix_dapo_filter mix_with_label"
#   bash scripts/run_ablation.sh --smoke --variants mix_g1
#
# Mix runs test their final library on the 842 LoCoMo test questions themselves; HotpotQA
# variants are then scored greedily on the 500 test questions into
# experiments/eval/hotpotqa_test500_ablation_<actor>_s<seed>.json. --actor and --seed switch the
# actor and the seed.

source "$(dirname "$0")/common.sh"
ACTOR=qwen; SEED=42; SMOKE=0
VARIANTS="mix_grse mix_wo_structure mix_g1 mix_dapo_filter mix_with_label hotpotqa_g1 hotpotqa_with_label hotpotqa_dapo_filter_wo_structure"
while [ $# -gt 0 ]; do
  case $1 in
    --variants) VARIANTS=$2; shift 2 ;;
    --actor) ACTOR=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
check_server "$QWEN_URL"; [ "$ACTOR" = gemma ] && check_server "$GEMMA_URL"
SMK=$([ "$SMOKE" = 1 ] && echo _smoke)

for v in $VARIANTS; do
  template=configs/grse/ablation/$v.json
  [ -f "$template" ] || { echo "unknown variant $v"; continue; }
  run "ablation_${v}_${ACTOR}_s$SEED$SMK.out" \
    python -m GRSE.train --config "$(run_config "$template" "$SEED" "$ACTOR" "$SMOKE")" --init-skills empty || continue
  case $v in
    hotpotqa_*)
      [ -z "$RUN_ID" ] && { echo "no run id for $v; skipping evaluation"; continue; }
      final=$([ "$SMOKE" = 1 ] && echo iter_1 || echo iter_5)
      run "ablation_${v}_${ACTOR}_s$SEED${SMK}_eval.out" \
        python -m GRSE.evaluate --config "$(run_config configs/grse/hotpotqa/test500.json "$SEED" "$ACTOR" "$SMOKE")" \
          --temperature 0 --library "$v=experiments/grpo/runs/$RUN_ID/checkpoints/$final" \
          --out "experiments/eval/hotpotqa_test500_ablation_${ACTOR}_s$SEED$SMK.json" ;;
  esac
done
echo "$(date '+%F %T') all done"
