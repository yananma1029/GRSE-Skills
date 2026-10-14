#!/usr/bin/env bash
# Cross-task skill transfer (Qwen3.8 actor): do learned skills fit only their task?
#   h2l  the final HotpotQA libraries (GRSE and MASkills, seeds 42 43 44) tested on every
#        LoCoMo category, next to the empty library
#   l2h  the final LoCoMo multi-hop libraries (GRSE and MASkills, seeds 42 43 44) tested on the
#        500 HotpotQA test questions, next to the empty library
# A transferred library keeps the target task's agent roles (role.md of the target's iteration-0
# checkpoint) and takes only skill_library.json from the source, agent by agent.
# Needs the Qwen runs of scripts/run_main.sh (HotpotQA and LoCoMo, grse and maskills).
#
#   bash scripts/run_transfer.sh --jobs "h2l l2h"
#   python analysis/transfer.py                       # F1 summary
#
# Results: experiments/eval/transfer_h2l_cat<c>.json, experiments/eval/transfer_l2h.json;
# the assembled libraries are kept in experiments/transfer/.

source "$(dirname "$0")/common.sh"
JOBS="h2l l2h"; SEEDS="42 43 44"; WORKERS=32
while [ $# -gt 0 ]; do
  case $1 in
    --jobs) JOBS=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
check_server "$QWEN_URL"
T=experiments/transfer

# run_of <log>: run directory recorded in logs/<log>
run_of() {
  local line
  line=$(grep -m1 -oE "(GRPO|MASKILLS) run: [^ ]+" "logs/$1") || { echo "no run id in logs/$1" >&2; exit 1; }
  case $line in
    GRPO*) echo "experiments/grpo/runs/${line##* }" ;;
    *) echo "experiments/runs/${line##* }" ;;
  esac
}

# transfer <out_dir> <target_iter0_checkpoint> <source_final_checkpoint>
transfer() {
  local out=$1 target=$2 source=$3
  for a in agent_1 agent_2; do
    mkdir -p "$out/$a"
    cp "$target/$a/role.md" "$out/$a/role.md"
    cp "$source/$a/skill_library.json" "$out/$a/skill_library.json"
  done
}

for job in $JOBS; do
  case $job in
    h2l)
      for c in 1 2 3 4; do
        target=$(run_of "locomo_cat${c}_grse_qwen_s42.out")/checkpoints/iter_0
        libs=(--library "empty=$target")
        for s in $SEEDS; do
          transfer "$T/h2l_cat$c/grse_s$s" "$target" "$(run_of "hotpotqa_grse_qwen_s$s.out")/checkpoints/iter_5"
          transfer "$T/h2l_cat$c/maskills_s$s" "$target" "$(run_of "hotpotqa_maskills_qwen_s$s.out")/checkpoints/iter_5"
          libs+=(--library "grse_s$s=$T/h2l_cat$c/grse_s$s" --library "maskills_s$s=$T/h2l_cat$c/maskills_s$s")
        done
        run "transfer_h2l_cat$c.out" \
          python -m GRSE.evaluate --config "$(run_config "configs/grse/locomo/cat${c}_grse.json" 42 qwen 0)" \
            "${libs[@]}" --workers "$WORKERS" --out "experiments/eval/transfer_h2l_cat$c.json"
      done ;;
    l2h)
      target=$(run_of hotpotqa_grse_qwen_s42.out)/checkpoints/iter_0
      libs=(--library "empty=$target")
      for s in $SEEDS; do
        transfer "$T/l2h/grse_s$s" "$target" "$(run_of "locomo_cat1_grse_qwen_s$s.out")/checkpoints/iter_5"
        transfer "$T/l2h/maskills_s$s" "$target" "$(run_of "locomo_cat1_maskills_qwen_s$s.out")/checkpoints/iter_5"
        libs+=(--library "grse_s$s=$T/l2h/grse_s$s" --library "maskills_s$s=$T/l2h/maskills_s$s")
      done
      run transfer_l2h.out \
        python -m GRSE.evaluate --config "$(run_config configs/grse/hotpotqa/test500.json 42 qwen 0)" \
          "${libs[@]}" --workers "$WORKERS" --out experiments/eval/transfer_l2h.json ;;
    *) echo "unknown job $job" ;;
  esac
done
echo "$(date '+%F %T') all done"
