# Shared helpers for scripts/run_*.sh (sourced, not run).
#
# Model servers (OpenAI-compatible, e.g. vLLM):
#   QWEN_URL   Qwen3.8-27B: critic and optimizer for every method, actor for --actor qwen
#              (default http://localhost:18001/v1)
#   GEMMA_URL  Gemma-4-31B: actor for --actor gemma (default http://localhost:18002/v1)
#   OPENAI_API_KEY is passed to the servers; local vLLM servers accept any value.
#
# Every run config is written to configs/runtime/ from a template in configs/grse/:
# the seed, the actor and the server URLs are filled in, nothing else changes.

set -u
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

QWEN_URL=${QWEN_URL:-http://localhost:18001/v1}
GEMMA_URL=${GEMMA_URL:-http://localhost:18002/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export MASKILLS_LLM_TIMEOUT=${MASKILLS_LLM_TIMEOUT:-600}   # greedy evaluation can run to the token limit
RUNTIME=configs/runtime
mkdir -p "$RUNTIME" logs experiments/eval

# llm_config <qwen|gemma>: path of that model's LLM config with the server URL filled in
llm_config() {
  local out="$RUNTIME/llm_$1.json"
  case $1 in
    qwen)  sed "s#http://localhost:18001/v1#$QWEN_URL#" configs/llm/qwen3.8-27b-local.json > "$out" ;;
    gemma) sed "s#http://localhost:18002/v1#$GEMMA_URL#" configs/llm/gemma-4-31b-local.json > "$out" ;;
    *) echo "unknown actor $1" >&2; return 1 ;;
  esac
  echo "$out"
}

# run_config <template> <seed> <actor> <smoke 0|1>: path of a run config built from <template>.
#   seed 42 is the template's own seed; another seed is written to grpo.seed and appended
#   to exp_name as _s<seed>. Actor gemma replaces base.actor_llm with the Gemma config and
#   appends _gemma to exp_name (critic and optimizer stay Qwen3.8). Smoke shrinks the run
#   to 1 iteration and a few questions.
run_config() {
  local template=$1 seed=$2 actor=$3 smoke=$4
  local out="$RUNTIME/$(basename "${template%.json}")_${actor}_s${seed}$([ "$smoke" = 1 ] && echo _smoke).json"
  python - "$template" "$seed" "$actor" "$smoke" "$out" "$QWEN_URL" "$GEMMA_URL" <<'PY'
import json, sys
template, seed, actor, smoke, out, qwen_url, gemma_url = sys.argv[1:]
seed, smoke = int(seed), smoke == "1"
c = json.load(open(template))
if actor == "gemma":
    gemma = json.load(open("configs/llm/gemma-4-31b-local.json"))
    c["base"]["exp_name"] += "_gemma"
    c["base"]["actor_llm"] = {k: gemma[k] for k in ("name", "model_string", "base_url", "api_key", "max_tokens")}
if seed != 42:
    c["grpo"]["seed"] = seed
    c["base"]["exp_name"] += f"_s{seed}"
if smoke:
    c["base"].update(exp_name=c["base"]["exp_name"] + "_smoke", num_iterations=1,
                     n_train=2, n_val=2, n_test=3, trajectories_per_iteration=2)
    c["grpo"]["questions_per_iteration"] = 2
text = json.dumps(c, indent=2)
text = text.replace("http://localhost:18001/v1", qwen_url).replace("http://localhost:18002/v1", gemma_url)
open(out, "w").write(text)
PY
  echo "$out"
}

# check_server <url>: stop early if a model server does not answer
check_server() {
  curl -s -m 10 "$1/models" | grep -q '"id"' || { echo "model server not reachable: $1" >&2; exit 1; }
}

# run <log> <command...>: run a command into logs/<log>; sets RUN_ID from the "... run: <id>" line
run() {
  local log="logs/$1"; shift
  local start=$(date +%s)
  echo "$(date '+%F %T') start -> $log"
  "$@" > "$log" 2>&1
  local code=$?
  RUN_ID=$(grep -m1 -oE "(GRPO|MASKILLS) run: [^ ]+" "$log" | awk '{print $3}')
  echo "$(date '+%F %T') done  (exit $code, $(( ($(date +%s) - start) / 60 )) min${RUN_ID:+, run $RUN_ID})"
  return $code
}
