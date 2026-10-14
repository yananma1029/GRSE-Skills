#!/usr/bin/env python3
"""Upstream ``train_hotpotqa.py`` with GRSE's output-format handling.

Identical to scripts/train_hotpotqa.py (same arguments) except that
GRSE/maskills_format.py is installed first, so MASkills gets the same SKILL.md
repair, frontmatter instruction, 1200-token skill budget and JSON-mode critic
as the GRSE runs.

    OPENAI_API_KEY=EMPTY python scripts/train_hotpotqa_format_repair.py \
        --n-train 100 --n-val 50 --n-test 100 --iters 5 \
        --actor-model configs/llm/qwen3.8-27b-local.json --tag qwen38_empty_fmt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GRSE import maskills_format  # noqa: E402
from maskills.experiments.language import LanguageExperiment  # noqa: E402

maskills_format.install()
maskills_format.route_critic_to_optimizer(LanguageExperiment)
maskills_format.announce_run_id(LanguageExperiment)

import train_hotpotqa  # noqa: E402

if __name__ == "__main__":
    train_hotpotqa.main()
