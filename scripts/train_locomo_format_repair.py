#!/usr/bin/env python3
"""Upstream per-category ``train_locomo.py`` with GRSE's output-format handling.

Identical to scripts/train_locomo.py (same arguments) except that
GRSE/maskills_format.py is installed first, so MASkills gets the same SKILL.md
repair, frontmatter instruction, 1200-token skill budget and JSON-mode critic
as the GRSE runs. The LOCOMO prompt patches go in before the format handling,
which appends to them; the runner's own later call is then a no-op.

    OPENAI_API_KEY=EMPTY python scripts/train_locomo_format_repair.py --category 2 \
        --actor-model configs/llm/qwen3.8-27b-local.json \
        --optimizer-model configs/llm/qwen3.8-27b-local.json --gate-tolerance 0.02
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GRSE import maskills_format  # noqa: E402
from maskills.experiments import locomo_percat  # noqa: E402
from maskills.experiments.locomo import LocomoExperiment  # noqa: E402

locomo_percat.install_runtime_patches()
maskills_format.install()
maskills_format.route_critic_to_optimizer(LocomoExperiment)
maskills_format.announce_run_id(LocomoExperiment)

import train_locomo  # noqa: E402

if __name__ == "__main__":
    train_locomo.main()
