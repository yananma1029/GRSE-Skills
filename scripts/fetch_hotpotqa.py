"""Fetch HotpotQA dev-distractor into the layout the QA loader expects.

The official mirror (curtis.ml.cmu.edu) is frequently unreachable, so this
pulls the same split from the Hugging Face mirror ``hotpotqa/hotpot_qa``
(``distractor`` config, ``validation`` split) and rewrites it to the original
``hotpot_dev_distractor_v1.json`` list format::

    [{"_id", "question", "answer", "type", "level",
      "supporting_facts": [[title, sent_id], ...],
      "context": [[title, [sentence, ...]], ...]}, ...]

Usage::

    python scripts/fetch_hotpotqa.py                # -> env/lang_benchmark/HotPotQA/
    python scripts/fetch_hotpotqa.py --out some/dir

Requires ``pyarrow`` (``pip install pyarrow``). Also clones the upstream
``hotpotqa/hotpot`` repo alongside, which the metrics need for the official
EM/F1 scorer, unless ``--no-scorer`` is given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HF_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
SCORER_REPO = "https://github.com/hotpotqa/hotpot.git"
OUT_NAME = "hotpot_dev_distractor_v1.json"


def _to_list(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def convert(parquet_path: Path, json_path: Path) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    items = []
    for row in table.to_pylist():
        sf = row["supporting_facts"]
        ctx = row["context"]
        items.append({
            "_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "supporting_facts": [
                [t, int(s)] for t, s in zip(_to_list(sf["title"]), _to_list(sf["sent_id"]))
            ],
            "context": [
                [t, _to_list(s)] for t, s in zip(_to_list(ctx["title"]), _to_list(ctx["sentences"]))
            ],
        })
    json_path.write_text(json.dumps(items, ensure_ascii=False))
    return len(items)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="env/lang_benchmark/HotPotQA",
                   help="target directory (default: env/lang_benchmark/HotPotQA)")
    p.add_argument("--no-scorer", action="store_true",
                   help="skip cloning hotpotqa/hotpot (official EM/F1 scorer)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / OUT_NAME

    if json_path.exists():
        print(f"[skip] {json_path} already exists")
    else:
        parquet_path = out / "validation.parquet"
        print(f"[download] {HF_URL}")
        urllib.request.urlretrieve(HF_URL, parquet_path)
        n = convert(parquet_path, json_path)
        parquet_path.unlink()
        print(f"[ok] wrote {n} questions -> {json_path}")

    scorer_dir = out / "hotpot"
    if args.no_scorer:
        pass
    elif (scorer_dir / "hotpot_evaluate_v1.py").exists():
        print(f"[skip] {scorer_dir} already present")
    else:
        print(f"[clone] {SCORER_REPO} -> {scorer_dir}")
        rc = subprocess.call(["git", "clone", "--depth", "1", SCORER_REPO, str(scorer_dir)])
        if rc != 0:
            sys.exit("git clone failed; clone hotpotqa/hotpot manually into " + str(scorer_dir))
    print("done")


if __name__ == "__main__":
    main()
