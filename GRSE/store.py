"""Resume an existing local run without allocating a second run directory."""

import json
from dataclasses import fields
from pathlib import Path

from maskills.store.local import LocalStore


class GroupRunStore(LocalStore):
    def __init__(self, root_dir, resume_run_id=None):
        super().__init__(root_dir)
        if resume_run_id and ("/" in resume_run_id or "\\" in resume_run_id or resume_run_id in (".", "..")):
            raise ValueError("resume_run_id must be a run id, not a path")
        self.resume_run_id = resume_run_id

    def create_run(self, run_id, config):
        if not self.resume_run_id:
            # Upstream run names have second resolution: avoid accidental reuse.
            from uuid import uuid4
            return super().create_run(f"{run_id}_{uuid4().hex[:8]}", config)
        run_dir = Path(self.get_run_dir(self.resume_run_id))
        path = run_dir / "config.json"
        if not path.is_file() or not (run_dir / "grpo" / "options.json").is_file():
            raise ValueError("Resume target is not a GRPO run")
        previous = json.loads(path.read_text())
        current = {}
        for field in fields(config):
            value = getattr(config, field.name)
            current[field.name] = value.to_dict() if hasattr(value, "to_dict") else value
        # JSON canonicalization accounts for tuple/list conversion.
        if json.loads(json.dumps(current)) != previous:
            raise ValueError("Resume requires the original MASkills configuration")
        return self.resume_run_id
