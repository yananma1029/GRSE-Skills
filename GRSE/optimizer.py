"""Retain upstream operators; allow success evidence to bootstrap skills."""

import re

from maskills.core.skill_operators import SkillEvolutionOptimizer
from maskills.core.skills import parse_skill_md

from .prompts import CONSOLIDATE_SKILLS, INDUCE_SKILL


def _normalize_skill_md(raw):
    """Repair common SKILL.md format slips from open models.

    Seen with Qwen: frontmatter wrapped in an unclosed ```yaml fence and the
    opening ``---`` dropped, e.g. "```yaml\\nname: X\\ndescription: Y\\n---\\nbody".
    """
    text = raw.strip()
    opening = re.match(r"\A(`{3,})[\w-]*[ \t]*\r?\n", text)
    if opening:
        # Remove the outer opener once. An unclosed wrapper is repairable,
        # but its last fence may belong to a code example in the skill body.
        text = text[opening.end():]
        lines = text.splitlines(keepends=True)
        inner_fence = None
        for index, line in enumerate(lines):
            fence = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\r\n]*)\r?\n?", line)
            if not fence:
                continue
            marker, info = fence.groups()
            if inner_fence:
                if (marker[0] == inner_fence[0] and len(marker) >= len(inner_fence)
                        and not info.strip()):
                    inner_fence = None
            elif (index == len(lines) - 1 and marker[0] == "`"
                  and len(marker) >= len(opening.group(1)) and not info.strip()):
                lines.pop()
            else:
                inner_fence = marker
        text = "".join(lines).strip()
    if not text.startswith("---") and re.match(r"\Aname\s*:", text) and "\n---" in text:
        text = "---\n" + text
    if text.startswith("---"):
        # Also seen (~4%): closing "---" dropped, body starts right after
        # "description:". Close the frontmatter after its key: value lines.
        lines = text.splitlines()
        end = 1
        while end < len(lines) and re.match(r"[\w-]+\s*:", lines[end]):
            end += 1
        after = next((line.strip() for line in lines[end:] if line.strip()), "")
        if end > 1 and after != "---":
            lines.insert(end, "---")
            text = "\n".join(lines)
    return text


class GroupSkillOptimizer(SkillEvolutionOptimizer):
    prompt_profile = "default"  # "locomo" is set by GRSE.locomo.install

    def _induce_prompt(self, hard_evidence, existing_library, residual_summary):
        if self.prompt_profile == "locomo":
            from . import locomo
            # Upstream's prompt takes one evidence block; lay it out as upstream does.
            lines = ([f"Residual failure pattern: {residual_summary.strip()}"]
                     if residual_summary.strip() else [])
            lines += [f"{i}. {ev}" for i, ev in enumerate(hard_evidence[:12], 1)]
            existing = "\n".join(f"- {s.name} (id: {s.skill_id}): {s.description}"
                                 for s in existing_library) or "(none)"
            return locomo.INDUCE_SKILL.format(existing=existing, evidence="\n".join(lines))
        return INDUCE_SKILL.format(
            existing="\n".join(s.metadata_line() for s in existing_library) or "(empty)",
            residual=residual_summary or "(none)",
            evidence="\n".join(hard_evidence[:12]),
        )

    def _finish_body(self, skill):
        if self.prompt_profile == "locomo":
            from . import locomo
            skill.body = locomo.dedup(skill.body)
        return skill

    def induce_skill(self, hard_evidence, existing_library, residual_summary=""):
        if not hard_evidence and not residual_summary:
            return None
        prompt = self._induce_prompt(hard_evidence, existing_library, residual_summary)
        # Like upstream: a bad induction skips this round instead of killing the run.
        try:
            raw = self._llm_call(prompt, max_tokens=1200).strip()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("induction LLM call failed: %s", exc)
            return None
        if raw.upper() == "NONE":
            return None
        try:
            skill = parse_skill_md(_normalize_skill_md(raw))
        except ValueError as exc:
            self.logger.warning("induced skill did not parse: %s | raw starts: %r", exc, raw[:200])
            return None
        if not skill.body.strip():
            self.logger.warning("induced skill has no instructions; skipped")
            return None
        skill.provenance = "induced"
        skill.skill_id = ""
        skill.__post_init__()
        return self._finish_body(skill)

    def consolidate_skills(self, library, group_ids):
        """Upstream merge, with the induction output spec and format repair."""
        skills = [s for s in (library.get(sid) for sid in group_ids) if s is not None]
        if len(skills) < 2:
            return None
        blocks = []
        for s in skills:
            body = s.body.strip()
            blocks.append(f"### {s.name} (id: {s.skill_id})\n{s.description}\n\n"
                          f"{body[:800] + (' …' if len(body) > 800 else '')}")
        template = CONSOLIDATE_SKILLS
        if self.prompt_profile == "locomo":
            from . import locomo
            template = locomo.CONSOLIDATE_SKILLS
        prompt = template.format(skills="\n\n".join(blocks))
        try:
            # 1200 like induction: upstream's 900 truncates ~3k-char Qwen skills.
            raw = self._llm_call(prompt, max_tokens=1200).strip()
            macro = parse_skill_md(_normalize_skill_md(raw))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("consolidation failed for %s: %s", group_ids, exc)
            return None
        if not macro.body.strip():
            self.logger.warning("consolidated skill has no instructions; skipped")
            return None
        macro.provenance = "consolidated"
        macro.skill_id = ""
        macro.__post_init__()
        macro.invocations = sum(s.invocations for s in skills)
        macro.utility = sum(s.utility for s in skills) / len(skills)
        return self._finish_body(macro)
