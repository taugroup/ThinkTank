"""Skill registry.

OWNER: Person 1 (orchestration). Discovers the available Agent Skills by reading the
frontmatter of each `skills/*/SKILL.md`, and exposes a catalog the Policy Director
uses to DECIDE which skills each agent needs for a given task. Skills are therefore
orchestrator-assigned at runtime, not hardcoded on the agents.
"""

from __future__ import annotations

import os

from config import SKILLS_DIR


def _parse_frontmatter(path: str) -> dict[str, str]:
    """Parse the simple `key: value` YAML frontmatter block at the top of a file."""
    meta: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---"):
        return meta
    end = text.find("---", 3)
    if end == -1:
        return meta
    for line in text[3:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


def list_skills(skills_dir: str = SKILLS_DIR) -> dict[str, str]:
    """Return {skill_key: description} for every skill that has a SKILL.md."""
    catalog: dict[str, str] = {}
    if not os.path.isdir(skills_dir):
        return catalog
    for name in sorted(os.listdir(skills_dir)):
        md = os.path.join(skills_dir, name, "SKILL.md")
        if os.path.isfile(md):
            meta = _parse_frontmatter(md)
            catalog[meta.get("name", name)] = meta.get("description", "")
    return catalog


def skill_catalog_text(skills_dir: str = SKILLS_DIR) -> str:
    """Render the catalog as a compact list for the Director's prompt."""
    return "\n".join(f"- {k}: {v}" for k, v in list_skills(skills_dir).items())


def valid_skills(keys: list[str], skills_dir: str = SKILLS_DIR) -> list[str]:
    """Filter a proposed skill list down to ones that actually exist."""
    available = set(list_skills(skills_dir))
    return [k for k in keys if k in available]
