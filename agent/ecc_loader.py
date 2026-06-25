from pathlib import Path
from typing import Dict, List

ECC_RULES_DIR = Path("data/knowledge/ecc_rules")
MAX_RULES_TOKENS = 2000

RULES_MAP: Dict[str, List[str]] = {
    "coding": ["coding-style.md", "python.md", "git-workflow.md"],
    "security": ["security.md", "python.md"],
    "testing": ["testing.md"],
    "code_review": ["coding-style.md", "security.md"],
    "general": ["coding-style.md"],
    "planning": ["coding-style.md"],
    "architect": ["coding-style.md"],
}


def load_ecc_rules_for_role(role: str) -> str:
    selected = RULES_MAP.get(role, RULES_MAP["general"])
    parts: List[str] = []
    total_chars = 0
    limit = MAX_RULES_TOKENS * 4

    for filename in selected:
        filepath = ECC_RULES_DIR / filename
        if filepath.exists():
            label = filename.replace(".md", "").upper()
            content = filepath.read_text(encoding="utf-8").strip()
            section = f"## {label}\n{content}"
            if total_chars + len(section) > limit:
                remaining = limit - total_chars
                if remaining > 80:
                    parts.append(section[:remaining] + "\n...[truncated]")
                break
            parts.append(section)
            total_chars += len(section)

    return "\n\n".join(parts)
