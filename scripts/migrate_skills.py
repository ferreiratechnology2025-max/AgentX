"""Migrate skills_learnt.md from plain text to structured metadata format."""
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

SKILLS_FILE = Path("data/knowledge/skills_learnt.md")
BACKUP_FILE = Path("data/knowledge/skills_learnt.bak")


def migrate():
    if not SKILLS_FILE.exists():
        print("[MIGRATE] skills_learnt.md nao encontrado")
        return

    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Check if already in structured format
    if content.startswith("<!--"):
        print("[MIGRATE] skills_learnt.md ja esta no formato estruturado")
        return

    if not content:
        print("[MIGRATE] skills_learnt.md vazio")
        return

    # Parse plain text: one skill per line (current format)
    lines = content.split("\n")
    skills = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        skills.append({
            "skill_id": str(uuid.uuid4()),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "role": "general",
            "usage_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_used": datetime.now(timezone.utc).isoformat(),
            "utility_score": 0.5,
            "text": line,
        })

    # Backup
    import shutil
    shutil.copy2(SKILLS_FILE, BACKUP_FILE)
    print(f"[MIGRATE] Backup salvo em {BACKUP_FILE}")

    # Write structured format
    lines_out = []
    for s in skills:
        lines_out.append(f'<!-- skill_id: {s["skill_id"]} -->')
        lines_out.append(f'<!-- extracted_at: {s["extracted_at"]} -->')
        lines_out.append(f'<!-- role: {s["role"]} -->')
        lines_out.append(f'<!-- usage_count: {s["usage_count"]} -->')
        lines_out.append(f'<!-- success_count: {s["success_count"]} -->')
        lines_out.append(f'<!-- failure_count: {s["failure_count"]} -->')
        lines_out.append(f'<!-- last_used: {s["last_used"]} -->')
        lines_out.append(f'<!-- utility_score: {s["utility_score"]:.4f} -->')
        lines_out.append(s["text"])
        lines_out.append("---")

    with open(SKILLS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"[MIGRATE] {len(skills)} skills migradas para formato estruturado")


if __name__ == "__main__":
    migrate()
