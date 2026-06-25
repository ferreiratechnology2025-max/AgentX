"""Skill Manager — structured skill extraction, utility tracking, relevance selection, forgetting curve, pruning."""
import json
import math
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Literal
from datetime import datetime, timezone

SKILLS_FILE = Path("data/knowledge/skills_learnt.md")


class SkillManager:
    """Gerencia skills com metadados estruturados: extração, utility tracking, seleção por relevância."""

    def __init__(self, skills_file: str = "data/knowledge/skills_learnt.md"):
        self.skills_file = Path(skills_file)

    def load(self) -> list[dict]:
        """Parse skills_learnt.md com metadados em HTML comments."""
        if not self.skills_file.exists():
            return []
        skills = []
        current_meta = {}
        current_text = []
        try:
            with open(self.skills_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    if line.startswith('<!-- ') and line.endswith(' -->'):
                        inner = line[5:-4]
                        if ': ' in inner:
                            key, value = inner.split(': ', 1)
                            current_meta[key] = value
                    elif line == '---':
                        if current_text:
                            skill = dict(current_meta)
                            skill['text'] = '\n'.join(current_text).strip()
                            skill['usage_count'] = int(skill.get('usage_count', 0))
                            skill['success_count'] = int(skill.get('success_count', 0))
                            skill['failure_count'] = int(skill.get('failure_count', 0))
                            skill['utility_score'] = float(skill.get('utility_score', 0.5))
                            skills.append(skill)
                        current_meta = {}
                        current_text = []
                    elif line.strip():
                        current_text.append(line)
        except Exception as e:
            print(f" [SKILL] Erro ao ler skills: {e}")
        if not skills and self.skills_file.exists():
            with open(self.skills_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text and not text.startswith('<!--'):
                for line in text.split('\n'):
                    line = line.strip()
                    if line:
                        skills.append({
                            'skill_id': str(uuid.uuid4()),
                            'text': line,
                            'usage_count': 0, 'success_count': 0, 'failure_count': 0,
                            'utility_score': 0.5, 'role': 'general',
                        })
        return skills

    def save(self, skills: list[dict]):
        """Salva skills no formato estruturado com metadados."""
        self.skills_file.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for s in skills:
            lines.append(f'<!-- skill_id: {s.get("skill_id", str(uuid.uuid4()))} -->')
            lines.append(f'<!-- extracted_at: {s.get("extracted_at", datetime.now(timezone.utc).isoformat())} -->')
            lines.append(f'<!-- role: {s.get("role", "general")} -->')
            lines.append(f'<!-- usage_count: {s.get("usage_count", 0)} -->')
            lines.append(f'<!-- success_count: {s.get("success_count", 0)} -->')
            lines.append(f'<!-- failure_count: {s.get("failure_count", 0)} -->')
            lines.append(f'<!-- last_used: {s.get("last_used", datetime.now(timezone.utc).isoformat())} -->')
            lines.append(f'<!-- utility_score: {s.get("utility_score", 0.5):.4f} -->')
            lines.append(s.get('text', ''))
            lines.append('---')
        try:
            with open(self.skills_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            print(f" [SKILL] Erro ao salvar skills: {e}")

    def update_utility(self, skill_id: str, outcome: Literal["success", "failure", "neutral"]):
        """Atualiza utility_score via Bayesian update."""
        skills = self.load()
        for skill in skills:
            if skill.get('skill_id') == skill_id:
                skill['usage_count'] += 1
                skill['last_used'] = datetime.now(timezone.utc).isoformat()
                if outcome == "success":
                    skill['success_count'] += 1
                elif outcome == "failure":
                    skill['failure_count'] += 1
                alpha = 1 + skill['success_count']
                beta = 1 + skill['failure_count']
                skill['utility_score'] = alpha / (alpha + beta)
                break
        self.save(skills)

    QUARANTINE_THRESHOLD = 3  # usage_count minimo antes de injetar skill

    def select_by_relevance(self, skills: list[dict], role: str, token_budget: int = 400) -> list[dict]:
        """Seleciona skills por utility_score * role_bonus, respeitando token budget.
        Skills em quarentena (usage_count < QUARANTINE_THRESHOLD) nao sao injetadas."""
        if not skills:
            return []
        skills = self._apply_forgetting_curve(skills)
        # Quarentena: skills com poucas uses nao sao injetadas
        skills = [s for s in skills if s.get('usage_count', 0) >= self.QUARANTINE_THRESHOLD]
        for skill in skills:
            role_bonus = 1.2 if skill.get('role') == role else 1.0
            skill['_score'] = skill['utility_score'] * role_bonus
        skills.sort(key=lambda s: s['_score'], reverse=True)
        selected = []
        token_count = 0
        for skill in skills:
            skill_tokens = max(1, int(len(skill.get('text', '')) / 3.8))
            if token_count + skill_tokens > token_budget:
                break
            selected.append(skill)
            token_count += skill_tokens
        return selected

    def _apply_forgetting_curve(self, skills: list[dict]) -> list[dict]:
        """Aplica decadencia exponencial (half-life 30 dias) a skills nao usadas."""
        now = datetime.now(timezone.utc)
        for skill in skills:
            last_str = skill.get('last_used')
            if last_str:
                try:
                    last_used = datetime.fromisoformat(last_str)
                    days = max(0, (now - last_used).total_seconds() / 86400)
                    skill['utility_score'] *= math.exp(-days / 30)
                except (ValueError, TypeError):
                    pass
        return skills

    def prune(self, skills: list[dict]) -> list[dict]:
        """Remove skills com usage_count >= 10 e utility_score < 0.3."""
        kept = []
        for skill in skills:
            if skill.get('usage_count', 0) >= 10 and skill.get('utility_score', 0.5) < 0.3:
                print(f" [SKILL] Podando skill {skill.get('skill_id', '?')} (score={skill.get('utility_score', 0):.2f})")
            else:
                kept.append(skill)
        return kept
