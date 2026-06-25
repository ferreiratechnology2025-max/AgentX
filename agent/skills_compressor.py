"""Compressor semântico de skills — unifica regras redundantes quando >30 linhas"""
import asyncio
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SKILLS_FILE = Path("data/knowledge/skills_learnt.md")
BACKUP_FILE = Path("data/knowledge/skills_learnt.bak")
MAX_LINES_BEFORE_COMPRESS = 30


def _parse_structured_skills(text: str) -> list[dict]:
    """Parse skills no formato estruturado (HTML comments + text + ---)."""
    skills = []
    current_meta = {}
    current_text = []
    for line in text.split('\n'):
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
                skills.append(skill)
            current_meta = {}
            current_text = []
        elif line.strip():
            current_text.append(line)
    return skills


def _format_structured_skills(skills: list[dict]) -> str:
    """Formata skills no formato estruturado com metadados."""
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
    return '\n'.join(lines) + '\n'


async def compress_skills_if_needed(llm_manager: Optional["LLMManager"] = None, dry_run: bool = False) -> dict:
    """
    Verifica tamanho do skills_learnt.md e comprime via LLM se >30 linhas.
    Preserva metadados das skills durante compressao.

    Args:
        llm_manager: Instância do LLMManager. Se None, tenta carregar da config.
        dry_run: Se True, apenas simula sem alterar arquivos.

    Returns:
        dict com status, linhas_antes, linhas_depois, motivo.
    """
    from llm.manager import LLMManager

    if not SKILLS_FILE.exists():
        return {"status": "skipped", "motivo": "skills_learnt.md não existe"}

    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        conteudo = f.read()

    # Parse structured skills para extrair apenas texto
    old_skills = _parse_structured_skills(conteudo)
    old_text = '\n'.join(s['text'] for s in old_skills if s.get('text'))
    old_line_count = old_text.count('\n') + 1

    if not old_skills:
        # Fallback para formato antigo (texto plano)
        old_text = conteudo.strip()
        old_line_count = len(conteudo.split('\n'))

    if old_line_count <= MAX_LINES_BEFORE_COMPRESS:
        return {"status": "skipped", "motivo": f"{old_line_count} linhas <= {MAX_LINES_BEFORE_COMPRESS}"}

    print(f"[SKILLS COMPRESSOR] {old_line_count} linhas detectadas — comprimindo...")

    # Inicializa LLM se necessário
    if llm_manager is None:
        try:
            import yaml
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            llm_manager = LLMManager(config)
        except Exception as e:
            return {"status": "error", "motivo": f"Falha ao carregar LLM: {e}"}

    prompt = f"""Voce e um compilador de conhecimento tecnico. Abaixo esta uma lista de regras e padroes aprendidos por um agente autonomo.
Sua tarefa e reescrever essa lista eliminando regras duplicadas, unificando conceitos parecidos e resolvendo contradicoes.
Gere uma nova lista limpa, mantendo apenas instrucoes altamente densas, acionaveis e minimalistas.

Regras Atuais:
{old_text}

Saida esperada (Apenas a lista formatada em Markdown, sem introducoes ou observacoes):
- Ao executar [contexto], a abordagem correta e [regra]."""

    try:
        resposta, _ = await llm_manager.generate(prompt, max_tokens=1024, temperature=0.3)
    except Exception as e:
        return {"status": "error", "motivo": f"Falha na geracao LLM: {e}"}

    novo = resposta.strip()
    if not novo:
        return {"status": "error", "motivo": "LLM retornou resposta vazia"}

    # Preservar metadados agregados das skills antigas
    # Agrupar utility_scores das skills unificadas
    total_usage = sum(int(s.get('usage_count', 0)) for s in old_skills)
    total_success = sum(int(s.get('success_count', 0)) for s in old_skills)
    total_failure = sum(int(s.get('failure_count', 0)) for s in old_skills)
    avg_score = sum(float(s.get('utility_score', 0.5)) for s in old_skills) / max(len(old_skills), 1)

    # Criar novas skills estruturadas a partir do output do LLM
    new_skills = []
    for line in novo.split('\n'):
        line = line.strip()
        if line:
            if line.startswith('- '):
                line = line[2:]
            elif line.startswith('* '):
                line = line[2:]
            new_skills.append({
                'skill_id': str(uuid.uuid4()),
                'extracted_at': datetime.now(timezone.utc).isoformat(),
                'role': 'general',
                'usage_count': max(1, total_usage // max(len(new_skills) + 1, 1)),
                'success_count': max(1, total_success // max(len(new_skills) + 1, 1)),
                'failure_count': total_failure // max(len(new_skills) + 1, 1),
                'last_used': datetime.now(timezone.utc).isoformat(),
                'utility_score': min(0.9, avg_score),
                'text': line,
            })

    linhas_novas = len(new_skills)

    if dry_run:
        print(f"[DRY-RUN] {old_line_count} -> ~{linhas_novas} linhas (nao alterado)")
        return {
            "status": "dry_run",
            "linhas_antes": old_line_count,
            "linhas_depois_estimado": linhas_novas,
        }

    # Backup + escrita
    shutil.copy2(SKILLS_FILE, BACKUP_FILE)
    with open(SKILLS_FILE, "w", encoding="utf-8") as f:
        f.write(_format_structured_skills(new_skills))

    print(f"[SKILLS COMPRESSOR] {old_line_count} -> {linhas_novas} skills (backup em {BACKUP_FILE})")
    return {
        "status": "compressed",
        "linhas_antes": old_line_count,
        "linhas_depois": linhas_novas,
    }


if __name__ == "__main__":
    async def _cli():
        result = await compress_skills_if_needed(dry_run=False)
        print(result)
    asyncio.run(_cli())
