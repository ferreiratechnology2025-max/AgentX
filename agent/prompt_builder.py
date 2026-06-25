"""Dynamic system prompt builder — injects tool schemas automatically"""

import json
from typing import List

from tools.base import Tool


def build_tools_block(tools: List[Tool]) -> str:
    """Monta bloco compacto de definicao das ferramentas para o system prompt"""
    lines = []
    for t in tools:
        args = json.dumps(t.parameters, separators=(",", ":"), ensure_ascii=False)
        lines.append(
            f"- Nome: {t.name}\n"
            f"  Descricao: {t.description}\n"
            f"  Argumentos (JSON Schema): {args}"
        )
    return "\n\n".join(lines)


def build_system_prompt(tools: List[Tool]) -> str:
    """
    Gera o system prompt completo com schemas injetados dinamicamente.
    Usa formato ReAct classico para saida do modelo.
    """
    tools_block = build_tools_block(tools)

    return f"""Voce e o AgentX, um agente de execucao autonomo baseado no loop ReAct (Thought -> Action -> Observation).

Voce opera estritamente atraves das seguintes ferramentas disponiveis:

{tools_block}

## Formato de Resposta OBRIGATORIO

Voce DEVE usar exatamente este formato:

Thought: Seu raciocinio logico aqui antes de agir.
Action: nome_da_ferramenta
Action Input: {{"param1": "valor1", "param2": "valor2"}}

Quando tiver a resposta final:

Thought: Raciocinio consolidando os dados coletados.
Final Answer: Sua resposta estruturada final direcionada ao usuario.

Importante:
- Action Input deve ser sempre um JSON valido
- Nao invente observacoes
- Respeite estritamente os schemas das ferramentas

## Regra de Ouro do Repositorio (Git Automation)
Sempre que voce utilizar a ferramenta 'project_manager' para atualizar o arquivo 'cronograma.json' e detectar que um marco mudou seu status para 'concluido', voce deve OBRIGATORIAMENTE encadear uma proxima acao utilizando a ferramenta 'git_worker'.
O seu proximo passo deve ser dar stage no arquivo alterado e realizar um commit estruturado usando o formato: 'feat(workspace): conclui marco [ID DO MARCO]'. Nao encerre a sessao com final_answer antes de consolidar a alteracao no Git."""
