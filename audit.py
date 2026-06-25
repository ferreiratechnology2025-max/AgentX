"""Automated cronograma audit - run on startup or via scheduled task"""

import asyncio
import json
import sys
from datetime import date

from agent.core import AutonomousAgent, AgentConfig
from tools.registry import ToolRegistry
from tools.builtin import set_memory_instance
from llm.manager import LLMManager
from memory.persistent import MemoriaPersistente
import yaml


AUDIT_PROMPT = (
    'Consulte o arquivo `cronograma.json` usando a ferramenta de projeto. '
    f'Hoje é {date.today().isoformat()}. '
    'Realize uma auditoria de pacing com base nos seguintes princípios de primeira ordem:\n'
    '1. Calcule o desvio de progresso (Progresso Real - Progresso Esperado) para cada marco ativo ou atrasado.\n'
    '2. Se o desvio for negativo, calcule quantos dias de atraso o marco possui em relação à data planejada.\n'
    '3. Emita um alerta estruturado contendo: [GARGALO DETECTADO] se houver desvio negativo, '
    'o impacto no prazo final do projeto e uma sugestão direta para cortar escopo ou acelerar a entrega.'
)


async def run_audit(save_to_memory: bool = True):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    memory = MemoriaPersistente(
        db_path=config["memory"]["db_path"],
        index_path=config["memory"]["index_path"],
        embedding_dim=config["memory"]["embedding_dim"],
        max_memories=config["memory"]["max_memories"],
    )

    set_memory_instance(memory)
    llm = LLMManager(config)
    tool_registry = ToolRegistry()

    agent = AutonomousAgent(
        llm, tool_registry,
        AgentConfig(max_steps=5, temperature=0.3, parallel_tools=False, verbose=False)
    )

    print(f"\n📋 Auditoria automática de cronograma — {date.today().isoformat()}")
    print("=" * 50)

    async for event in agent.run(AUDIT_PROMPT):
        t = event["type"]
        if t == "thought":
            print(f"  🧠 {event['content'][:200]}...")
        elif t == "action":
            print(f"  🛠️  {event['tool']}({json.dumps(event['arguments'])[:100]})")
        elif t == "observation":
            print(f"  👁️  {event['content'][:200]}")
        elif t == "final":
            print(f"\n{'=' * 50}")
            print(f"  🏁 {event['content']}")
            print(f"{'=' * 50}\n")
        elif t == "error":
            print(f"  ❌ {event['content']}")

    if save_to_memory and agent.state and agent.state.steps:
        last = agent.state.steps[-1]
        if last.type == "thought":
            await memory.salvar(
                f"Auditoria de cronograma ({date.today().isoformat()}): {last.content[:500]}",
                importancia=1.2,
            )
            print(f"  💾 Alerta salvo na memória persistente")


def main():
    asyncio.run(run_audit(save_to_memory=True))


if __name__ == "__main__":
    main()
