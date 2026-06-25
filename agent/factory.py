"""Agent Factory - Criacao dinamica de agentes especializados"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass

from llm.pool import get_llm_pool
from agent.core import AutonomousAgent, AgentConfig
from agent.ecc_loader import load_ecc_rules_for_role
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

ECC_AGENTS_DIR = Path("data/knowledge/ecc_agents")


@dataclass
class AgentRoleConfig:
    """Configuracao de um agente especializado"""
    role: str
    model_id: Optional[str] = None
    tools_subset: Optional[List[str]] = None
    system_prompt: str = ""
    max_steps: int = 10
    temperature: float = 0.7


class AgentFactory:
    """
    Factory para criar agentes especializados.

    Uso:
        factory = AgentFactory()
        coder = await factory.create_agent(role="coding", system_prompt="...")
    """

    DEFAULT_PROMPTS = {
        "general": "Voce e um assistente geral.",
        "coding": "Voce e um programador expert.",
        "research": "Voce e um pesquisador.",
        "judging": "Voce e um juiz avaliador.",
        "code_review": "Voce e um revisor de codigo senior.",
        "security": "Voce e um especialista em seguranca.",
        "tdd": "Voce e um especialista em TDD.",
        "planning": "Voce e um especialista em planejamento.",
        "architect": "Voce e um arquiteto de software senior.",
    }

    ROLE_TOOLS = {
        "coding": ["calculator", "project_manager", "process_orchestrator"],
        "research": ["save_memory"],
        "judging": ["save_memory"],
        "code_review": ["calculator", "project_manager"],
        "security": ["calculator", "project_manager", "process_orchestrator"],
        "tdd": ["calculator", "project_manager", "process_orchestrator"],
        "planning": ["save_memory", "project_manager"],
        "architect": ["save_memory", "project_manager"],
    }

    def __init__(self):
        self.llm_pool = get_llm_pool()
        self.created_agents: Dict[str, AutonomousAgent] = {}
        self._lock = asyncio.Lock()
        self._tool_registry_cache: Optional[ToolRegistry] = None

    def _get_tool_registry(self) -> ToolRegistry:
        """Obtem ou cria ToolRegistry (cache singleton)"""
        if self._tool_registry_cache is None:
            self._tool_registry_cache = ToolRegistry()
        return self._tool_registry_cache

    async def create_agent(
        self,
        role: str,
        agent_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools_subset: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        max_steps: int = 10,
        temperature: float = 0.7,
        yolo_mode: bool = False
    ) -> AutonomousAgent:
        """Cria um novo agente especializado"""
        async with self._lock:
            if agent_id is None:
                agent_id = f"{role}_{len(self.created_agents) + 1}"

            # Carrega prompt do ECC se for role especializada
            if system_prompt is None:
                system_prompt = self._load_ecc_prompt(role)

            if tools_subset is None:
                tools_subset = self.ROLE_TOOLS.get(role)

            llm = await self.llm_pool.get_model(model_id=model_id, role=role)

            tool_registry = self._get_tool_registry()

            agent_config = AgentConfig(
                max_steps=max_steps,
                temperature=temperature,
                parallel_tools=True,
                verbose=True,
                yolo_mode=yolo_mode
            )

            agent = AutonomousAgent(
                llm_manager=llm,
                tool_registry=tool_registry,
                config=agent_config
            )

            if tools_subset is not None:
                agent.available_tool_names = set(tools_subset)

            # Injeta regras ECC no system prompt
            final_prompt = system_prompt or ""
            ecc_rules = load_ecc_rules_for_role(role)
            if ecc_rules:
                final_prompt += f"\n\n## REGRAS ECC\n{ecc_rules}"
            if final_prompt:
                agent.custom_system_prompt = final_prompt

            if role == "security":
                agent.final_answer_validator = self._security_checklist_validator
                agent.final_answer_voter = self._security_checklist_voter

            self.created_agents[agent_id] = agent

            logger.info(f"Agente criado: {agent_id} | Role: {role} | Tools: {tools_subset}")

            return agent

    @staticmethod
    def _parse_checklist_verdicts(text: str) -> Dict[int, str]:
        """Extrai veredicto (SIM/NAO) por item 1-8 do texto do checklist.

        Aceita notação individual ("1. SIM") e por faixa ("2-8. NÃO").
        Retorna dict {item: 'SIM'|'NAO'} para os itens encontrados.
        """
        verdicts: Dict[int, str] = {}

        # Primeiro: faixas como "2-8. NÃO" ou "1-8. NÃO em todas"
        for m in re.finditer(r"\b([1-8])\s*[-]\s*([1-8])\.?\s*(SIM|N[AÃ]O)\b",
                              text, re.IGNORECASE):
            start_i, end_i = int(m.group(1)), int(m.group(2))
            v = "SIM" if m.group(3).upper() == "SIM" else "NAO"
            for i in range(start_i, end_i + 1):
                verdicts[i] = v

        # Depois: itens individuais "N." com janela de contexto até o próximo item
        boundaries = [(m.start(), int(m.group(1)))
                      for m in re.finditer(r"\b([1-8])\.", text)]
        for idx, (pos, item) in enumerate(boundaries):
            end_pos = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else pos + 200
            chunk = text[pos:end_pos]
            if re.search(r"\bSIM\b", chunk, re.IGNORECASE):
                verdicts[item] = "SIM"
            elif re.search(r"\bN[AÃ]O\b", chunk, re.IGNORECASE):
                verdicts[item] = "NAO"

        return verdicts

    @staticmethod
    def _build_voted_checklist(
        majority: Dict[int, str],
        sim_details: Dict[int, str],
    ) -> str:
        """Constrói o final_answer votado a partir dos veredictos majoritários.

        majority: {item: 'SIM'|'NAO'} para todos os 8 itens
        sim_details: {item: detalhe textual} para itens com SIM
        """
        parts: list = []
        i = 1
        while i <= 8:
            if majority.get(i) == "SIM":
                detail = sim_details.get(i, "SIM")
                parts.append(f"{i}. {detail}")
                i += 1
            else:
                # Agrupa NÃOs consecutivos em faixa
                j = i
                while j <= 8 and majority.get(j, "NAO") != "SIM":
                    j += 1
                parts.append(f"{i}-{j-1}. NAO." if j - i > 1 else f"{i}. NAO.")
                i = j
        return " ".join(parts)

    @classmethod
    async def _security_checklist_voter(
        cls,
        messages: list,
        tools_spec: list,
        first_output,   # ReActOutput
        llm,            # LLMManager
        temperature: float,
    ):
        """Voter 3x para o checklist de security.

        Gera 2 respostas adicionais com as mesmas messages (observation já incluída),
        vota por maioria por item (2/3), retorna ReActOutput com final_answer votado.
        """
        from tools.schemas import ReActOutput as _ReActOutput
        from llm.manager import GenerationFailedError

        outputs = [first_output]
        for _ in range(2):
            try:
                extra, _ = await llm.generate_with_validation(
                    messages=messages,
                    tools=tools_spec,
                    max_tokens=512,
                    temperature=temperature,
                )
                outputs.append(extra)
            except GenerationFailedError:
                pass  # usa o que tem

        if len(outputs) == 1:
            return first_output  # fallback: sem votos extras

        # Vota por item
        from collections import Counter
        item_votes: Dict[int, list] = {i: [] for i in range(1, 9)}
        sim_details: Dict[int, str] = {}

        for out in outputs:
            if not out.final_answer:
                continue
            parsed = cls._parse_checklist_verdicts(out.final_answer)
            for item, verdict in parsed.items():
                item_votes[item].append(verdict)
                if verdict == "SIM" and item not in sim_details:
                    # Extrai o detalhe do item SIM da resposta original
                    m = re.search(
                        rf"\b{item}\.\s*(.*?)(?=\s+\d+\.|\s+\d+-\d+\.|$)",
                        out.final_answer,
                        re.IGNORECASE | re.DOTALL,
                    )
                    sim_details[item] = m.group(1).strip() if m else "SIM"

        majority: Dict[int, str] = {}
        for item in range(1, 9):
            votes = item_votes[item]
            if not votes:
                majority[item] = "NAO"
                continue
            counts = Counter(votes)
            top_verdict, top_count = counts.most_common(1)[0]
            majority[item] = top_verdict

        voted_text = cls._build_voted_checklist(majority, sim_details)
        return first_output.copy(update={"final_answer": voted_text})

    @staticmethod
    def _security_checklist_validator(final_answer: str) -> Optional[str]:
        """Verifica que todos os 8 itens do checklist aparecem no final_answer.

        Aceita notação individual ("1.", "2.") e por faixa ("2-8.").
        Retorna mensagem de correção se algum item estiver ausente, None se válido.
        """
        covered: set = set()
        for m in re.finditer(r"\b([1-8])\.", final_answer):
            covered.add(int(m.group(1)))
        for m in re.finditer(r"\b([1-8])-([1-8])\b", final_answer):
            start, end = int(m.group(1)), int(m.group(2))
            covered.update(range(start, end + 1))
        missing = sorted(set(range(1, 9)) - covered)
        if not missing:
            return None
        missing_str = ", ".join(str(n) for n in missing)
        return (
            f"Checklist incompleto: itens {missing_str} não respondidos. "
            f"Para cada item ausente declare: N. NÃO. "
            f"Complete o checklist com todos os 8 itens numerados."
        )

    def _load_ecc_prompt(self, role: str) -> str:
        """Tenta carregar prompt de agente do ECC; fallback para DEFAULT_PROMPTS."""
        agent_file = ECC_AGENTS_DIR / f"{role}.md"
        if agent_file.exists():
            content = agent_file.read_text(encoding="utf-8").strip()
            if content:
                return content
        return self.DEFAULT_PROMPTS.get(role, self.DEFAULT_PROMPTS["general"])

    def get_agent(self, agent_id: str) -> Optional[AutonomousAgent]:
        """Obtem agente pelo ID"""
        return self.created_agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """Lista todos os agentes"""
        return [
            {
                "agent_id": agent_id,
                "role": getattr(agent, 'custom_system_prompt', '')[:50]
            }
            for agent_id, agent in self.created_agents.items()
        ]

    async def cleanup(self):
        """Libera todos os agentes e modelos"""
        async with self._lock:
            self.created_agents.clear()
            await self.llm_pool.unload_all()


_factory_instance: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Obtem instancia singleton da Factory"""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = AgentFactory()
    return _factory_instance
