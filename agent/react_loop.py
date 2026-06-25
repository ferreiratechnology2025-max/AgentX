"""ReactLoop — core ReAct thought/action/observation loop with LLM integration."""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

from llm.manager import LLMManager, Message, GenerationFailedError, ToolCall
from tools.schemas import TelemetryData, ReActOutput
from tools.registry import ToolRegistry
from .state import AgentState, Step
from .session_manager import save_checkpoint
from .skill_manager import SkillManager
from .tool_executor import ToolExecutor, PermissionRequiredException


class ReactLoop:
    """Loop ReAct: thought -> action -> observation, com auto-continue e skill extraction."""

    def __init__(self, llm_manager: LLMManager, tool_executor: ToolExecutor,
                 skill_manager: SkillManager, tool_registry: ToolRegistry,
                 max_steps: int = 10, temperature: float = 0.7,
                 parallel_tools: bool = True, verbose: bool = True):
        self.llm = llm_manager
        self.tool_executor = tool_executor
        self.skill_manager = skill_manager
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.temperature = temperature
        self.parallel_tools = parallel_tools
        self.verbose = verbose
        self.auto_continue_count: int = 0

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 3.8))

    def _should_auto_continue(self, state: AgentState) -> bool:
        if self.auto_continue_count >= 2:
            return False
        if not state or not state.steps:
            return False
        last = state.steps[-1]
        return len(last.content) > 20

    def _apply_auto_continue(self, state: AgentState) -> str:
        self.auto_continue_count += 1
        msg = (f"[System]: Limite de passos atingido, mas o processo foi auto-estendido "
               f"(extensao {self.auto_continue_count}/2). Continue sua execucao ate concluir o objetivo.")
        state.add_observation(msg)
        return msg

    def _build_prompt(self, state: AgentState) -> str:
        MAX_PROMPT_CHARS = 12000
        prompt = f"\nObjetivo: {state.goal}\n\n"
        if state.context:
            prompt += f"\nContexto adicional:\n{state.context}\n\n"
        prompt += ("\nContinue seu raciocinio. Se tiver informacoes suficientes, "
                   "forneca 'Final Answer:'.\nCaso contrario, decida a proxima acao.\n")
        prompt += ("\nIMPORTANTE: Se a ultima Observation indicou SUCESSO, "
                   "gere Final Answer agora. NAO execute outra tool.\n")
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS] + "\n\n[Contexto truncado por limite de tokens]"
        return prompt

    def _build_messages(self, state: AgentState, system_prompt: str) -> List[Message]:
        prompt = self._build_prompt(state)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt),
        ]
        for step in state.steps[-5:]:
            if step.type == "thought":
                messages.append(Message(role="assistant", content=step.content))
            elif step.type == "action":
                messages.append(Message(role="assistant", content=f"Action: {step.content}"))
            elif step.type == "observation":
                messages.append(Message(role="user", content=f"Observation: {step.content}"))
        return messages

    def _normalize_output(self, output: ReActOutput) -> str:
        thought = output.thought
        if thought.startswith("Thought:"):
            thought = thought[len("Thought:"):].strip()
        if thought.startswith("Action:"):
            thought = "(acao)"
        if output.final_answer:
            return f"Thought: {thought}\nFinal Answer: {output.final_answer}"
        if output.action:
            args_str = json.dumps(output.action.arguments, separators=(",", ":"))
            return f"Thought: {thought}\nAction: {output.action.name}\nAction Input: {args_str}"
        return thought

    def _normalize_response(self, text: str) -> str:
        try:
            data = json.loads(text)
            if "final_answer" in data:
                t = data.get("thought", "")
                return f"Thought: {t}\nFinal Answer: {data['final_answer']}"
            if "action" in data:
                t = data.get("thought", "")
                args = json.dumps(data.get("arguments", {}), separators=(",", ":"))
                return f"Thought: {t}\nAction: {data['action']}\nAction Input: {args}"
        except (json.JSONDecodeError, TypeError):
            pass
        return text

    def _build_trajectory(self, state: AgentState) -> str:
        lines = []
        for step in state.steps:
            tag = {"thought": "", "action": "", "observation": ""}.get(step.type, ".")
            content = step.content[:200].replace("\n", " ")
            lines.append(f"{tag} {content}")
        return "\n".join(lines)

    def _save_checkpoint(self, session_id: str, state: AgentState) -> None:
        if not session_id or not state:
            return
        steps = state.steps
        summary_lines = []
        for s in steps[-6:]:
            tag = {"thought": "", "action": "", "observation": ""}.get(s.type, ".")
            summary_lines.append(f"{tag} {s.content[:200]}")
        save_checkpoint(
            session_id=session_id, goal=state.goal, status=state.status,
            steps_count=len(steps), summary="\n".join(summary_lines),
        )

    async def think(self, state: AgentState, system_prompt: str,
                    tools_spec: List[dict], available_tool_names: Optional[set] = None,
                    final_answer_voter: Optional[callable] = None,
                    temperature: Optional[float] = None) -> Tuple[str, List[ToolCall], TelemetryData]:
        """Gera pensamento, extrai tool_calls do ReActOutput validado."""
        msgs = self._build_messages(state, system_prompt)
        if available_tool_names:
            tools_spec = [s for s in tools_spec if s['name'] in available_tool_names]

        start = time.perf_counter()
        validated, usage = await self.llm.generate_with_validation(
            messages=msgs, tools=tools_spec,
            max_tokens=512, temperature=temperature or self.temperature,
        )
        elapsed = time.perf_counter() - start

        if validated.final_answer and final_answer_voter is not None:
            validated = await final_answer_voter(
                msgs, tools_spec, validated, self.llm, temperature or self.temperature,
            )

        tool_calls = []
        if validated.action:
            tool_calls.append(ToolCall(
                name=validated.action.name,
                arguments=validated.action.arguments,
                id="call_0",
            ))

        normalized = self._normalize_output(validated)
        prompt_tokens = usage.get("prompt_tokens", self._estimate_tokens(
            msgs[0].content + self._build_prompt(state)))
        completion_tokens = usage.get("completion_tokens", self._estimate_tokens(normalized))

        telemetry = TelemetryData(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", prompt_tokens + completion_tokens),
            latency_ms=round(elapsed * 1000, 2),
            throughput_tps=round(completion_tokens / max(0.001, elapsed), 1),
        )
        return normalized, tool_calls, telemetry

    async def generate_final(self, thought: str, temperature: Optional[float] = None) -> Tuple[str, TelemetryData]:
        prompt = f"Baseado no seu raciocinio, forneca a resposta final.\n\nSeu raciocinio:\n{thought}\n\nResposta final:"
        start = time.perf_counter()
        response, usage = await self.llm.generate(
            prompt, max_tokens=512, temperature=temperature or self.temperature)
        elapsed = time.perf_counter() - start
        prompt_tokens = usage.get("prompt_tokens", self._estimate_tokens(prompt))
        completion_tokens = usage.get("completion_tokens", self._estimate_tokens(response))
        telemetry = TelemetryData(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", prompt_tokens + completion_tokens),
            latency_ms=round(elapsed * 1000, 2),
            throughput_tps=round(completion_tokens / max(0.001, elapsed), 1),
        )
        return response, telemetry

    async def extract_skill(self, state: AgentState, llm: LLMManager) -> None:
        """Extrai skill da trajetoria via LLM e salva via SkillManager."""
        if not state or len(state.steps) <= 2:
            return
        trajectory = self._build_trajectory(state)
        prompt = (
            f"Analise a trajetoria de execucao abaixo e extraia apenas a RECONEXAO TECNICA, "
            f"REGEX, ou REGRA DE SINTAXE que garantiu o sucesso da tarefa. "
            f"Seja extremamente minimalista (maximo 2 linhas). Do contrario, ignore.\n\n"
            f"Tarefa: {state.goal}\n"
            f"Trajetoria: {trajectory}\n\n"
            f"Saida esperada:\n"
            f"- Ao executar [contexto], a abordagem correta e [regra/comando]."
        )
        try:
            response, _ = await llm.generate(prompt, max_tokens=128, temperature=0.3)
            skill = response.strip()
            if not skill or len(skill) < 10:
                return
            import uuid
            from datetime import datetime, timezone
            new_skill = {
                'skill_id': str(uuid.uuid4()),
                'extracted_at': datetime.now(timezone.utc).isoformat(),
                'role': state.context or 'general',
                'usage_count': 0, 'success_count': 0, 'failure_count': 0,
                'last_used': datetime.now(timezone.utc).isoformat(),
                'utility_score': 0.5, 'text': skill,
            }
            skills = self.skill_manager.load()
            skills.append(new_skill)
            self.skill_manager.save(skills)
            if self.verbose:
                print(f" [SKILL] Nova skill aprendida: {skill[:80]}...")
        except Exception as e:
            if self.verbose:
                print(f" [SKILL] Erro ao extrair skill: {e}")
