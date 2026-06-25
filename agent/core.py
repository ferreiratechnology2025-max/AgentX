"""Agent Core — orchestrates ReactLoop, ToolExecutor, SkillManager for SSE event streaming."""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass

from llm.manager import LLMManager, Message, GenerationFailedError
from tools.registry import ToolRegistry
from tools.base import ToolPermission, ToolResult
from tools.schemas import TelemetryData
from .state import AgentState, Step
from .prompt_builder import build_system_prompt
from .react_loop import ReactLoop
from .tool_executor import ToolExecutor, PermissionRequiredException
from .skill_manager import SkillManager


@dataclass
class AgentConfig:
    max_steps: int = 10
    temperature: float = 0.7
    parallel_tools: bool = True
    verbose: bool = True
    yolo_mode: bool = False


class AutonomousAgent:
    """Orquestrador: delega ReAct loop, tool execution, skill management para modulos especializados."""

    def __init__(self, llm_manager: LLMManager, tool_registry: ToolRegistry,
                 config: Optional[AgentConfig] = None, tools_subset: Optional[List[str]] = None):
        self.llm = llm_manager
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()
        self.state: Optional[AgentState] = None
        self.custom_system_prompt: str = ""
        self.available_tool_names: Optional[set] = set(tools_subset) if tools_subset else None

        self.tool_executor = ToolExecutor(tool_registry, self.config.yolo_mode)
        self.skill_manager = SkillManager()
        self.react_loop = ReactLoop(
            llm_manager, self.tool_executor, self.skill_manager, tool_registry,
            max_steps=config.max_steps if config else 10,
            temperature=config.temperature if config else 0.7,
            parallel_tools=config.parallel_tools if config else True,
            verbose=config.verbose if config else True,
        )

        self.final_answer_validator: Optional[callable] = None
        self.final_answer_voter: Optional[callable] = None
        self._last_injected_skill_ids: List[str] = []

        tool_count = len(tools_subset) if tools_subset else len(tool_registry)
        print(f" Agente inicializado com {tool_count} ferramentas")
        if self.config.verbose:
            tools = tools_subset if tools_subset else [t.name for t in tool_registry.list_all()]
            for name in tools:
                print(f"   - {name}")

    async def _get_system_prompt(self) -> str:
        all_tools = self.tool_registry.list_all()
        if self.available_tool_names:
            tools = [t for t in all_tools if t.name in self.available_tool_names]
        else:
            tools = all_tools
        base = build_system_prompt(tools)
        if self.custom_system_prompt:
            base = self.custom_system_prompt + "\n\n" + base
        try:
            skills = self.skill_manager.load()
            skills = self.skill_manager.prune(skills)
            role = getattr(self.state, 'role', 'general') if self.state else 'general'
            selected = self.skill_manager.select_by_relevance(skills, role)
            self._last_injected_skill_ids = [s.get('skill_id', '') for s in selected if s.get('skill_id')]
            if selected:
                lines = [s['text'] for s in selected if s.get('text')]
                base += "\n\n## REGRAS APRENDIDAS EM EXECUCOES ANTERIORES\n" + "\n".join(lines)
                if self.config.verbose:
                    print(f" [SKILL] {len(selected)} skills injetadas para role={role}")
        except Exception as e:
            print(f" [SKILL] Erro ao selecionar skills: {e}")
        return base

    def _check_final_answer(self, text: str) -> Optional[str]:
        if self.final_answer_validator is None:
            return None
        correction = self.final_answer_validator(text)
        self.final_answer_validator = None
        return correction

    async def run(self, goal: str, context: Optional[str] = None,
                  session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        self.state = AgentState(goal=goal, context=context)
        self.tool_executor.reset_usage_counts()
        yield {"type": "status", "content": f"Objetivo: {goal}"}

        while True:
            for step_num in range(self.config.max_steps):
                self.state.current_step = step_num

                # 1. THINK
                try:
                    tools_spec = self.tool_registry.to_llm_spec()
                    system = await self._get_system_prompt()
                    thought, tool_calls, telemetry = await self.react_loop.think(
                        self.state, system, tools_spec, self.available_tool_names,
                        self.final_answer_voter, self.config.temperature,
                    )
                except GenerationFailedError as exc:
                    self.state.status = "failed"
                    self.react_loop._save_checkpoint(session_id, self.state)
                    yield {"type": "error", "content": f"Falha na geracao (sub-tarefa abortada): {exc}"}
                    return
                self.state.add_step(Step(type="thought", content=thought))
                yield {"type": "thought", "content": thought, "step": step_num, "telemetry": telemetry.model_dump()}

                if "Final Answer:" in thought:
                    await self.react_loop.extract_skill(self.state, self.llm)
                    final_answer = thought.split("Final Answer:")[-1].strip()
                    correction = self._check_final_answer(final_answer)
                    if correction:
                        self.state.add_observation(correction)
                        self.state.add_step(Step(type="observation", content=correction))
                        yield {"type": "observation", "content": correction, "step": step_num}
                        continue
                    self.state.status = "completed"
                    self.react_loop._save_checkpoint(session_id, self.state)
                    yield {"type": "final", "content": final_answer, "telemetry": telemetry.model_dump()}
                    return

                if not tool_calls:
                    tool_calls = self.llm.parse_tool_calls(thought)

                if not tool_calls:
                    await self.react_loop.extract_skill(self.state, self.llm)
                    final, telemetry = await self.react_loop.generate_final(thought, self.config.temperature)
                    self.state.status = "completed"
                    self.react_loop._save_checkpoint(session_id, self.state)
                    yield {"type": "final", "content": final, "telemetry": telemetry.model_dump()}
                    return

                # 2. EXECUTE TOOLS
                try:
                    if self.config.parallel_tools and len(tool_calls) > 1:
                        observations = await asyncio.gather(*[
                            self.tool_executor.execute(tc.name, tc.arguments, tc.id) for tc in tool_calls
                        ])
                    else:
                        observations = []
                        for tc in tool_calls:
                            obs = await self.tool_executor.execute(tc.name, tc.arguments, tc.id)
                            observations.append(obs)
                except PermissionRequiredException:
                    self.state.status = "awaiting_approval"
                    self.state.pending_action = self.tool_executor.permission_state
                    yield {"type": "awaiting_approval", "pending": self.state.pending_action}
                    return

                for tc, obs in zip(tool_calls, observations):
                    self.state.add_step(Step(type="action", content=f"{tc.name}({tc.arguments})"))
                    yield {"type": "action", "tool": tc.name, "arguments": tc.arguments, "step": step_num}
                    obs_content = obs.output[:500]
                    self.state.add_observation(obs_content)
                    self.state.add_step(Step(type="observation", content=obs_content))
                    yield {"type": "observation", "content": obs_content, "step": step_num}

                await asyncio.sleep(0.05)

            if self.react_loop._should_auto_continue(self.state):
                msg = self.react_loop._apply_auto_continue(self.state)
                yield {"type": "observation", "content": msg[:500]}
                continue

            self.state.status = "failed"
            yield {"type": "error",
                   "content": f"Limite de {self.config.max_steps} steps apos {self.react_loop.auto_continue_count} auto-extensoes"}
            return

    async def run_with_judge(self, goal: str, context: Optional[str] = None,
                              session_id: Optional[str] = None,
                              max_iterations: int = 2) -> AsyncGenerator[Dict[str, Any], None]:
        from agent.judge import get_judge, Verdict
        judge = get_judge()
        iteration = 0
        feedback_history = []
        while iteration < max_iterations:
            iteration += 1
            if feedback_history:
                fb_text = "\n".join(f"Feedback do Judge (iter {i+1}): {fb}" for i, fb in enumerate(feedback_history))
                enhanced_goal = f"{goal}\n\nFeedback de iteracoes anteriores:\n{fb_text}\n\nPor favor, melhore sua resposta com base no feedback acima."
            else:
                enhanced_goal = goal
            final_output = None
            async for event in self.run(enhanced_goal, context, session_id):
                yield event
                if event.get("type") == "final":
                    final_output = event.get("content")
            if not final_output:
                yield {"type": "error", "content": "Worker nao produziu output final"}
                return
            evaluation = await judge.evaluate(task=goal, worker_output=final_output, role="general")
            yield {"type": "judge_evaluation", "iteration": iteration, "score": evaluation.score,
                   "verdict": evaluation.verdict.value, "reasoning": evaluation.reasoning,
                   "feedback": evaluation.feedback}
            if evaluation.verdict == Verdict.APPROVED:
                return
            elif evaluation.verdict == Verdict.NEEDS_REVISION:
                feedback_history.append(evaluation.feedback)
            else:
                if iteration < max_iterations:
                    feedback_history.append(evaluation.feedback)
                else:
                    yield {"type": "error", "content": "Maximo de iteracoes REJECTED"}
                    return
        yield {"type": "error", "content": f"Maximo de iteracoes ({max_iterations}) atingido"}

    async def resume_loop(self, rejected: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        if self.state.status != "awaiting_approval":
            yield {"type": "error", "content": "Agente nao esta aguardando aprovacao"}
            return
        if rejected:
            msg = f"Usuario rejeitou a execucao da ferramenta {self.state.pending_action['name']}. Mude sua estrategia."
            self.state.add_observation(msg)
            self.state.add_step(Step(type="observation", content=msg[:500]))
            yield {"type": "observation", "content": msg[:500]}
            self.state.pending_action = None
            self.state.status = "running"
            async for event in self._continue_loop():
                yield event
            return
        if self.state.pending_action:
            tool_name = self.state.pending_action["name"]
            tool = self.tool_registry.get(tool_name)
            if tool:
                yield {"type": "status", "content": f" Executando {tool_name} (aprovado pelo usuario)"}
                result = await tool.execute(**self.state.pending_action["arguments"])
                obs_content = result.output[:500]
                self.state.add_observation(obs_content)
                self.state.add_step(Step(type="observation", content=obs_content))
                yield {"type": "observation", "content": obs_content}
                self.state.pending_action = None
                self.state.status = "running"
        async for event in self._continue_loop():
            yield event

    async def _continue_loop(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            for step_num in range(self.state.current_step, self.config.max_steps):
                self.state.current_step = step_num
                try:
                    tools_spec = self.tool_registry.to_llm_spec()
                    system = await self._get_system_prompt()
                    thought, tool_calls, telemetry = await self.react_loop.think(
                        self.state, system, tools_spec, self.available_tool_names,
                        self.final_answer_voter, self.config.temperature,
                    )
                except GenerationFailedError as exc:
                    self.state.status = "failed"
                    yield {"type": "error", "content": f"Falha na geracao (sub-tarefa abortada): {exc}"}
                    return
                self.state.add_step(Step(type="thought", content=thought))
                yield {"type": "thought", "content": thought, "step": step_num, "telemetry": telemetry.model_dump()}
                if "Final Answer:" in thought:
                    await self.react_loop.extract_skill(self.state, self.llm)
                    final_answer = thought.split("Final Answer:")[-1].strip()
                    correction = self._check_final_answer(final_answer)
                    if correction:
                        self.state.add_observation(correction)
                        self.state.add_step(Step(type="observation", content=correction))
                        yield {"type": "observation", "content": correction, "step": step_num}
                        continue
                    self.state.status = "completed"
                    yield {"type": "final", "content": final_answer, "telemetry": telemetry.model_dump()}
                    return
                if not tool_calls:
                    tool_calls = self.llm.parse_tool_calls(thought)
                if not tool_calls:
                    await self.react_loop.extract_skill(self.state, self.llm)
                    final, telemetry = await self.react_loop.generate_final(thought, self.config.temperature)
                    self.state.status = "completed"
                    yield {"type": "final", "content": final, "telemetry": telemetry.model_dump()}
                    return
                try:
                    if self.config.parallel_tools and len(tool_calls) > 1:
                        observations = await asyncio.gather(*[
                            self.tool_executor.execute(tc.name, tc.arguments, tc.id) for tc in tool_calls
                        ])
                    else:
                        observations = []
                        for tc in tool_calls:
                            obs = await self.tool_executor.execute(tc.name, tc.arguments, tc.id)
                            observations.append(obs)
                except PermissionRequiredException:
                    yield {"type": "awaiting_approval", "pending": self.state.pending_action}
                    return
                for tc, obs in zip(tool_calls, observations):
                    self.state.add_step(Step(type="action", content=f"{tc.name}({tc.arguments})"))
                    yield {"type": "action", "tool": tc.name, "arguments": tc.arguments, "step": step_num}
                    obs_content = obs.output[:500]
                    self.state.add_observation(obs_content)
                    self.state.add_step(Step(type="observation", content=obs_content))
                    yield {"type": "observation", "content": obs_content, "step": step_num}
                await asyncio.sleep(0.05)
            if self.react_loop._should_auto_continue(self.state):
                msg = self.react_loop._apply_auto_continue(self.state)
                yield {"type": "observation", "content": msg[:500]}
                self.state.current_step = 0
                continue
            self.state.status = "failed"
            yield {"type": "error",
                   "content": f"Limite de {self.config.max_steps} steps apos {self.react_loop.auto_continue_count} auto-extensoes"}
            return

    def get_state(self) -> Dict[str, Any]:
        return self.state.to_dict() if self.state else {}
