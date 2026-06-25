"""Orchestrator - Coordena multiplos workers para tarefas complexas"""

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any, AsyncGenerator

from llm.pool import get_llm_pool
from agent.factory import get_agent_factory
from agent.judge import get_judge, Verdict

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class SubTask:
    id: str
    description: str
    role: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    judge_score: Optional[int] = None
    judge_verdict: Optional[Verdict] = None
    iterations: int = 0


@dataclass
class TaskPlan:
    goal: str
    subtasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    final_result: Optional[str] = None


class Orchestrator:
    """
    Orchestrator coordena workers para tarefas complexas.

    Fluxo:
    1. Decompoe goal em sub-tasks (workhorse)
    2. Executa sub-tasks sequencialmente
    3. Avalia outputs com Judge (default=workhorse; escala pro reasoner se threshold atingido)
    4. Sintetiza resultado final
    """

    MAX_SUBTASKS = 5
    MAX_ITERATIONS_PER_SUBTASK = 2

    def __init__(self):
        self.llm_pool = get_llm_pool()
        self.factory = get_agent_factory()
        self.judge = get_judge()
        self._load_judge_config()

    def _load_judge_config(self) -> None:
        """Carrega thresholds de escalada do config.yaml."""
        import yaml
        try:
            with open("config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            cfg = {}
        judge_cfg = cfg.get("judge", {})
        self.judge_default_model: str = judge_cfg.get("default_model", "gemma3ne4b")
        self.judge_escalation_model: str = judge_cfg.get("escalation_model", "qwen35-9b")
        self.escalation_needs_revision: int = judge_cfg.get("escalation_needs_revision", 2)
        self.escalation_score_threshold: int = judge_cfg.get("escalation_score_threshold", 4)
        self.hitl_retry_attempts: int = judge_cfg.get("hitl_retry_attempts", 2)
        pool_cfg = cfg.get("llm_pool", {})
        self.reasoner_timeout: int = pool_cfg.get("reasoner_timeout", 30)
        logger.info(
            f"Judge config: default={self.judge_default_model}, "
            f"escalation={self.judge_escalation_model} "
            f"(needs_revision>={self.escalation_needs_revision} OR score<={self.escalation_score_threshold})"
        )

    async def run(self, goal: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Executa goal complexo decompondo em sub-tasks sequenciais."""
        logger.info(f"Orchestrator: {goal[:80]}...")

        yield {"type": "orchestrator_start", "goal": goal}

        # 1. Decompor
        yield {"type": "decomposition_start"}
        plan = await self._decompose_goal(goal)
        yield {
            "type": "decomposition_complete",
            "subtasks": [
                {"id": st.id, "description": st.description,
                 "role": st.role, "dependencies": st.dependencies}
                for st in plan.subtasks
            ]
        }

        # 2. Executar sub-tasks sequencialmente
        for subtask in plan.subtasks:
            # Verificar dependencias
            deps_ok = all(
                next((s for s in plan.subtasks if s.id == d), None) is not None
                and next((s for s in plan.subtasks if s.id == d), None).status == TaskStatus.COMPLETED
                for d in subtask.dependencies
            ) if subtask.dependencies else True

            if not deps_ok:
                subtask.status = TaskStatus.FAILED
                yield {"type": "subtask_error", "subtask_id": subtask.id,
                       "error": "Dependencias nao satisfeitas"}
                continue

            yield {"type": "subtask_start", "subtask_id": subtask.id,
                   "description": subtask.description, "role": subtask.role}

            subtask.status = TaskStatus.RUNNING
            worker = await self.factory.create_agent(
                role=subtask.role,
                agent_id=f"worker_{subtask.id}",
                yolo_mode=True
            )

            iteration = 0
            feedback_history = []
            needs_revision_count = 0  # rastreia NEEDS_REVISION consecutivos para escalada

            parse_retries = 0
            while iteration < self.MAX_ITERATIONS_PER_SUBTASK:
                iteration += 1
                subtask.iterations = iteration

                enhanced = subtask.description
                if feedback_history:
                    fb_text = "\n".join(
                        f"Feedback (iter {i+1}): {fb}"
                        for i, fb in enumerate(feedback_history)
                    )
                    enhanced = (
                        f"{subtask.description}\n\n"
                        f"## Feedback do Judge\n{fb_text}\n\n"
                        f"## IMPORTANTE\n"
                        f"- Considere o feedback para melhorar a QUALIDADE do codigo\n"
                        f"- NAO mude requisitos explicitos do goal original\n"
                        f"- Se o goal pediu 'recursivo', mantenha recursivo\n"
                        f"- Se o goal pediu '3 testes', mantenha 3 testes\n"
                        f"- Melhore qualidade (type hints, docstrings, legibilidade)\n"
                        f"- NAO mude a abordagem fundamental\n\n"
                        f"## Goal Original\n{subtask.description}"
                    )

                hitl_retries = 0
                worker_output = None
                pending_action = None
                async for event in worker.run(enhanced):
                    yield {**event, "subtask_id": subtask.id}
                    if event.get("type") == "final":
                        worker_output = event.get("content")
                    elif event.get("type") == "awaiting_approval":
                        pending_action = event.get("pending")
                        logger.warning(f"Auto-rejeitando: {pending_action}")

                # HITL reformulation loop: ate hitl_retry_attempts retries apos rejeicao
                while (not worker_output and pending_action
                       and hitl_retries < self.hitl_retry_attempts):
                    hitl_retries += 1
                    logger.warning(
                        f"HITL reformulation attempt {hitl_retries}/{self.hitl_retry_attempts}"
                    )
                    pending_action = None
                    async for event in worker.resume_loop(rejected=True):
                        yield {**event, "subtask_id": subtask.id}
                        if event.get("type") == "final":
                            worker_output = event.get("content")
                        elif event.get("type") == "awaiting_approval":
                            pending_action = event.get("pending")
                            logger.warning(f"Auto-rejeitando novamente: {pending_action}")

                if not worker_output:
                    subtask.status = TaskStatus.FAILED
                    break

                # Capturar skill IDs injetados durante execucao para feedback loop
                injected_skill_ids = list(getattr(worker, '_last_injected_skill_ids', []))

                # needs_revision_count alto -> escalada direta (qualidade não melhorou)
                # score borderline -> vota 3x com default; empate 1-1-1 -> reasoner
                direct_escalate = needs_revision_count >= self.escalation_needs_revision
                judge_model = self.judge_escalation_model if direct_escalate else self.judge_default_model

                yield {
                    "type": "judge_evaluation",
                    "subtask_id": subtask.id,
                    "iteration": iteration,
                    "judge_model": judge_model,
                    "escalated": direct_escalate,
                }

                judge_timeout = self.reasoner_timeout if direct_escalate else None
                evaluation = await self.judge.evaluate(
                    task=subtask.description,
                    worker_output=worker_output,
                    role=subtask.role,
                    model_id=judge_model,
                    timeout=judge_timeout,
                )

                # Votação para scores borderline (apenas quando não escalou por revision count)
                voted = False
                escalated_via_tie = False
                if (not direct_escalate
                        and evaluation.score <= self.escalation_score_threshold
                        and evaluation.parse_success):
                    evaluation, escalated_via_tie = await self._vote_judge_verdict(
                        task=subtask.description,
                        worker_output=worker_output,
                        role=subtask.role,
                        first_evaluation=evaluation,
                        subtask_id=subtask.id,
                    )
                    voted = True

                yield {
                    "type": "judge_result",
                    "subtask_id": subtask.id,
                    "iteration": iteration,
                    "score": evaluation.score,
                    "verdict": evaluation.verdict.value,
                    "reasoning": evaluation.reasoning,
                    "feedback": evaluation.feedback,
                    "parse_success": evaluation.parse_success,
                    "voted": voted,
                    "escalated_via_tie": escalated_via_tie,
                }

                subtask.judge_score = evaluation.score
                subtask.judge_verdict = evaluation.verdict

                if not evaluation.parse_success:
                    if parse_retries >= 1:
                        logger.warning(
                            f"Sub-task {subtask.id}: Judge parse failed after retry, "
                            f"marking as FAILED."
                        )
                        subtask.status = TaskStatus.FAILED
                        break
                    logger.warning(
                        f"Sub-task {subtask.id}: Judge parse failed, retrying without "
                        f"consuming iteration..."
                    )
                    parse_retries += 1
                    iteration -= 1  # nao consome iteration
                    subtask.iterations = iteration
                    feedback_history.append(
                        "IMPORTANTE: O avaliador nao conseguiu parsear sua resposta. "
                        "Gere uma resposta mais estruturada com um JSON valido ao final. "
                        "O JSON deve conter: score, verdict, reasoning, criteria_scores, feedback."
                    )
                    continue

                # Feedback loop: atualizar utility_score das skills injetadas
                if evaluation.parse_success:
                    if evaluation.verdict == Verdict.APPROVED:
                        outcome = "success" if evaluation.score >= 8 else "neutral"
                    elif evaluation.verdict == Verdict.NEEDS_REVISION:
                        outcome = "neutral"
                    else:
                        outcome = "failure" if evaluation.score <= 4 else "neutral"
                    for skill_id in injected_skill_ids:
                        try:
                            worker.skill_manager.update_utility(skill_id, outcome)
                        except Exception as e:
                            logger.debug(f"Skill utility update failed: {e}")

                if evaluation.verdict == Verdict.APPROVED:
                    needs_revision_count = 0
                    subtask.result = worker_output
                    subtask.status = TaskStatus.COMPLETED
                    break
                elif evaluation.verdict == Verdict.NEEDS_REVISION:
                    needs_revision_count += 1
                    feedback_history.append(evaluation.feedback)
                    if iteration >= self.MAX_ITERATIONS_PER_SUBTASK:
                        subtask.result = worker_output
                        subtask.status = TaskStatus.FAILED
                        break
                else:
                    if iteration < self.MAX_ITERATIONS_PER_SUBTASK:
                        needs_revision_count += 1
                        feedback_history.append(evaluation.feedback)
                    else:
                        subtask.result = worker_output
                        subtask.status = TaskStatus.REJECTED

            yield {"type": "subtask_complete", "subtask_id": subtask.id,
                   "status": subtask.status.value,
                   "judge_score": subtask.judge_score,
                   "iterations": subtask.iterations}

        # 3. Sintetizar
        yield {"type": "synthesis_start"}
        final_result = await self._synthesize_results(goal, plan.subtasks)
        plan.final_result = final_result
        plan.status = TaskStatus.COMPLETED

        yield {
            "type": "orchestrator_complete",
            "final_result": final_result,
            "subtasks_summary": [
                {"id": st.id, "status": st.status.value,
                 "judge_score": st.judge_score, "iterations": st.iterations}
                for st in plan.subtasks
            ]
        }

    async def _vote_judge_verdict(
        self,
        task: str,
        worker_output: str,
        role: str,
        first_evaluation,       # JudgeEvaluation
        subtask_id: str = "?",
    ):
        """Vota 3x (default model) em scores borderline.

        - 2/3 ou 3/3 maioria num veredicto -> retorna esse JudgeEvaluation
        - Empate genuíno (1-1-1, todos diferentes) -> escalona pro reasoner (1 chamada autoritativa)

        needs_revision_count >= threshold vai direto pro reasoner, sem passar aqui.
        """
        logger.info(f"Judge voting: score borderline em {subtask_id}, rodando 2 votos adicionais")

        eval2 = await self.judge.evaluate(
            task=task, worker_output=worker_output,
            role=role, model_id=self.judge_default_model,
        )
        eval3 = await self.judge.evaluate(
            task=task, worker_output=worker_output,
            role=role, model_id=self.judge_default_model,
        )

        verdicts = [first_evaluation.verdict, eval2.verdict, eval3.verdict]
        counts = Counter(verdicts)
        top_verdict, top_count = counts.most_common(1)[0]

        logger.info(
            f"Judge voting {subtask_id}: {[v.value for v in verdicts]} "
            f"-> majority={top_verdict.value} ({top_count}/3)"
        )

        if top_count >= 2:
            # Maioria clara: retorna o JudgeEvaluation com o veredicto vencedor
            for ev in (first_evaluation, eval2, eval3):
                if ev.verdict == top_verdict:
                    return ev, False   # (evaluation, escalated)
            return first_evaluation, False

        # Empate genuíno 1-1-1: escala pro reasoner
        logger.info(f"Judge voting {subtask_id}: empate 1-1-1, escalando pro reasoner")
        reasoner_eval = await self.judge.evaluate(
            task=task, worker_output=worker_output,
            role=role, model_id=self.judge_escalation_model,
            timeout=getattr(self, 'reasoner_timeout', None),
        )
        return reasoner_eval, True

    async def _decompose_goal(self, goal: str) -> TaskPlan:
        """Decompoe goal em sub-tasks usando LLM Worker (reutiliza modelo)."""
        llm = await self.llm_pool.get_model(role="general")

        # Ferramentas bloqueadas em execucao autonoma
        from tools.base import ToolPermission
        from tools.builtin import BUILTIN_TOOLS
        blocked = [
            t.name for t in BUILTIN_TOOLS
            if t.permission in (ToolPermission.CONFIRM, ToolPermission.ADMIN)
        ]
        blocked_text = ", ".join(blocked) if blocked else "Nenhuma"

        roles_desc = (
            "Roles disponiveis:\n"
            "- general: Tarefas gerais\n"
            "- coding: Programacao e implementacao\n"
            "- research: Pesquisa e analise de informacao\n"
            "- code_review: Revisao de codigo e boas praticas\n"
            "- security: Analise de seguranca e vulnerabilidades\n"
            "- tdd: Desenvolvimento orientado a testes\n"
            "- planning: Planejamento de features e arquitetura\n"
            "- architect: Decisoes de design de sistemas"
        )
        prompt = (
            f"Voce e um ORCHESTRATOR. Decomponha a tarefa abaixo em sub-tarefas.\n\n"
            f"Goal: {goal}\n\n"
            f"REGRA IMPORTANTE: Se o goal e simples (1 acao clara), NAO decomponha. "
            f"Crie apenas 1 subtask.\n"
            f"Goals simples (NAO decompor): calculator, save_memory, "
            f"criar 1 arquivo, pegar data/hora, responder pergunta direta.\n"
            f"Goals complexos (DECOMPOR): criar multiplas coisas, escrever codigo + testes, "
            f"analisar + relatar, sequencia de tarefas dependentes.\n\n"
            f"RESTRICAO: As seguintes ferramentas estao BLOQUEADAS em "
            f"execucao autonoma: {blocked_text}\n"
            f"Nao crie sub-tarefas que requerem essas ferramentas.\n\n"
            f"{roles_desc}\n\n"
            f"Cada sub-tarefa deve ter: id (task_1, task_2, ...), "
            f"description (clara e especifica), "
            f"role (uma das listadas acima), "
            f"dependencies (lista de ids de que depende, ou [] se independente).\n\n"
            f"Responda APENAS com JSON:\n"
            f'{{"subtasks": ['
            f'{{"id": "task_1", "description": "...", "role": "general", "dependencies": []}}'
            f"]}}"
        )

        response, _ = await llm.generate(prompt, max_tokens=1024, temperature=0.3)

        # Extrai JSON da resposta usando raw_decode (ignora texto extra)
        try:
            decoder = json.JSONDecoder()
            start = response.find('{')
            if start >= 0:
                data, _ = decoder.raw_decode(response, start)
            else:
                raise ValueError("No JSON object found")
            subtasks = [
                SubTask(id=st["id"], description=st["description"],
                        role=st.get("role", "general"),
                        dependencies=st.get("dependencies", []))
                for st in data["subtasks"]
            ]
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Falhou ao parse decomposition: {e}")
            subtasks = [SubTask(id="task_1", description=goal, role="general")]

        if not subtasks:
            subtasks = [SubTask(id="task_1", description=goal, role="general")]

        return TaskPlan(goal=goal, subtasks=subtasks[:self.MAX_SUBTASKS])

    async def _synthesize_results(self, goal: str, subtasks: List[SubTask]) -> str:
        """Sintetiza resultados em resposta final usando LLM Worker."""
        llm = await self.llm_pool.get_model(role="general")

        completed = [st for st in subtasks if st.status == TaskStatus.COMPLETED]
        if not completed:
            return "Nenhuma sub-tarefa foi concluida com sucesso."

        failed_ids = [st.id for st in subtasks if st.status != TaskStatus.COMPLETED]

        sb = []
        for st in completed:
            sb.append(
                f"## {st.id}: {st.description}\n"
                f"Resultado: {st.result or 'N/A'}\n"
                f"Judge: {st.judge_score or 'N/A'}/10"
            )
        if failed_ids:
            sb.append(
                f"## Nota\n"
                f"Sub-tarefas nao concluidas (excluidas da sintese): {', '.join(failed_ids)}"
            )
        subtasks_text = "\n\n".join(sb)

        prompt = (
            f"Voce e um ORCHESTRATOR. Sintetize os resultados das sub-tarefas "
            f"em uma resposta final coerente.\n\n"
            f"Goal original: {goal}\n\n"
            f"Resultados:\n{subtasks_text}\n\n"
            f"Resposta final:"
        )

        response, _ = await llm.generate(prompt, max_tokens=1024, temperature=0.3)
        return response


_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance
