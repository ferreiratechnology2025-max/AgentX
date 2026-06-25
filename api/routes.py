"""FastAPI routes for AgentX"""

import asyncio
import json
import time
import uuid
from typing import Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import yaml

from agent.core import AutonomousAgent, AgentConfig, PermissionRequiredException
from agent.garbage_collector import run_garbage_collector
from agent.skills_compressor import compress_skills_if_needed
from api.telegram_gateway import start_telegram_gateway
from agent.session_manager import load_checkpoint, list_sessions, save_checkpoint
from llm.pool import get_llm_pool
from tools.registry import ToolRegistry
from tools.builtin import set_memory_instance
from memory.persistent import MemoriaPersistente


# ============== CONFIGURAÇÃO ==============

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# ============== INITIALIZATION ==============

print("Initializing AgentX API...")

# Memória persistente
memory = MemoriaPersistente(
    db_path=config['memory']['db_path'],
    index_path=config['memory']['index_path'],
    embedding_dim=config['memory']['embedding_dim'],
    max_memories=config['memory']['max_memories']
)

# Injeta memória nas tools
set_memory_instance(memory)

# LLM Pool (modelos carregam sob demanda via get_model())
llm_pool = get_llm_pool()

# Tool Registry
tool_registry = ToolRegistry()

print(f"System ready: {len(tool_registry)} tools, memory: {memory.index.ntotal} vectors")


# ============== FASTAPI APP ==============

app = FastAPI(
    title="AgentX - Autonomous Agent",
    description="ReAct-based agent with tool use, optimized for 8GB VRAM",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (Web UI)
try:
    app.mount("/static", StaticFiles(directory="api/static"), name="static")
except:
    print(" Static directory not found, UI will not be available")

# Startup hooks
async def periodic_gc_task():
    """Tarefa em background: GC + compressao de skills a cada 24h"""
    await asyncio.sleep(10)
    while True:
        try:
            loop = asyncio.get_running_loop()
            print("[API BACKGROUND] Garbage Collection...")
            await loop.run_in_executor(None, run_garbage_collector, False)
            print("[API BACKGROUND] Verificando skills...")
            current_llm = await llm_pool.get_model(role="general")
            await _periodic_compress(current_llm)
        except Exception as e:
            print(f"[API BACKGROUND ERROR] GC/Skills: {e}")
        await asyncio.sleep(86400)


async def _get_llm():
    """Obtem LLM atual do pool (para uso em callbacks)."""
    return await llm_pool.get_model(role="general")


async def _periodic_compress(llm_instance):
    """Compressao de skills com LLM fornecido."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: asyncio.run(compress_skills_if_needed(llm_instance, dry_run=False))
    )
    if result["status"] == "compressed":
        print(f"[SKILLS] Comprimido: {result['linhas_antes']} -> {result['linhas_depois']} linhas")
    elif result["status"] == "skipped":
        print(f"[SKILLS] Ignorado ({result['motivo']})")


async def auto_unload_loop():
    """Descarrega modelos nao-Router ociosos a cada 60s."""
    await asyncio.sleep(30)
    while True:
        try:
            pool = get_llm_pool()
            if pool.auto_unload and pool.loaded_models:
                current_time = time.time()
                for model_id, info in list(pool.loaded_models.items()):
                    if model_id == pool.ROUTER_MODEL_ID:
                        continue
                    idle = current_time - info.last_used
                    if idle > pool.auto_unload_timeout:
                        print(f"[AUTO-UNLOAD] Descarregando {model_id} (idle {idle:.0f}s)")
                        pool._unload_model(model_id)
        except Exception as e:
            print(f"[AUTO-UNLOAD] Erro: {e}")
        await asyncio.sleep(60)


@app.on_event("startup")
async def _startup():
    loop = asyncio.get_running_loop()
    current_llm = await llm_pool.get_model(role="general")
    await loop.run_in_executor(None, run_garbage_collector, False)
    result = await loop.run_in_executor(
        None, lambda: asyncio.run(compress_skills_if_needed(current_llm, dry_run=False))
    )
    if result["status"] == "compressed":
        print(f"[SKILLS] Comprimido no boot: {result['linhas_antes']} -> {result['linhas_depois']} linhas")
    asyncio.create_task(periodic_gc_task())
    asyncio.create_task(auto_unload_loop())
    global _telegram_gateway
    try:
        _telegram_gateway = await start_telegram_gateway(current_llm, tool_registry)
        print("[API STARTUP] Telegram Gateway iniciado.")
    except Exception as e:
        print(f"[API STARTUP] Telegram Gateway falhou: {e}")
    print("[API STARTUP] GC + Skills Compressor + Auto-Unload + Telegram Gateway.")


@app.on_event("shutdown")
async def _shutdown():
    global _telegram_gateway
    if _telegram_gateway and _telegram_gateway._running:
        await _telegram_gateway.stop()
        print(" [API SHUTDOWN] Telegram Gateway parado.")


# ============== SESSION MANAGEMENT ==============

# Sessões ativas em memória (em produção usaria Redis)
ACTIVE_SESSIONS: Dict[str, AutonomousAgent] = {}

# Telegram bot gateway (iniciado no startup)
_telegram_gateway = None


# ============== MODELS ==============

class AgentRequest(BaseModel):
    goal: str = Field(..., description="Objetivo do agente", min_length=1, max_length=500)
    context: Optional[str] = Field(None, description="Contexto adicional", max_length=2000)
    max_steps: Optional[int] = Field(8, ge=1, le=20, description="Máximo de steps")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Temperatura")
    stream: bool = Field(True, description="Habilitar streaming")
    yolo_mode: bool = Field(False, description="Modo autônomo - aprova todas as tools automaticamente")


class ApprovalRequest(BaseModel):
    approved: bool = Field(..., description="Aprovar ou rejeitar ação")
    reason: Optional[str] = Field(None, description="Motivo da rejeição")


# ============== RATE LIMITER ==============

class RateLimiter:
    """Rate limiter simples em memória"""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def check(self, client_id: str = "default"):
        now = datetime.now().timestamp()
        window_start = now - 60
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Limpa requests antigos
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        if len(self.requests[client_id]) >= self.requests_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        self.requests[client_id].append(now)


rate_limiter = RateLimiter(config['api'].get('rate_limit', 100))


# ============== ENDPOINTS ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a interface web"""
    try:
        with open("api/static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <body>
                <h1>AgentX API</h1>
                <p>Web UI not found. Please ensure api/static/index.html exists.</p>
                <p>Available endpoints:</p>
                <ul>
                    <li>POST /agent/run - Execute agent</li>
                    <li>GET /agent/status - Agent status</li>
                    <li>GET /tools/list - List tools</li>
                    <li>GET /health - Health check</li>
                </ul>
            </body>
        </html>
        """, status_code=200)


@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    """
    Executa o agente autônomo com streaming SSE
    
    Exemplo:
    ```bash
    curl -X POST http://localhost:8000/agent/run \\
      -H "Content-Type: application/json" \\
      -d '{"goal": "Calculate 15% of 250 and save to memory"}'
    ```
    """
    await rate_limiter.check()
    
    session_id = str(uuid.uuid4())[:8]
    
    agent_config = AgentConfig(
        max_steps=request.max_steps,
        temperature=request.temperature,
        parallel_tools=True,
        verbose=True,
        yolo_mode=request.yolo_mode
    )
    
    current_llm = await llm_pool.get_model(role="general")
    agent = AutonomousAgent(current_llm, tool_registry, agent_config)
    ACTIVE_SESSIONS[session_id] = agent
    
    async def event_generator():
        yield f"data: {json.dumps({'type': 'session_init', 'session_id': session_id})}\n\n"
        
        try:
            async for event in agent.run(request.goal, request.context, session_id=session_id):
                yield f"data: {json.dumps(event)}\n\n"
        except PermissionRequiredException:
            yield f"data: {json.dumps({'type': 'awaiting_approval', 'pending': agent.state.pending_action})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            # Limpa sessão após 5 minutos
            async def cleanup():
                await asyncio.sleep(300)
                ACTIVE_SESSIONS.pop(session_id, None)
            asyncio.create_task(cleanup())
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/agent/session/{session_id}/approve")
async def approve_action(session_id: str, approval: ApprovalRequest):
    """
    Aprova ou rejeita uma ação pendente
    
    Exemplo:
    ```bash
    curl -X POST http://localhost:8000/agent/session/abc123/approve \\
      -H "Content-Type: application/json" \\
      -d '{"approved": true}'
    ```
    """
    agent = ACTIVE_SESSIONS.get(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if agent.state.status != "awaiting_approval":
        raise HTTPException(status_code=400, detail="No pending action")
    
    async def event_generator():
        if not approval.approved:
            # Rejeitou - delega tratamento ao resume_loop
            yield f"data: {json.dumps({'type': 'status', 'content': ' Ação rejeitada pelo usuário'})}\n\n"
            async for event in agent.resume_loop(rejected=True):
                yield f"data: {json.dumps(event)}\n\n"
        else:
            # Aprovou - executa e retoma
            yield f"data: {json.dumps({'type': 'status', 'content': ' Ação aprovada pelo usuário. Executando...'})}\n\n"
            
            async for event in agent.resume_loop():
                yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/sessions")
async def get_sessions():
    """Lista checkpoints de sessões disponíveis em disco"""
    ids = list_sessions()
    sessions = []
    for sid in ids:
        cp = load_checkpoint(sid)
        if cp:
            sessions.append({
                "session_id": cp.session_id,
                "goal": cp.goal,
                "status": cp.status,
                "steps": cp.steps_count,
                "last_updated": cp.last_updated,
            })
    return {"sessions": sessions, "active": len(ACTIVE_SESSIONS)}


@app.get("/agent/status")
async def get_agent_status():
    """Retorna status do sistema"""
    pool_status = llm_pool.get_status()
    return {
        "status": "healthy",
        "tools": len(tool_registry),
        "tools_list": tool_registry.get_names(),
        "memory_count": memory.index.ntotal if hasattr(memory, 'index') else 0,
        "active_sessions": len(ACTIVE_SESSIONS),
        "saved_sessions": len(list_sessions()),
        "pool": pool_status,
        "vram": pool_status.get("vram", {}),
    }


@app.get("/tools/list")
async def list_tools():
    """Lista todas as ferramentas disponíveis"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "permission": tool.permission.value,
                "max_uses": tool.max_uses_per_session
            }
            for tool in tool_registry.list_all()
        ]
    }


@app.get("/memories")
async def list_memories(limit: int = 20, search: Optional[str] = None):
    """Lista ou busca memórias"""
    if search:
        results = await memory.buscar(search, k=limit)
        return {"results": results}
    else:
        memories = await memory.listar_recentes(limit)
        return {"memories": memories}


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    """Remove uma memória"""
    success = await memory.deletar(memory_id)
    if success:
        return {"status": "deleted", "id": memory_id}
    raise HTTPException(status_code=404, detail="Memory not found")


@app.post("/judge/evaluate")
async def judge_evaluate(request: Request):
    """
    Avalia output de worker com Judge Agent.

    Body:
    {
        "task": "descricao da tarefa",
        "worker_output": "output gerado",
        "role": "coding|research|general",
        "context": "contexto adicional (opcional)"
    }
    """
    from agent.judge import get_judge

    body = await request.json()
    task = body.get("task")
    worker_output = body.get("worker_output")
    role = body.get("role", "general")
    context = body.get("context")

    if not task or not worker_output:
        raise HTTPException(status_code=400, detail="task e worker_output sao obrigatorios")

    judge = get_judge()
    evaluation = await judge.evaluate(task, worker_output, role, context)

    return {
        "score": evaluation.score,
        "verdict": evaluation.verdict.value,
        "reasoning": evaluation.reasoning,
        "criteria_scores": evaluation.criteria_scores,
        "feedback": evaluation.feedback
    }


@app.post("/orchestrator/run")
async def orchestrator_run(request: dict):
    """
    Executa goal complexo com Orchestrator (SSE streaming).

    Body:
    {
        "goal": "Construir uma API REST com autenticacao"
    }
    """
    from agent.orchestrator import get_orchestrator

    goal = request.get("goal")
    if not goal:
        raise HTTPException(status_code=400, detail="goal e obrigatorio")

    orchestrator = get_orchestrator()

    async def event_generator():
        async for event in orchestrator.run(goal):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health_check():
    """Health check para monitoramento"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
