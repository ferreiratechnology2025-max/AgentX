# Architecture

Multi-agent orchestration system with Worker, Judge, and supporting components.

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
│  ┌────────────────────────────────────────────────────┐ │
│  │                 Orchestrator                        │ │
│  │  Decompose goal → assign roles → execute → evaluate │ │
│  │         → feedback loop → synthesize                │ │
│  └──────────────┬──────────────────────────────────────┘ │
│                 │                                         │
│  ┌──────────────▼──────────────────────────────────────┐ │
│  │             Agent Factory                            │ │
│  │  Creates specialized agents with:                    │ │
│  │  • Role-specific system prompt + ECC rules           │ │
│  │  • Tool subset                                   │ │
│  │  • Model routing via LLM Pool                       │ │
│  └──────────────┬──────────────────────────────────────┘ │
│                 │                                         │
│  ┌──────────────▼──────────────────────────────────────┐ │
│  │              Worker Agents (N instances)              │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐              │ │
│  │  │ coding   │ │ security │ │  tdd     │  ...         │ │
│  │  └──────────┘ └──────────┘ └──────────┘              │ │
│  │  Each runs ReAct loop: Thought → Action → Observation │ │
│  └──────────────┬──────────────────────────────────────┘ │
│                 │                                         │
│  ┌──────────────▼──────────────────────────────────────┐ │
│  │              Judge Agent                             │ │
│  │  Evaluates worker output with:                       │ │
│  │  • Role-specific rubrics                             │ │
│  │  • CoT reasoning + structured JSON verdict           │ │
│  │  • parse_success → NEEDS_REVISION/APPROVED/REJECTED  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌────────────────────────────────────────────────────┐   │
│  │              Supporting Systems                     │   │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────────┐    │   │
│  │  │ LLM Pool   │ │ Memory   │ │ Tool Registry  │    │   │
│  │  │ Manager    │ │ (FAISS + │ │ (6 tools with  │    │   │
│  │  │ (lazy load,│ │  SQLite) │ │  permissions)  │    │   │
│  │  │ VRAM mgmt) │ │          │ │                │    │   │
│  │  └────────────┘ └──────────┘ └────────────────┘    │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Components

### Orchestrator
- **File**: `agent/orchestrator.py`
- **Role**: Goal decomposition, task assignment, feedback loop, result synthesis
- **Model**: Qwen Q3_K_M (same as Worker pool)
- **Flow**: Decompose → create workers → execute → evaluate → retry if needed → synthesize
- **Retry**: Up to 1 parse retry + Judge feedback loop within same iteration

### Workers
- **File**: `agent/core.py` (AutonomousAgent)
- **Role**: Execute sub-tasks via ReAct loop with tool subset
- **8 Roles**: general, coding, research, code_review, security, tdd, planning, architect
- **Tools**: Subset defined per role in `agent/factory.py`
- **Validation**: GBNF grammar + Pydantic + `_validate_output()` regex parser

### Judge
- **File**: `agent/judge.py`
- **Model**: Phi-4-mini Q5_K_M (different from Worker to avoid bias)
- **Temperature**: 0.1 (deterministic)
- **Evaluation**: Role-specific rubrics, CoT reasoning + structured JSON
- **Verdicts**: APPROVED, NEEDS_REVISION, REJECTED
- **Parse**: Brace-balancing JSON extraction, `parse_success` field, score normalization (/10)

### LLM Pool Manager
- **File**: `llm/pool.py` (pool), `llm/manager.py` (per-model)
- **Strategy**: Lazy loading (models loaded on demand)
- **VRAM**: Tracks per-model usage, warns at > 90%, provides fallback
- **chat_format**: Per-model (Qwen: chatml, Phi-4-mini: null for native template)
- **Inference**: Synchronous (no `asyncio.to_thread` — avoids thread pool corruption)

### Agent Factory
- **File**: `agent/factory.py`
- **Creates**: Specialized agents with role-specific prompts, tools, and models
- **ECC Integration**: Loads agent prompts from `data/knowledge/ecc_agents/`
- **Rules**: Injects ECC rules into system prompt (< 2000 tokens per role)

### Tool Registry
- **File**: `tools/registry.py`
- **6 Tools**: calculator, current_datetime, save_memory, project_manager, git_worker, process_orchestrator
- **Permissions**: SAFE (auto), CONFIRM (requires approval), ADMIN (blocked)

### Memory System
- **File**: `memory/` (FAISS + SQLite)
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2`
- **Index**: FAISS IVF with LRU cache
- **Persistence**: SQLite database with checkpoint/restore

### Sandbox
- **File**: `tools/sandbox.py`
- **Protection**: Symlink protection, blocked path patterns
- **Scope**: All file operations limited to `workspace/` directory

## Data Flow

```
User Goal
  │
  ▼
Orchestrator._decompose_goal()
  │  LLM generates JSON: {subtasks: [{id, description, role, dependencies}]}
  │  raw_decode parsing ignores extra text
  ▼
For each sub-task:
  │
  ├── Agent Factory creates worker with role-specific prompt + ECC rules
  │
  ├── Worker executes ReAct loop:
  │     Thought → Action → Observation → (repeat) → Final Answer
  │     • GBNF grammar guides structured output
  │     • Pydantic validates tool args
  │     • _validate_output() regex-parses ReAct format
  │     • MAX_PARSE_ATTEMPTS=3 with prompt reinforcement on retry
  │
  ├── Judge.evaluate()
  │     • System prompt separates judge role (native template)
  │     • User prompt with triple anchor (### INICIO ###/### CRITERIOS ###/### FIM ###)
  │     • Brace-balancing JSON extraction
  │     • parse_success=False → orchestrator retries with structured output feedback
  │
  ├── If NEEDS_REVISION:
  │     Worker retries with Judge feedback in context
  │     (iteration -= 1, up to max_iterations)
  │
  └── If APPROVED:
        Result stored for synthesis
  │
  ▼
Orchestrator synthesizes final result from all sub-task outputs
  │
  ▼
Return to user
```

## VRAM Management

| Component | Model | VRAM | Notes |
|-----------|-------|------|-------|
| Worker | Qwen Q3_K_M | ~3.5 GB | Q3_K_M saves ~40% vs Q4_K_M |
| Judge | Phi-4-mini Q5_K_M | ~3.1 GB | Uses native phi-3 template |
| System overhead | — | ~2 GB | FastAPI, FAISS, CUDA runtime |
| **Total** | | **~8.6 GB** | Headroom: ~3.4 GB |
| VRAM limit | — | 12.0 GB | config.yaml `max_vram_gb` |

Both models loaded simultaneously. Lazy loading and fallback prevent OOM.

## Security Architecture

```
User Input
  │
  ├── Tool Permission Check (SAFE / CONFIRM / ADMIN)
  │     ├── SAFE → execute immediately
  │     ├── CONFIRM → HITL approval via Web UI
  │     └── ADMIN → blocked
  │
  ├── GBNF Grammar → forces valid JSON output from LLM
  │
  ├── Pydantic Validation → validates all tool arguments
  │
  ├── Sandbox Filesystem → symlink protection, blocked patterns
  │
  └── VRAM Monitor → warns at > 90%
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| No `asyncio.to_thread` for inference | Prevents thread pool corruption after multiple LLM calls |
| `chat_format: null` for Phi-4-mini | Uses native phi-3 template; chatml corrupts tokenization |
| MAX_PARSE_ATTEMPTS = 3 | LLMs are probabilistic; retry with prompt reinforcement |
| Judge temp = 0.1 | Evaluation should be deterministic |
| ECC rules < 2000 tokens | Context window is a hard limit (Worker: 16K, Judge: 8K) |
| Dual model (different arch) | Prevents self-enhancement bias |
| Synchronous inference | `asyncio.to_thread` causes hangs after ~3-4 LLM calls |
