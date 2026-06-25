# AgentX — Multi-Agent Autonomous System

Multi-agent orchestration with ReAct loop, tool use, persistent memory (SQLite + FAISS),
real-time Web UI, Judge feedback loop, and **closed-loop skill learning**. Runs entirely
on Ollama — no GGUF downloads, no llama-cpp-python. Optimized for 12 GB VRAM with
intelligent resource management and three-tier model architecture.

## Features

### Core Architecture
- **Multi-Agent Architecture** — Orchestrator decomposes goals, Workers execute sub-tasks, Judge evaluates outputs
- **8 Specialized Roles** — coding, research, general, code_review, security, tdd, planning, architect
- **ReAct Loop** — Thought → Action → Observation cycle with structured JSON output (constrained decoding via Ollama `format:`)
- **6 Built-in Tools** — calculator, datetime, save_memory, project_manager, git_worker, process_orchestrator
- **Human-in-the-Loop (HITL)** — CONFIRM-permission tools block for operator approval; auto-approved in orchestrator `yolo_mode`
- **Persistent Memory** — Semantic search via FAISS + SQLite with LRU cache
- **Async Architecture** — Non-blocking I/O for concurrent multi-agent execution
- **Streaming API** — SSE (Server-Sent Events) for real-time traces
- **Web UI** — Live visualization of agent reasoning and approval controls

### Safety & Robustness
- **Safety System** — Tool permissions (SAFE / CONFIRM / ADMIN), sandbox filesystem, Pydantic validation
- **Parse Robustness** — Constrained JSON decoding + Pydantic validation + retry with prompt reinforcement (3 attempts, backoff [0.5, 1.5, 3.0]s)
- **Auto-Continue** — Up to 2 automatic step extensions preventing premature cutoffs
- **Session Lifecycle** — PENDING → RUNNING → COMPLETED / FAILED / REJECTED; retries exhausted on NEEDS_REVISION → FAILED (never stuck)
- **HITL Retry with Reformulation** — Rejected tools trigger reformulation loop (up to `hitl_retry_attempts`) before subtask fails

### Intelligence & Learning
- **Judge Agent** — Evaluates workers with role-specific rubrics; CoT + structured JSON verdict; considers goal context when evaluating efficiency
- **Judge Voting** — 3× majority vote for scores ≤ threshold; genuine 1-1-1 tie escalates to Reasoner tier
- **Security Checklist Voter** — 3× generation with per-item majority vote (2/3) for security role outputs
- **Security Checklist Validator** — Deterministic post-generation regex check; guarantees all 8 OWASP items present
- **Skills Learning (Closed-Loop)** — Micro post-mortem extracts technical rules from successful trajectories with **utility tracking** (Bayesian update), **feedback loop** (Judge → Skills), **forgetting curve** (half-life 30 days), **relevance-based selection** (utility_score × role_bonus), and **quarantine period** (new skills require 3+ uses before injection)
- **Skills Compressor** — Semantic deduplication of learned rules every 24h (scheduled, preserves metadata)
- **Garbage Collector** — Automatic cleanup of expired sessions (>7 days)

### Resource Management
- **LLM Pool Manager** — Lazy loading, three model tiers, role-based routing
- **VRAM Tracking** — Real-time monitoring via `nvidia-smi` with usage alerts
- **LRU Unloading** — Automatic model unloading when VRAM > 85% (Router protected)
- **Proactive Unloading** — Models auto-unload after 5min idle (`auto_unload` config)
- **Reasoner Timeout + Fallback** — 30s timeout for Reasoner loading; fallback to enhanced Workhorse prompt if unavailable
- **ECC Integration** — Role-specific decomposed prompts with few-shots loaded from `data/knowledge/ecc_agents/`

## Architecture

```
User Goal
  │
  ▼
Orchestrator  ──uses──►  Workhorse (gemma3ne4b, num_ctx=12288)
  │  decompose → assign roles → execute → evaluate → synthesize
  │
  ├── Worker [security]     ──►  Workhorse + Security Voter (3× per-item majority)
  ├── Worker [code_review]  ──►  Workhorse
  ├── Worker [coding]       ──►  Workhorse
  ├── Worker [tdd]          ──►  Workhorse
  ├── Worker [architect]    ──►  Workhorse
  ├── Worker [planning]     ──►  Workhorse
  ├── Worker [research]     ──►  Workhorse
  └── Worker [general]      ──►  Workhorse
        │
        ▼
    Judge (Workhorse by default)
    • Role-specific rubrics, CoT + structured JSON verdict
    • Considers goal context when evaluating efficiency
    • score ≤ 4  ──►  voting 3× (majority 2/3)
    • tie 1-1-1  ──►  Reasoner (qwen35-9b, 30s timeout)
    │                    └── fallback: Workhorse enhanced prompt
    • NEEDS_REVISION ≥ 2  ──►  Reasoner direct escalation
    • Only COMPLETED subtasks enter synthesis
        │
        ▼
    Skill Manager (feedback loop)
    • Captures injected skill_ids per worker
    • Maps Judge verdict → outcome (success/failure/neutral)
    • Bayesian update of utility_score
    • Quarantine: new skills require 3+ uses before injection
    • Forgetting curve + pruning on next selection
```

## Model Tiers

| Tier | Model | VRAM | num_ctx | Used for |
|------|-------|------|---------|----------|
| Router | `granite4htiny` | ~4877 MiB | 4096 | task routing and classification |
| Workhorse | `gemma3ne4b` | ~3377 MiB | **12288** | all workers, judge default, synthesis |
| Reasoner | `qwen35-9b` | ~5500 MiB | 16384 | escalation only (lazy loaded, 30s timeout) |

**Resource Management:**
- Router + Workhorse resident simultaneously: ~8254 MiB total
- Headroom on 12 GB VRAM: ~3862 MiB (baseline)
- Clean system baseline (OS + Ollama only): ~590 MiB
- **Auto-unload**: models idle > 5min are automatically unloaded (Router protected)
- **LRU eviction**: when VRAM > 85%, least-recently-used model is unloaded before loading new one
- **Reasoner fallback**: if Reasoner fails to load (timeout/OOM), escalates to Workhorse with enhanced prompt

## Setup

### 1. Install dependencies

```bash
# Production
pip install -r requirements.txt

# Development (includes pytest, black, ruff)
pip install -e ".[dev]"
```

### 2. Install Ollama and pull models

```bash
# Install Ollama: https://ollama.com/download
ollama pull granite4htiny
ollama pull gemma3ne4b
ollama pull qwen35-9b
```

### 3. Configure `config.yaml`

```yaml
ollama:
  base_url: "http://localhost:11434"

llm_pool:
  auto_unload: true              # Enable proactive model unloading
  auto_unload_timeout: 300       # Seconds before idle model is unloaded
  reasoner_timeout: 30           # Seconds to wait for Reasoner loading
  models:
    - id: "granite4htiny"
      role_preference: ["routing", "classification"]
      num_ctx: 4096
    - id: "gemma3ne4b"
      role_preference: ["general", "coding", "research", "orchestrator",
                        "judging", "evaluation", "code_review", "security",
                        "tdd", "planning", "architect"]
      num_ctx: 12288             # Increased from 8192 for longer ReAct loops
    - id: "qwen35-9b"
      role_preference: ["reasoning"]
      num_ctx: 16384

judge:
  default_model: "gemma3ne4b"
  escalation_model: "qwen35-9b"
  escalation_needs_revision: 2
  escalation_score_threshold: 4
  hitl_retry_attempts: 2         # Max reformulation attempts after HITL rejection

agent:
  max_steps: 15
  temperature: 0.3
  parallel_tools: false
  yolo_mode: false
```

### 4. Run

```bash
python main.py
```

Server at http://localhost:8000 — Web UI at http://localhost:8000/

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| POST | `/agent/run` | Execute agent with goal (SSE streaming). Supports `yolo_mode: true` for auto-approval |
| POST | `/agent/session/{id}/approve` | Approve/reject pending action |
| GET | `/agent/status` | System status (tools, memory, **VRAM**, loaded models) |
| POST | `/judge/evaluate` | Evaluate worker output with Judge Agent |
| POST | `/orchestrator/run` | Execute complex goal with multi-agent orchestration (SSE) |
| GET | `/tools/list` | List available tools |
| GET | `/memories` | List or search memories |
| DELETE | `/memories/{id}` | Delete a memory |
| GET | `/sessions` | List saved session checkpoints |
| GET | `/health` | Health check |

**Example `/agent/run` request with yolo_mode:**
```json
{
  "goal": "Create a file called test.txt with content Hello World",
  "yolo_mode": true,
  "max_steps": 10,
  "temperature": 0.7
}
```

**Example `/agent/status` response:**
```json
{
  "status": "ok",
  "vram": {
    "used_gb": 6.2,
    "total_gb": 12.0,
    "free_gb": 5.8,
    "usage_percent": 51.7
  },
  "pool": {
    "configured_models": ["granite4htiny", "gemma3ne4b", "qwen35-9b"],
    "connected_models": ["granite4htiny", "gemma3ne4b"]
  },
  "tools": ["calculator", "project_manager", "git_worker", "..."],
  "memory_count": 142
}
```

## Built-in Tools

| Tool | Description | Permission |
|------|-------------|------------|
| `calculator` | Safe math expression evaluator | SAFE |
| `current_datetime` | Current system date and time | SAFE |
| `save_memory` | Save information to persistent memory | SAFE |
| `project_manager` | Read/write/list files in workspace and project source | CONFIRM |
| `git_worker` | Controlled git operations (status, stage, commit) | CONFIRM (HITL) |
| `process_orchestrator` | Async process control (start/status/stop) | CONFIRM |

### `project_manager` sandbox

| Operation | Allowed paths |
|-----------|---------------|
| read | `./workspace/`, `./data/knowledge/`, project root (source files) |
| write | `./workspace/`, `./data/knowledge/` only |

**Blocked in all cases:** `.env*`, `models/`, `private/`, `__pycache__`.

HITL applies in interactive mode; auto-approved in orchestrator `yolo_mode=True`.

## Specialized Roles

| Role | Description | Tools |
|------|-------------|-------|
| `general` | General assistant | save_memory |
| `coding` | Expert programmer | calculator, project_manager, process_orchestrator |
| `research` | Information gathering | save_memory |
| `code_review` | Code quality and security review (8-item checklist) | calculator, project_manager |
| `security` | OWASP vulnerability detection (8-class checklist + voter) | calculator, project_manager, process_orchestrator |
| `tdd` | Test-driven development | calculator, project_manager, process_orchestrator |
| `planning` | Feature planning | save_memory, project_manager |
| `architect` | System design decisions | save_memory, project_manager |

### ECC Role Prompts

Each role loads a decomposed system prompt from `data/knowledge/ecc_agents/<role>.md` containing:
- Role directive with output format constraints
- Few-shot examples using correct tool call syntax
- Hard rules (e.g., "never give a general review — answer the 8-item checklist only")
- Critical rules (e.g., "after successful tool call, generate Final Answer with code content")

Security and code_review roles use binary SIM/NÃO checklists with 8 numbered items. Range notation (`2-8. NÃO`) is supported in both generation and parsing.

The coding role includes explicit instructions to:
- Include code content in Final Answer for Judge evaluation
- Preserve original goal requirements when receiving feedback
- Only run tests if explicitly requested in the goal

## Security Checklist Quality Pipeline

For the `security` role, three mechanisms ensure complete and accurate outputs:

1. **Validator** — Post-generation regex verifies all 8 items present (individual `N.` or range `N-M.`). One automatic retry with correction message if any item is missing.
2. **Voter** — 3× LLM calls reusing the same tool observation (no extra file reads). Per-item majority vote (2/3) determines final verdict for each of the 8 classes.
3. **Judge voting** — Scores ≤ 4 trigger 3× judge calls (majority wins); 1-1-1 tie escalates to Reasoner.

## Skills System (Closed-Loop Learning)

The skills system transforms AgentX from a stateless agent into a **self-improving system** that learns from its own trajectory.

### Lifecycle

```
Successful trajectory (>2 steps)
  │
  ▼
Skill Extraction (LLM micro post-mortem)
  │  Extracts technical rule (1-2 lines)
  │  Initializes: utility_score=0.5, usage_count=0
  ▼
Structured Storage (data/knowledge/skills_learnt.md)
  │  HTML comments with metadata:
  │  skill_id, extracted_at, role, usage_count,
  │  success_count, failure_count, last_used, utility_score
  ▼
Quarantine Period
  │  New skills (usage_count < 3) are NOT injected
  │  Prevents unvalidated rules from polluting context
  ▼
Relevance Selection (per _think() call)
  │  1. Filter: usage_count ≥ 3 (quarantine passed)
  │  2. Apply forgetting curve (half-life 30 days)
  │  3. Prune low-utility skills (usage≥10, score<0.3)
  │  4. Score = utility_score × role_bonus (1.2× if role matches)
  │  5. Select top-k within token budget (400 tokens)
  ▼
Injection into System Prompt
  │  "## REGRAS APRENDIDAS" section
  ▼
Feedback Loop (after Judge evaluation)
  │  Judge verdict → outcome:
  │    score ≥ 8 → "success"
  │    score ≤ 4 → "failure"
  │    otherwise → "neutral"
  ▼
Bayesian Update
  │  alpha = 1 + success_count
  │  beta = 1 + failure_count
  │  utility_score = alpha / (alpha + beta)
  ▼
Scheduled Compression (every 24h)
     Cluster by semantic similarity (threshold 0.85)
     Merge with LLM, preserve aggregated metadata
```

### Skill File Format

```markdown
<!-- skill_id: abc123-def456 -->
<!-- extracted_at: 2026-06-25T14:30:00Z -->
<!-- source_trajectory: task_xyz -->
<!-- role: security -->
<!-- usage_count: 5 -->
<!-- success_count: 4 -->
<!-- failure_count: 0 -->
<!-- last_used: 2026-06-25T16:00:00Z -->
<!-- utility_score: 0.83 -->
Ao executar security review de APIs REST, verifique rate limiting antes de reportar brute force.
---
```

### Quarantine Period

New skills start with `usage_count = 0` and `utility_score = 0.5` (prior neutro). They are **not injected** into the context until they pass the quarantine threshold (`usage_count ≥ 3`). This prevents unvalidated rules from polluting the context.

**Rationale**: Skills extracted from a single successful trajectory may be overfitted or context-specific. Requiring 3+ uses ensures the skill is generalizable and actually helpful.

### Forgetting Curve

Skills decay over time using Ebbinghaus-inspired retention:

```
retention = e^(-days_since_use / 30)
```

- Skill used yesterday: retention ≈ 0.97
- Skill unused for 30 days: retention ≈ 0.37
- Skill unused for 90 days: retention ≈ 0.05
- Skill unused for 175 days: retention ≈ 0.003

### Pruning

Skills with `usage_count ≥ 10` and `utility_score < 0.3` are automatically pruned during selection and compression. This prevents low-quality skills from polluting the context.

### Migration

Existing plain-text skills can be migrated to the structured format:

```bash
python scripts/migrate_skills.py
```

This generates UUIDs, initializes `utility_score=0.5`, and preserves all existing rules.

## Human-in-the-Loop (HITL) Flow

```
Agent decides to call CONFIRM tool
  │
  ├── yolo_mode=False (interactive /agent/run)
  │     └── raises PermissionRequiredException
  │           └── SSE "awaiting_approval" event
  │                 └── Web UI shows approval box
  │                       ├── Approve → tool executes
  │                       └── Reject  → error observation injected
  │                             └── ReactLoop receives retry_with_reformulation=True
  │                                   └── Agent reformulates action (up to hitl_retry_attempts)
  │                                         ├── Reformulation succeeds → tool executes
  │                                         └── Reformulation fails → subtask FAILED
  │
  └── yolo_mode=True (orchestrator workers)
        └── tool executes automatically
```

### HITL Retry with Reformulation

When an operator rejects a CONFIRM tool call, the agent does not immediately fail the subtask. Instead:

1. **Error observation injected**: "Tool X rejected by operator. Reason: {reason}. Please reformulate with a safer alternative."
2. **Agent reformulates**: The ReactLoop continues, allowing the agent to propose a different action.
3. **Retry limit**: Up to `hitl_retry_attempts` (default: 2) reformulations before subtask fails.

**Example**:
- Agent proposes `git push --force` → operator rejects ("destructive operation")
- Agent reformulates to `git push` (without force) → operator approves → success
- If agent proposes another destructive operation → operator rejects again → subtask fails after 2 attempts

## Configuration Reference (`config.yaml`)

| Section | Key fields |
|---------|------------|
| `ollama` | `base_url` — Ollama endpoint |
| `llm` | `model_id` — default model when pool not used |
| `llm_pool` | `auto_unload`, `auto_unload_timeout`, `reasoner_timeout`, `models[]` — list of `{id, role_preference, num_ctx}` |
| `judge` | `default_model`, `escalation_model`, `escalation_needs_revision`, `escalation_score_threshold`, `hitl_retry_attempts` |
| `agent` | `max_steps`, `temperature`, `parallel_tools`, `yolo_mode` |
| `api` | `host`, `port`, `rate_limit` |
| `memory` | `embedding_dim`, `max_memories`, `db_path`, `index_path` |

## Development

### Project Structure

```
agentx/
├── main.py                      # FastAPI entry point + background tasks
├── config.yaml                  # Configuration
├── pyproject.toml               # Package metadata + dev dependencies
├── requirements.txt             # Production dependencies
│
├── agent/                       # Core agent logic (764 lines, 4 modules)
│   ├── core.py                  # AutonomousAgent orchestrator (316 lines)
│   ├── react_loop.py            # ReAct thought/action loop (222 lines)
│   ├── tool_executor.py         # Tool execution + HITL + reformulation (83 lines)
│   ├── skill_manager.py         # Skills learning + utility tracking (143 lines)
│   ├── orchestrator.py          # Multi-worker coordination + HITL retry loop
│   ├── judge.py                 # Evaluation with rubrics + Reasoner fallback
│   ├── factory.py               # Agent creation by role
│   ├── prompt_builder.py        # System prompt construction
│   ├── ecc_loader.py            # ECC rules loader
│   ├── session_manager.py       # Checkpoint persistence
│   ├── skills_compressor.py     # Semantic deduplication
│   ├── garbage_collector.py     # Session cleanup
│   └── state.py                 # State dataclasses
│
├── llm/                         # LLM management
│   ├── manager.py               # Ollama HTTP client + parse retry
│   └── pool.py                  # 3-tier pool + VRAM tracking + LRU unload
│
├── tools/                       # Built-in tools
│   ├── base.py                  # BaseTool abstract class
│   ├── builtin.py               # calculator, datetime, save_memory
│   ├── schemas.py               # Pydantic schemas
│   ├── project_manager.py       # Filesystem sandbox
│   ├── process_orchestrator.py  # Async subprocess
│   ├── sandbox.py               # Path validation
│   ├── git_worker.py            # Git operations
│   └── registry.py              # Tool registry
│
├── memory/                      # Persistent memory
│   ├── persistent.py            # FAISS + SQLite
│   └── cache.py                 # LRU cache
│
├── api/                         # HTTP API
│   ├── routes.py                # FastAPI routes + SSE
│   └── telegram_gateway.py      # Telegram bot integration
│
├── tests/                       # Test suite (64 tests)
│   ├── test_skills.py           # Skills system (10 tests)
│   ├── test_judge.py            # Judge evaluation (7 tests)
│   ├── test_schemas.py          # Pydantic schemas (5 tests)
│   ├── test_sandbox.py          # Sandbox validation (6 tests)
│   └── ...
│
├── scripts/
│   └── migrate_skills.py        # Skills format migration
│
├── data/knowledge/
│   ├── skills_learnt.md         # Learned skills (structured)
│   ├── ecc_rules/               # ECC rules (5 files)
│   └── ecc_agents/              # Role prompts (8 files)
│
└── models/                      # GGUF models + Modelfiles
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agent --cov-report=term-missing

# Specific test file
pytest tests/test_skills.py -v
```

### Code Quality

```bash
# Format code
black agent/ tests/

# Lint
ruff check agent/ tests/

# Type check (if mypy configured)
mypy agent/
```

### CI/CD

GitHub Actions workflow (`.github/workflows/test.yml`) runs on every push/PR:
- Installs dependencies with `pip install -e ".[dev]"`
- Runs `pytest` with coverage
- Uploads coverage to Codecov

## Technical Details

- **LLM Backend**: Ollama REST API (`/api/chat`), streamed token generation
- **Constrained Decoding**: Ollama `format:` parameter with JSON Schema ensures valid output at token level
- **Parse Retry**: 3 attempts with backoff [0.5, 1.5, 3.0]s + prompt reinforcement
- **Memory**: sentence-transformers (`all-MiniLM-L6-v2`) for embeddings, FAISS IVF index with proper training
- **Cache**: LRU cache with TTL for embeddings and search results
- **Streaming**: SSE via FastAPI `StreamingResponse`
- **Tool errors**: Failures surface as explicit observation text — no silent empty strings
- **Orchestrator subtask lifecycle**: PENDING → RUNNING → COMPLETED / FAILED / REJECTED. Retries exhausted on NEEDS_REVISION → FAILED (not stuck at RUNNING)
- **Process Orchestrator**: `asyncio.create_subprocess_exec` with command whitelist and token blocklist
- **VRAM Monitoring**: `nvidia-smi` via subprocess (no torch dependency)
- **Background Tasks**: Auto-unload loop (60s), garbage collector (24h), skills compressor (24h)
- **Reasoner Timeout**: 30s timeout for loading; fallback to Workhorse enhanced prompt if unavailable
- **HITL Reformulation**: Up to `hitl_retry_attempts` retry attempts after operator rejection before subtask fails
- **Skills Quarantine**: New skills require `usage_count ≥ 3` before injection (configurable via `QUARANTINE_THRESHOLD`)
- **Path Resolution**: Relative paths automatically resolve against `workspace/` directory
- **Final Answer Parser**: Prioritizes `action_name` when `final_answer` is empty; otherwise returns `final_answer`
- **Judge Calibration**: Considers goal context when evaluating efficiency (doesn't penalize inherent complexity of specified approach)

## Performance Metrics

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| VRAM peak usage | 96.8% (OOM risk) | ~50% (auto-unload) | -47% |
| Available context | 8192 tokens | 12288 tokens | +50% |
| Parse failure rate | 1-3% | ~0.1% (retry) | -97% |
| Skills quality | FIFO (noise) | Quarantine + relevance-ranked | Quality ↑ |
| `core.py` size | 908 lines | 316 lines | -65% |
| Test coverage | 19/29 passing | 64/64 passing | +237% |
| HITL rejection handling | Subtask fails | Reformulation loop | UX ↑ |
| Reasoner availability | No timeout/fallback | 30s timeout + Workhorse fallback | Resilience ↑ |
| Orchestrator decomposition | Over-decomposition (5 subtasks for simple goals) | Intelligent (1 subtask for simple goals) | -80% time |

## Known Limitations

1. **Skills quarantine delay**: New skills require 3+ uses before injection. This means genuinely useful skills take time to become active. Trade-off: context cleanliness vs. speed of learning.

2. **Reasoner fallback quality**: When Reasoner is unavailable, the system falls back to Workhorse with an enhanced prompt. This is better than total failure, but may not match Reasoner's reasoning depth.

3. **HITL reformulation limit**: After `hitl_retry_attempts` reformulation attempts, the subtask fails. In complex scenarios, the agent may need more attempts to find an acceptable action.

4. **Single-session focus**: AgentX is optimized for single-session execution. Multi-session parallel execution is not yet supported.

5. **Judge calibration**: Judge may give lower scores for functionally correct code if it doesn't meet all rubric criteria (e.g., type hints, docstrings). The Judge now considers goal context when evaluating efficiency.

6. **Worker feedback loop**: Workers may change implementation approach when receiving Judge feedback, even if the goal specified a particular approach. The system now includes instructions to preserve original requirements.

## License

[Your license here]

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Format code (`black agent/ tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for semantic search
- [sentence-transformers](https://www.sbert.net/) for embeddings
