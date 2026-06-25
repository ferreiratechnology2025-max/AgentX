# AgentX Usage Guide

Multi-agent system with orchestration, specialized worker roles, and judge evaluation.

## Basic Usage

### Start the server
```bash
python main.py
```
Server starts at http://localhost:8000 — Web UI at http://localhost:8000/

### Execute a simple goal
```bash
curl -N -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "What is 15% of 250?"}'
```

### Check system status
```bash
curl http://localhost:8000/agent/status
```
Returns: tools, memory stats, VRAM usage, loaded models, agent count.

---

## Multi-Agent Orchestration

Execute complex goals that require multiple steps and specialized roles:

```bash
curl -N -X POST http://localhost:8000/orchestrator/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Write a Python factorial function with tests"}'
```

### What happens:
1. **Orchestrator** decomposes goal into sub-tasks
2. **Agent Factory** creates specialized workers for each sub-task
3. **Workers** execute sub-tasks via ReAct loop with appropriate tools
4. **Judge** evaluates each worker output (score 0-10, verdict)
5. **Orchestrator** synthesizes final result

### Orchestrator with custom parameters:
```bash
curl -N -X POST http://localhost:8000/orchestrator/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Implement a login system",
    "max_iterations": 3,
    "context": "Use FastAPI and JWT authentication"
  }'
```

---

## Using Specialized Roles

### Code Review
```bash
curl -N -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Review this Python code for quality issues: def add(a,b): return a+b",
    "system_prompt": "You are a senior code reviewer."
  }'
```

The code_review role automatically injects ECC rules for coding-style and security.

### Security Audit
The security role injects OWASP Top 10 checklist and Python security rules. Injected automatically when Orchestrator assigns the security role.

### TDD Workflow
The tdd role enforces Red-Green-Refactor cycle with edge case coverage. Assigned automatically for tasks involving testing.

### Planning
The planning role guides feature decomposition and implementation ordering. Useful for complex multi-file changes.

### Architecture
The architect role evaluates system design trade-offs. Assign explicitly for architecture decisions.

---

## ECC Integration

ECC (Everything Claude Code) rules are automatically injected into worker system prompts:

| Role | Rules Injected |
|------|----------------|
| coding | coding-style, python, git-workflow |
| security | security, python |
| testing / tdd | testing |
| code_review | coding-style, security |
| general | coding-style |

Rules are < 2000 tokens total. No manual configuration needed.

To verify ECC injection for a role:
```python
from agent.ecc_loader import load_ecc_rules_for_role
rules = load_ecc_rules_for_role("coding")
print(f"{len(rules)} chars of ECC rules loaded")
```

---

## Judge Evaluation

You can also use the Judge independently:

```bash
curl -X POST http://localhost:8000/judge/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a Python factorial function",
    "worker_output": "def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
    "role": "coding"
  }'
```

### Response includes:
- `score` (0-10, normalized)
- `verdict` (APPROVED / NEEDS_REVISION / REJECTED)
- `parse_success` (true if JSON parsed correctly)
- `reasoning` (detailed analysis)
- `criteria_scores` (per-criterion breakdown)
- `feedback` (actionable improvement suggestions)

---

## Monitoring

### VRAM Usage
Check VRAM in server logs or via `/agent/status`:
```json
{
  "vram_usage_gb": 6.8,
  "vram_warning": false,
  "loaded_models": ["qwen2.5-7b-worker", "phi-4-mini-judge"]
}
```

### Judge Performance
Monitor Judge output quality via `parse_success` field:
- `true` → evaluation was parsed correctly
- `false` → model didn't output valid JSON; check `chat_format` in config

### Worker Parse Status
Server logs show `Parse falhou (tentativa N/3)` when Worker produces invalid ReAct. Three failures → task fallback.

---

## Troubleshooting

### "Role Bleed" — Judge outputs garbled text
**Fix:** Verify `chat_format: null` for phi-4-mini in `config.yaml`. This is the most common configuration error.

### "VRAM exceeded 90%" warning in logs
**Fix:** Reduce `n_gpu_layers` in `config.yaml` (try 16 instead of 24). Or switch Judge to a smaller model.

### Worker keeps failing parse
**Check:** `MAX_PARSE_ATTEMPTS` should be 3 (restored in FASE 4.6). Verify `data/knowledge/skills_learnt.md` isn't corrupted.

### Judge returns score=0 repeatedly
**Check:** `parse_success` field. If false, the model isn't outputting valid JSON. Adjust temperature (try 0.1) or check chat_format.

### Orchestrator infinite loop
**Check:** Orchestrator has `parse_retries` limit (max 1) per task. If loops persist, verify Judge is returning parseable output.

### Tool calls failing validation
**Check:** Tool name spelling in `config.yaml`. All tool calls go through Pydantic validation — check error message for schema details.

---

## Advanced Configuration

### Custom system prompts
```bash
curl -N -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Analyze this code",
    "system_prompt": "You focus only on security issues."
  }'
```

### YOLO mode (auto-approve all tools)
Set `yolo_mode: true` in `config.yaml` under `agent`. **Not recommended for production.** Orchestrator still blocks CONFIRM tools autonomously.

### Model selection
The LLM Pool routes by role preference:
- coding/research/general → qwen2.5-7b-worker
- judging/evaluation → phi-4-mini-judge
