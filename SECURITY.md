# Security Policy

## Threat Model

### Agents Can Execute Arbitrary Code
- Worker agents execute tool calls via ReAct loop
- `process_orchestrator` can start/stop system processes
- `project_manager` can read/write files in workspace
- `git_worker` can stage and commit code

### LLM Can Generate Malicious Tool Calls
- Prompt injection via user goal or environment variables
- LLM may hallucinate tool names or arguments
- ECC-injected rules could conflict with safe operation

### Data Exposure
- Secrets in environment variables visible to agent
- Memory system persists sensitive information
- Logs may capture sensitive data

## Mitigations

### Tool Permissions
| Level | Behavior | Tools |
|-------|----------|-------|
| SAFE | Auto-approved | calculator, current_datetime, save_memory |
| CONFIRM | Requires user approval | project_manager, git_worker, process_orchestrator |
| ADMIN | Blocked in autonomous mode | (reserved for future use) |

### Sandbox Filesystem
- All file I/O goes through `project_manager` workspace
- Symlink protection prevents path traversal
- Blocked patterns: `.env`, `private/`, `models/`, `config.yaml`
- Write operations limited to `workspace/` directory

### Input Validation
- **GBNF Grammar**: Forces LLM to output valid JSON for tool calls
- **Pydantic Validation**: All tool arguments validated against schema
- **Command Whitelist**: `process_orchestrator` only accepts 9 allowed runtimes
- **Token Blocklist**: 11 dangerous tokens/commands blocked

### VRAM Protection
- VRAM monitoring warns at > 90% usage
- Lazy loading prevents OOM by design
- Fallback unloads least-used model if VRAM full

### Judge Independence
- Judge uses different model (Phi-4-mini) than Worker (Qwen)
- Prevents self-enhancement bias
- `parse_success` field catches non-compliant evaluations

## Security Checklist

- [ ] All file I/O goes through sandbox
- [ ] All tool calls validated by Pydantic
- [ ] CONFIRM tools require user approval via Web UI
- [ ] Blocked patterns (.env, private/, models/) enforced
- [ ] Symlink attacks prevented
- [ ] VRAM monitored with warning above 90%
- [ ] Judge model != Worker model
- [ ] GBNF grammar forces valid JSON
- [ ] Command whitelist enforced for process execution
- [ ] Memory system uses `workspace/` only

## Incident Response

| Incident | Response |
|----------|----------|
| Unauthorized command execution | Check logs, kill process, review sandbox rules |
| Agent accesses blocked file | Check sandbox rules, add to blocked patterns |
| VRAM exceeds 90% | Monitor logs, reduce n_gpu_layers or model quantization |
| Secrets exposed | Rotate secrets, update .env, remove from git history |
| Judge evaluation fails repeatedly | Check parse_success field, verify chat_format in config |
