# Fase 1: Substituir LLM Pool Manager por runtime Ollama

**Brief para Claude Code**. Baseado em `agentx-router-workhorse-reasoner-plano.md`.

Implementar APENAS Fase 1: trocar infraestrutura de runtime (llama-cpp-python direto → Ollama), mantendo assinatura pública intacta. Sem mudança de lógica de agente. Testável isoladamente.

---

## Estado atual

- **Runtime**: llama-cpp-python (`llama_cpp.Llama`)
- **Gerenciamento**: custom pool em `llm/pool.py` (lazy loading, VRAM tracking, fallback)
- **Modelos**: 2 globais em `config.yaml` (qwen2.5-7b worker + phi-4-mini judge)
- **Output estruturado**: GBNF grammar + regex parser (`_validate_output`) + 3 retries

### Arquivos atuais

```
llm/
  ├── __init__.py
  ├── pool.py          ← LLMPoolManager (lazy loading, VRAM tracking, role routing)
  ├── manager.py       ← LLMManager (llama_cpp.Llama wrapper, grammar, parsing)
```

### Assinatura pública (NÃO pode quebrar)

Chamadas que **existem no código atual** e **devem continuar funcionando**:

```python
# Em agent/orchestrator.py, agent/factory.py, agent/judge.py:
from llm.pool import get_llm_pool

pool = get_llm_pool()
llm = await pool.get_model(model_id=..., role=...)  # Retorna objeto com métodos async

# Métodos da classe retornada (presentes em LLMManager hoje):
await llm.generate(prompt, max_tokens, temperature, stop, system_prompt)
await llm.generate_with_tools(messages, tools, max_tokens, temperature)
await llm.generate_with_validation(messages, tools, max_tokens, temperature)

# Retorno esperado:
# generate*, generate_with_validation => (str | ReActOutput | None, dict com 'usage')
```

Locais onde é chamado:
- `agent/orchestrator.py:228, :295` → `pool.get_model(role="general")`
- `agent/factory.py:99` → `pool.get_model(model_id=..., role=...)`
- `agent/judge.py:77` → `pool.get_model(model_id=..., role="judging")`
- `agent/core.py:46` → recebe `LLMManager` como argumento (tipo esperado)

---

## Estado alvo (Fase 1)

- **Runtime**: Ollama (OpenAI-compatible HTTP client)
- **Gerenciamento**: HTTP requests para `http://localhost:11434/v1/chat/completions`
- **Modelos**: Ainda 2 global (mas carregados no Ollama, não em llm/pool.py)
- **Output estruturado**: JSON schema nativo do Ollama (`format` param no payload)

### O que muda

1. `llm/manager.py` → Client fino que fala HTTP com Ollama
2. `llm/pool.py` → Simplifica: remove tracking VRAM custom, n_gpu_layers, paths .gguf absolutos
   - Mantém roteamento por role (pode ficar mais simples)
   - Carrega sempre (ou lazy-load na app startup)
3. `config.yaml`:
   - Remove `llm.n_gpu_layers`, `llm.rope_scaling`, `llm.n_threads`
   - Mantém `llm_pool.models[].id` e `role_preference`
   - Adiciona `ollama_base_url` (default: `http://localhost:11434`)

### O que NÃO muda

- Assinatura de `get_llm_pool()` → retorna pool
- Assinatura de `pool.get_model(...)` → retorna objeto com métodos async
- Métodos públicos: `generate()`, `generate_with_tools()`, `generate_with_validation()`
- Retorno: `(str|ReActOutput|None, usage_dict)`

---

## Arquivos a modificar

| Arquivo | Escopo | Notas |
|---------|--------|-------|
| `llm/manager.py` | **Reescrever** | De `Llama` wrapper → Ollama HTTP client |
| `llm/pool.py` | **Simplificar** | Remove VRAM tracking, n_gpu_layers; mantém pool e routing |
| `config.yaml` | **Update** | Remove llm.n_gpu_layers/rope_scaling; adiciona ollama_base_url |
| `llm/__init__.py` | **Sem mudança** | Se estiver vazio, deixar assim |
| **Outros arquivos** | **Sem mudança** | orchestrator, factory, judge, core — nenhuma mudança |

---

## Implementação (sequência recomendada)

### 1. Ollama API Contract
Usar endpoint OpenAI-compatible:
```python
POST http://localhost:11434/v1/chat/completions
{
  "model": "model_id",
  "messages": [{"role": "system"|"user"|"assistant", "content": "..."}],
  "temperature": 0.7,
  "max_tokens": 256,
  "format": "json" | null  # JSON schema validation nativo
}
```

Response:
```json
{
  "choices": [{"message": {"content": "..."}}],
  "usage": {"prompt_tokens": N, "completion_tokens": N}
}
```

### 2. Reescrever `llm/manager.py`
- Remove: Llama init, grammar, rope_scaling, n_gpu_layers, torch CUDA check
- Adiciona: `requests` HTTP client, `json.dumps(format_schema)` se necessário
- Mantém:
  - `async def generate(...)`
  - `async def generate_with_tools(...)`
  - `async def generate_with_validation(...)` (pode simplificar: schema nativo elimina retry)
  - `_validate_output()` (pode ser minimizado se usar `format="json"`)

### 3. Simplificar `llm/pool.py`
- Remove: `_get_current_vram_usage()`, `_check_vram_warning()`, `_find_fallback_model()`, `_unload_least_used()`, `unload_model()`, `unload_all()`
- Remove: `n_gpu_layers` de ModelConfig
- Simplifica: `get_model()` → sem lógica de VRAM, apenas roteamento + retorna LLMManager já construído
- Mantém: `_route_by_role()`, `ModelConfig` (menos campos)

### 4. Atualizar `config.yaml`
```yaml
llm:
  model_path: null  # Ollama não precisa disso
  # Remove: n_gpu_layers, n_threads, rope_scaling

ollama:
  base_url: "http://localhost:11434"
  
llm_pool:
  max_vram_gb: null  # Ou remove
  models:
    - id: "qwen2.5-7b-worker"
      # Remove: n_gpu_layers, path → virou "id" local do Ollama
      role_preference: [...]
    - id: "phi-4-mini-judge"
      role_preference: [...]
```

---

## Validação (Fase 1 testável isoladamente)

Após implementar, validar em `tests/` ou manualmente:

1. **Ollama está rodando**: `curl http://localhost:11434/v1/models`
2. **Pool carrega modelos**:
   ```python
   pool = get_llm_pool()
   llm = await pool.get_model(role="general")
   ```
3. **Generate retorna resposta**:
   ```python
   text, usage = await llm.generate("Olá", max_tokens=10)
   assert isinstance(text, str) and len(text) > 0
   assert "prompt_tokens" in usage
   ```
4. **Generate_with_validation retorna ReActOutput parseado**:
   ```python
   output, usage = await llm.generate_with_validation(
       messages=[...], tools=[...]
   )
   assert isinstance(output, (ReActOutput, type(None)))
   ```
5. **VRAM**: Checar `nvidia-smi` enquanto roda (Ollama + app)
   - Esperado: Worker + Judge residentes, nada de fallback/unload/retry

### Métricas a capturar
- **Taxa de acerto de formato**: % de responses com `_validate_output() != None` (antes/depois)
- **Latência**: tempo de primeira resposta (geração inicial)
- **Tokens/seg**: throughput de tokens
- **VRAM**: pico simultâneo (Worker + Judge apenas)

---

## Por que Fase 1 isolada é segura

- Não toca em logic de agente (orchestrator, factory, judge, core)
- Assinatura pública preservada → resto do código não quebra
- Testável sem Fase 2+ (não precisa de tiering)
- Se der problema, rollback é só reverter llm/ e config
- Fase 2 (reatribuição de roles) já fica pronta pra depois, só precisa mudar config

---

## Próximas fases (já mapeadas, não fazer agora)

- **Fase 2**: Reatribuir papéis por tier (config only)
- **Fase 3**: JSON schema nativo (reduz MAX_PARSE_ATTEMPTS de 3 → 1)
- **Fase 4+**: Prompt engineering, votação, etc.
