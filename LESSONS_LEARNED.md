# Lessons Learned

## 1. Role Bleed / Role Confusion in Judge Agent (FASE 4.5)

**Problem:** Phi-4-mini Judge suffered from role bleed — output garbled text (e.g., "PARACED COM TAREFACAO DE FACTORIAL") instead of structured evaluation. The model sometimes generated code instead of verdicts.

**Root Cause:** `chat_format='chatml'` was hardcoded in the LLM Pool and applied to all models. Phi-4-mini uses a native phi-3 template (`<|system|>`, `<|user|>`, `<|assistant|>` markers), not ChatML (`<|im_start|>`, `<|im_end|>`). Forcing chatml corrupted tokenization.

**Fix:** Added `chat_format: null` in `config.yaml` for phi-4-mini, allowing llama.cpp to use the model's native GGUF template. Combined with:
- System prompt as separate `role: system` message (real role boundary)
- Triple anchoring in user prompt (`### INICIO ###`, `### CRITERIOS ###`, `### FIM ###`)
- Judge temperature reduced to 0.1 (deterministic)

**Lesson:** Each GGUF model has its own `chat_template` embedded in the file. Never force a universal `chat_format` — use `null` unless you've verified the model uses ChatML.

## 2. VRAM Constraints Require Quantization Trade-Offs (FASE 3.5)

**Problem:** Two Q4_K_M models (Worker 9.9 GB + Judge 3.5 GB) would exceed 12 GB VRAM.

**Fix:** Downgrade Worker from Q4_K_M (5.2 GB) to Q3_K_M (3.5 GB). Saved ~1.7 GB.

**Trade-off:** ~15-20% quality loss on the Worker model. Verified empirically — Judge evaluations showed score reduction of ~1-2 points compared to Q4_K_M, but the system fits in VRAM and completes tasks.

**Lesson:** Quantization is a trade-off. Test empirically rather than assuming quality loss is unacceptable. A functioning Q3_K_M is better than a crashing Q4_K_M.

## 3. Self-Enhancement Bias Requires Different Judge Model (FASE 3.5)

**Problem:** Initial design used Qwen as both Worker and Judge. Judge consistently approved Worker outputs even when flawed — the model couldn't criticize its own outputs.

**Fix:** Use Phi-4-mini (different architecture, different training) as Judge. This is a fundamental constraint: **Judge must be a different model than Worker** to avoid self-enhancement bias.

**Lesson:** The bias is structural, not a prompt engineering issue. Even with strong "you are a strict evaluator" instructions, a model cannot reliably judge its own output.

## 4. Parser Robustness Requires Retries (FASE 4.6)

**Problem:** `MAX_PARSE_ATTEMPTS=1` caused sub-tasks to fail on the first parse error. Qwen Q3_K_M has ~10-20% probability of producing malformed ReAct output on any given call.

**Root Cause:** Quantization (Q3_K_M) reduces structural coherence. Malformed ReAct output (missing Action, invalid JSON, stray text) is statistically expected, not anomalous.

**Fix:** Restored `MAX_PARSE_ATTEMPTS=3` with prompt reinforcement on 2nd attempt:
```python
if attempt == 1:
    messages.append(Message(role="user", content=REACT_FORMAT_REMINDER))
```

**Result:** Parser success rate improved from ~80% to 90%+ (9/10 in empirical test).

**Lesson:** LLMs are probabilistic — always allow retries. A single attempt is a single point of failure.

## 5. Context Window Is a Hard Limit (FASE 1.5)

**Problem:** ReAct loops with 10+ steps can exceed 16K tokens. The LLM starts dropping earlier context, causing incoherent reasoning.

**Fix:** Set `n_ctx: 16384` for Worker and `n_ctx: 8192` for Judge. The Agent core tracks prompt length and truncates when approaching the limit.

**Lesson:** Context window is not a suggestion. Design every component (prompt templates, ReAct steps, memory injection) to fit within the limit. When in doubt, measure.

## 6. Less Is More in Knowledge Injection (FASE 6)

**Problem:** ECC (Everything Claude Code) system has 50K+ tokens across rules, agents, and skills. Injecting it all would overflow the context window.

**Fix:** Selective integration — only 5 rule files (< 2000 tokens total) and 5 adapted agent prompts. Rules are chosen per role (coding gets coding-style + python + git, security gets security + python, etc.).

**Lesson:** Knowledge injection must respect the model's context window. Injecting everything degrades performance on the primary task. A focused 2000 tokens of relevant rules outperforms 50K tokens of comprehensive but diluted knowledge.

## Summary Table

| Lesson | Category | Impact | Fix |
|--------|----------|--------|-----|
| 1. chat_format per model | Architecture | Critical (garbled output) | `chat_format: null` for native templates |
| 2. Quantization trade-off | Performance | High (VRAM limit) | Q3_K_M for Worker |
| 3. Different Judge model | Architecture | Critical (bias) | Phi-4-mini as Judge |
| 4. Parser retries | Robustness | High (task failures) | MAX_PARSE_ATTEMPTS=3 |
| 5. Context limits | Design | High (incoherent reasoning) | n_ctx limits + truncation |
| 6. Selective ECC | Design | Medium (context overflow) | < 2000 tokens, per-role |
