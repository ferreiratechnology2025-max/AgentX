"""LLM Manager with Ollama OpenAI-compatible endpoint"""

import asyncio
import json
import re
from typing import List, Dict, Optional, AsyncGenerator, Any, Tuple
from dataclasses import dataclass
import requests

from pydantic import ValidationError
from tools.schemas import ReActOutput, ReActToolCall


@dataclass
class ToolCall:
    """Representa uma chamada de ferramenta"""
    name: str
    arguments: Dict[str, Any]
    id: str


@dataclass
class Message:
    """Mensagem para o modelo"""
    role: str  # system, user, assistant, tool
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class GenerationFailedError(Exception):
    """Levantada quando generate_with_validation não consegue produzir ReActOutput válido.
    Com schema nativo, isso indica erro de rede/timeout, não falha de formato."""


class LLMManager:
    """Gerencia LLM via Ollama endpoint OpenAI-compatible"""

    MAX_PARSE_ATTEMPTS = 3   # Retry para falhas de parse (rede/timeout pode corromper resposta)
    MAX_HTTP_RETRIES = 3     # Retry para erros transitórios de rede/timeout antes de declarar falha
    HTTP_RETRY_BACKOFF = [0, 1, 3]  # segundos entre tentativas (0, 1s, 3s)
    PARSE_RETRY_BACKOFF = [0.5, 1.5, 3.0]  # backoff entre parse attempts (0.5s, 1.5s, 3s)

    # Schema flat: nested Optional[object] não sobrevive bem em constrained decoding.
    # action_name vazio ("") = final_answer; não-vazio = tool call.
    # Usado como fallback quando tools=[]; normalmente usa _build_react_schema(tools).
    REACT_SCHEMA = {
        "type": "object",
        "required": ["thought", "action_name", "action_arguments", "final_answer"],
        "properties": {
            "thought": {"type": "string"},
            "action_name": {"type": "string"},
            "action_arguments": {"type": "object"},
            "final_answer": {"type": "string"},
        },
    }

    @staticmethod
    def _build_react_schema(tools: List[Dict]) -> dict:
        """Gera REACT_SCHEMA com action_name restrito ao enum de tools disponíveis.

        "" incluído no enum = modelo sinaliza final_answer (sem tool call).
        Sem tools: string livre (fallback para o schema estático).
        """
        tool_names = [t["name"] for t in tools if "name" in t]
        action_name_field = (
            {"type": "string", "enum": [""] + tool_names}
            if tool_names
            else {"type": "string"}
        )
        return {
            "type": "object",
            "required": ["thought", "action_name", "action_arguments", "final_answer"],
            "properties": {
                "thought": {"type": "string"},
                "action_name": action_name_field,
                "action_arguments": {"type": "object"},
                "final_answer": {"type": "string"},
            },
        }

    def __init__(self, config: dict):
        self.config = config
        self.model_id = config['llm'].get('model_id')
        self.num_ctx: Optional[int] = config['llm'].get('num_ctx')  # None = default do modelo
        self.base_url = config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        self.endpoint = f"{self.base_url}/api/chat"

        ctx_info = f", num_ctx={self.num_ctx}" if self.num_ctx else ""
        print(f"LLM Ollama conectado: {self.model_id}{ctx_info}")
        print(f"Endpoint: {self.endpoint}")

    def _validate_output(self, text: str) -> Optional[ReActOutput]:
        """Valida output do LLM: tenta schema flat → schema aninhado → regex."""
        text = text.strip()
        if not text:
            return None

        # 1. Schema flat (formato nativo Fase 3: action_name / action_arguments / final_answer)
        try:
            data = json.loads(text)
            thought = data.get("thought", "")
            action_name = data.get("action_name", "")
            final_answer = data.get("final_answer", "")

            if action_name and not final_answer:
                return ReActOutput(
                    thought=thought,
                    action=ReActToolCall(
                        name=action_name,
                        arguments=data.get("action_arguments") or {}
                    )
                )
            if final_answer:
                return ReActOutput(thought=thought, final_answer=final_answer)
            if action_name:
                return ReActOutput(
                    thought=thought,
                    action=ReActToolCall(
                        name=action_name,
                        arguments=data.get("action_arguments") or {}
                    )
                )
            if thought:
                return ReActOutput(thought=thought, final_answer=thought)
        except (json.JSONDecodeError, ValidationError):
            pass

        # 2. Schema aninhado legado: {"thought": ..., "action": {"name":..., "arguments":...}}
        try:
            data = json.loads(text)
            validated = ReActOutput(**data)
            return validated
        except (json.JSONDecodeError, ValidationError):
            pass

        # 3. Fallback regex (formato ReAct texto clássico)
        try:
            thought_text = text
            thought_match = re.search(
                r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|\nObservation:|$)",
                text, re.DOTALL | re.IGNORECASE
            )
            if thought_match:
                thought_text = thought_match.group(1).strip()

            fa_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
            if fa_match:
                return ReActOutput(thought=thought_text, final_answer=fa_match.group(1).strip())

            action_match = re.search(
                r"Action:\s*([^\n]+)\s*Action Input:\s*(.*?)(?=Observation:|Final Answer:|Action:|$)",
                text, re.DOTALL | re.IGNORECASE
            )
            if action_match:
                tool_name = action_match.group(1).strip()
                json_str = re.sub(r'```json\n?', '', action_match.group(2).strip()).strip()
                args = json.loads(json_str)
                return ReActOutput(
                    thought=thought_text,
                    action=ReActToolCall(name=tool_name, arguments=args)
                )

            if "Thought:" in text:
                return ReActOutput(thought=thought_text, final_answer=thought_text)
        except (json.JSONDecodeError, Exception):
            pass

        return None

    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Extrai chamadas de ferramentas usando regex robusto"""
        tool_calls = []
        
        # Regex: captura Action + Action Input, para no proximo Action/Observation/Final ou fim
        # O JSON e capturado por balanceamento de chaves {}
        pattern = r"Action:\s*([^\n]+)\s*Action Input:\s*(\{.*?\})(?=\s*\n(?:Action:|Observation:|Final Answer:)|\s*$)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            tool_name = match[0].strip()
            json_str = match[1].strip()
            
            try:
                params = json.loads(json_str)
                tool_calls.append(ToolCall(
                    name=tool_name,
                    arguments=params,
                    id=f"call_{len(tool_calls)}"
                ))
            except json.JSONDecodeError as e:
                print(f"JSON parse error for {tool_name}: {e}")
                try:
                    if not json_str.endswith('}'):
                        json_str += '}'
                        params = json.loads(json_str)
                        tool_calls.append(ToolCall(
                            name=tool_name,
                            arguments=params,
                            id=f"call_{len(tool_calls)}"
                        ))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    def format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Formata ferramentas para o system prompt"""
        if not tools:
            return ""
        
        tools_desc = []
        tool_names = []
        
        for tool in tools:
            tools_desc.append(f"""
{tool['name']}: {tool['description']}
Parameters: {json.dumps(tool['parameters'], indent=2)}""")
            tool_names.append(tool['name'])
        
        template = """You are a helpful AI assistant with access to these tools:

{tools_description}

Use the following format:
Thought: your reasoning about what to do
Action: tool name (must be one of [{tool_names}])
Action Input: JSON object with parameters
Observation: result from tool
... (repeat Thought/Action/Observation as needed)
Final Answer: your final response to the user

Important: Always output valid JSON for Action Input. Never invent observations.
Output MUST contain exactly one of: Action/Action Input pair OR a Final Answer."""
        
        return template.format(
            tools_description="\n".join(tools_desc),
            tool_names=", ".join(tool_names)
        )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, dict]:
        """Geracao simples sem tools — retorna (texto, usage)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def sync_generate():
            options: dict = {"temperature": temperature, "num_predict": max_tokens}
            if self.num_ctx:
                options["num_ctx"] = self.num_ctx
            if stop:
                options["stop"] = stop
            payload = {
                "model": self.model_id,
                "messages": messages,
                "stream": False,
                "options": options,
            }
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()

        response = await asyncio.to_thread(sync_generate)
        content = response["message"]["content"]
        usage = {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0)
        }
        return content, usage
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Streaming de tokens"""
        messages = [{"role": "user", "content": prompt}]

        def sync_stream():
            payload = {
                "model": self.model_id,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            response = requests.post(self.endpoint, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            return response

        response = await asyncio.to_thread(sync_stream)
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
            except json.JSONDecodeError:
                pass
    
    async def generate_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Tuple[str, dict]:
        """Geração com tool calling — usa JSON schema nativo do Ollama.

        Faz até MAX_HTTP_RETRIES tentativas com backoff para erros de rede/timeout.
        Erros HTTP persistentes levantam GenerationFailedError.
        """
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        schema = self._build_react_schema(tools)

        def sync_generate():
            options: dict = {"temperature": temperature, "num_predict": max_tokens}
            if self.num_ctx:
                options["num_ctx"] = self.num_ctx
            payload = {
                "model": self.model_id,
                "messages": formatted_messages,
                "stream": False,
                "format": schema,
                "options": options,
            }
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()

        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(self.MAX_HTTP_RETRIES):
            if attempt > 0:
                await asyncio.sleep(self.HTTP_RETRY_BACKOFF[attempt])
            try:
                response = await asyncio.to_thread(sync_generate)
                content = response["message"]["content"]
                usage = {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0)
                }
                return content, usage
            except requests.exceptions.HTTPError as exc:
                # 4xx = erro permanente do cliente (schema/payload inválido) — sobe imediato, sem retry
                if exc.response is not None and exc.response.status_code < 500:
                    raise GenerationFailedError(
                        f"HTTP {exc.response.status_code} (permanente, não retryable): {exc}"
                    )
                last_exc = exc
                print(f"HTTP attempt {attempt + 1}/{self.MAX_HTTP_RETRIES} failed (5xx): {exc}")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                print(f"HTTP attempt {attempt + 1}/{self.MAX_HTTP_RETRIES} failed (rede/timeout): {exc}")

        raise GenerationFailedError(
            f"Ollama unreachable após {self.MAX_HTTP_RETRIES} tentativas: {last_exc}"
        )

    async def generate_with_validation(
        self,
        messages: List[Message],
        tools: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Tuple[ReActOutput, dict]:
        """Gera e valida — retry com backoff se parse falhar.

        Faz ate MAX_PARSE_ATTEMPTS tentativas com backoff progressivo.
        Levanta GenerationFailedError se todas falharem.
        """
        last_content = ""
        for attempt in range(self.MAX_PARSE_ATTEMPTS):
            if attempt > 0:
                backoff = self.PARSE_RETRY_BACKOFF[min(attempt, len(self.PARSE_RETRY_BACKOFF) - 1)]
                await asyncio.sleep(backoff)

            content, usage = await self.generate_with_tools(
                messages, tools, max_tokens=max_tokens, temperature=temperature
            )
            validated = self._validate_output(content)
            if validated is not None:
                return validated, usage

            msg = f"Parse falhou (tentativa {attempt+1}/{self.MAX_PARSE_ATTEMPTS})"
            print(msg)
            last_content = content

            if attempt < self.MAX_PARSE_ATTEMPTS - 1:
                messages.append(Message(
                    role="user",
                    content=(
                        "Sua resposta anterior nao seguiu o formato esperado. "
                        "Responda usando EXATAMENTE este formato JSON:\n"
                        '{"thought": "seu raciocinio", "action_name": "nome_da_ferramenta", '
                        '"action_arguments": {}, "final_answer": ""}\n'
                        "OU se for a resposta final:\n"
                        '{"thought": "seu raciocinio", "action_name": "", '
                        '"action_arguments": {}, "final_answer": "sua resposta final"}'
                    )
                ))

        raise GenerationFailedError(
            f"Parse falhou apos {self.MAX_PARSE_ATTEMPTS} tentativas. "
            f"Ultimo conteudo: {last_content[:200]!r}"
        )
