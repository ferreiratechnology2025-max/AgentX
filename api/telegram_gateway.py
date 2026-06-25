"""Telegram Bot Gateway — remote execution + HITL approval via Telegram"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes,
)

from agent.core import AutonomousAgent, AgentConfig
from agent.session_manager import save_checkpoint

logger = logging.getLogger(__name__)

ENV_PATH = Path("private/.env")
MAX_MSG_LEN = 3900


class TelegramGateway:
    """Bridge between Telegram chats and AgentX ReAct loop"""

    def __init__(self, llm_manager, tool_registry, agent_config: Optional[AgentConfig] = None):
        self.llm = llm_manager
        self.tool_registry = tool_registry
        self.agent_config = agent_config or AgentConfig()
        self.token: Optional[str] = None
        self.allowed_user_id: Optional[int] = None
        self.application: Optional[Application] = None
        self._sessions: Dict[int, AutonomousAgent] = {}
        self._running = False
        self._load_env()

    def _load_env(self):
        if not ENV_PATH.exists():
            logger.warning("⚠️ private/.env not found — Telegram bot disabled")
            return
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    self.token = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("TELEGRAM_ALLOWED_USER_ID="):
                    raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                    self.allowed_user_id = int(raw) if raw else None

    async def start(self):
        if not self.token or not self.allowed_user_id:
            logger.info("⏸️ Telegram bot: token or user_id missing, skipping")
            return
        app = (
            Application.builder()
            .token(self.token)
            .read_timeout(30)
            .write_timeout(30)
            .build()
        )
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("cancel", self._cmd_cancel))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        app.add_handler(CallbackQueryHandler(self._handle_callback))
        app.add_error_handler(self._error_handler)
        await app.initialize()
        await app.updater.start_polling()
        await app.start()
        self.application = app
        self._running = True
        logger.info("🤖 Telegram bot started — user_id=%s", self.allowed_user_id)

    async def stop(self):
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            self._running = False
            logger.info("🤖 Telegram bot stopped")

    # ── helpers ────────────────────────────────────────────────

    def _check_auth(self, user_id: int) -> bool:
        return user_id == self.allowed_user_id

    async def _send(self, chat_id: int, text: str, **kw):
        """Send message splitting if too long"""
        if not text:
            return
        while len(text) > MAX_MSG_LEN:
            idx = text.rfind("\n", 0, MAX_MSG_LEN)
            if idx < MAX_MSG_LEN // 2:
                idx = MAX_MSG_LEN
            await self.application.bot.send_message(chat_id=chat_id, text=text[:idx], **kw)
            text = text[idx:].strip()
        await self.application.bot.send_message(chat_id=chat_id, text=text, **kw)

    async def _process_events(self, chat_id: int, agent: AutonomousAgent, goal: str):
        """Consumes the ReAct async generator and forwards events to Telegram"""
        buffer = []
        async for event in agent.run(goal, session_id=f"tg_{chat_id}"):
            t = event["type"]
            if t == "thought":
                buffer.append(f"🧠 {event['content']}")
            elif t == "action":
                args = json.dumps(event.get("arguments", {}), separators=(",", ":"), ensure_ascii=False)
                buffer.append(f"🛠️ *{event['tool']}* `{args}`")
                await self._flush_buffer(chat_id, buffer)
            elif t == "observation":
                buffer.append(f"👁️ `{event['content'][:300]}`")
                await self._flush_buffer(chat_id, buffer)
            elif t == "final":
                body = f"✅ *Resposta Final:*\n{event['content']}"
                await self._flush_buffer(chat_id, buffer)
                await self._send(chat_id, body, parse_mode="Markdown")
                save_checkpoint(
                    f"tg_{chat_id}", goal, "completed",
                    len(agent.state.steps), event["content"][:200]
                )
            elif t == "awaiting_approval":
                await self._flush_buffer(chat_id, buffer)
                pending = event["pending"]
                keyboard = [[
                    InlineKeyboardButton("✅ Aprovar", callback_data=f"app:{chat_id}"),
                    InlineKeyboardButton("❌ Rejeitar", callback_data=f"rej:{chat_id}"),
                ]]
                await self._send(
                    chat_id,
                    f"🛡️ *Ação Requer Aprovação:* `{pending['name']}(...)`\nArgs: `{json.dumps(pending['arguments'], separators=(',',':'))}`",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )
            elif t == "error":
                await self._flush_buffer(chat_id, buffer)
                await self._send(chat_id, f"❌ {event['content'][:400]}")

    async def _flush_buffer(self, chat_id: int, buffer: list):
        if not buffer:
            return
        text = "\n\n".join(buffer)
        buffer.clear()
        await self._send(chat_id, text, parse_mode="Markdown")

    # ── handlers ───────────────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update.effective_user.id):
            await update.message.reply_text("⛔ Acesso não autorizado")
            return
        await update.message.reply_text(
            "🤖 *AgentX Telegram Gateway*\n\n"
            "Envie qualquer objetivo para executar o loop ReAct.\n\n"
            "/status — sistema\n"
            "/cancel — aborta execução atual",
            parse_mode="Markdown",
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update.effective_user.id):
            await update.message.reply_text("⛔ Acesso não autorizado")
            return
        tools = self.tool_registry.get_names()
        await update.message.reply_text(
            f"📊 *AgentX Status*\n"
            f"• Ferramentas: {len(tools)} (`{'`, `'.join(tools)}`)\n"
            f"• Sessões ativas: {len(self._sessions)}\n"
            f"• Modelo: {self.llm.model_path}",
            parse_mode="Markdown",
        )

    async def _cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update.effective_user.id):
            await update.message.reply_text("⛔ Acesso não autorizado")
            return
        chat_id = update.effective_chat.id
        if chat_id in self._sessions:
            del self._sessions[chat_id]
            await update.message.reply_text("⏹️ Execução cancelada.")
        else:
            await update.message.reply_text("Nenhuma execução ativa.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not self._check_auth(user.id):
            await update.message.reply_text("⛔ Acesso não autorizado")
            return
        goal = update.message.text.strip()
        chat_id = update.effective_chat.id

        if chat_id in self._sessions:
            await update.message.reply_text("⏳ Já há execução em andamento. Use /cancel para abortar.")
            return

        await update.message.reply_text(f"🎯 Executando: `{goal[:120]}`", parse_mode="Markdown")

        agent = AutonomousAgent(self.llm, self.tool_registry, self.agent_config)
        self._sessions[chat_id] = agent

        try:
            await self._process_events(chat_id, agent, goal)
        except Exception as e:
            await self._send(chat_id, f"❌ Erro: {str(e)[:300]}")
        finally:
            self._sessions.pop(chat_id, None)

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        if ":" not in data:
            return
        action, chat_id_str = data.split(":", 1)
        chat_id = int(chat_id_str)

        agent = self._sessions.get(chat_id)
        if not agent:
            await query.edit_message_text(text="❌ Sessão expirou.")
            return

        approved = action == "app"
        await query.edit_message_reply_markup(reply_markup=None)

        async for event in agent.resume_loop(rejected=not approved):
            t = event["type"]
            try:
                if t == "thought":
                    await self._send(chat_id, f"🧠 {event['content'][:400]}")
                elif t == "action":
                    args = json.dumps(event.get("arguments", {}), separators=(",", ":"), ensure_ascii=False)
                    await self._send(chat_id, f"🛠️ *{event['tool']}* `{args}`", parse_mode="Markdown")
                elif t == "observation":
                    await self._send(chat_id, f"👁️ `{event['content'][:300]}`", parse_mode="Markdown")
                elif t == "final":
                    await self._send(chat_id, f"✅ *Resposta Final:*\n{event['content']}", parse_mode="Markdown")
                elif t == "awaiting_approval":
                    pending = event["pending"]
                    keyboard = [[
                        InlineKeyboardButton("✅ Aprovar", callback_data=f"app:{chat_id}"),
                        InlineKeyboardButton("❌ Rejeitar", callback_data=f"rej:{chat_id}"),
                    ]]
                    await self._send(
                        chat_id,
                        f"🛡️ *Ação Requer Aprovação:* `{pending['name']}(...)`",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                    )
                elif t == "error":
                    await self._send(chat_id, f"❌ {event['content'][:300]}")
            except Exception as e:
                logger.error("Error sending telegram event: %s", e)

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error("Telegram error: %s", context.error)


async def start_telegram_gateway(llm, tool_registry, agent_config=None):
    """Factory: cria e inicia o gateway (usado no startup da API)"""
    gw = TelegramGateway(llm, tool_registry, agent_config)
    await gw.start()
    return gw
