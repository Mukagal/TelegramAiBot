import os
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WEBHOOK_URL    = os.getenv("WEBHOOK_URL", "")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is not set! Copy .env.example → .env and fill it.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set!")

KNOWLEDGE_FILE = Path(__file__).parent.parent / "data" / "company_knowledge.txt"
COMPANY_CONTEXT = KNOWLEDGE_FILE.read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""Ты — AI-ассистент интернет-магазина «Центр Красок #1» (centr-krasok.kz).
Твоя задача — дружелюбно и точно отвечать на вопросы пользователей о компании.

СТРОГИЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе информации из базы знаний ниже.
2. Если информации нет в базе знаний — честно скажи: "К сожалению, у меня нет точных данных по этому вопросу. Пожалуйста, свяжитесь с нами по телефону +7 (778) 061-50-00 или на email info.online@abis.kz."
3. НЕ выдумывай цены, акции, вакансии или другие факты, которых нет в базе.
4. Пиши кратко и по делу. Используй эмодзи умеренно для удобства чтения.
5. Если вопрос не связан с компанией или товарами — вежливо объясни, что ты ассистент магазина красок.
6. Общайся на том языке, на котором написал пользователь (русский, казахский, английский).

===== БАЗА ЗНАНИЙ О КОМПАНИИ =====
{COMPANY_CONTEXT}
===================================
"""

conversation_store: dict[int, list[dict]] = {}
MAX_HISTORY = 10

# ---------------------------------------------------------------------------
# Rate limiting — configure via .env
# ---------------------------------------------------------------------------
USER_MAX_MESSAGES_PER_HOUR = int(os.getenv("USER_MAX_MSG_PER_HOUR", "10"))
GLOBAL_MAX_TOKENS_PER_DAY  = int(os.getenv("GLOBAL_MAX_TOKENS_PER_DAY", "200000"))
AVG_TOKENS_PER_REQUEST     = 1500  # fallback estimate if OpenAI usage missing

_user_message_times: dict[int, list[float]] = defaultdict(list)
_global_token_budget: dict = {"date": "", "tokens": 0}


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


def _check_user_rate(chat_id: int) -> bool:
    """Return True if user is within their hourly message limit."""
    now = time.monotonic()
    window = now - 3600
    times = _user_message_times[chat_id]
    times[:] = [t for t in times if t > window]
    if len(times) >= USER_MAX_MESSAGES_PER_HOUR:
        return False
    times.append(now)
    return True


def _check_global_budget(tokens_used: int = 0) -> bool:
    """Return True if daily token budget is not exhausted.
    Pass tokens_used > 0 to record actual consumption after a call.
    """
    today = _today()
    if _global_token_budget["date"] != today:
        _global_token_budget["date"] = today
        _global_token_budget["tokens"] = 0
    _global_token_budget["tokens"] += tokens_used
    return _global_token_budget["tokens"] < GLOBAL_MAX_TOKENS_PER_DAY

# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if WEBHOOK_URL and TELEGRAM_TOKEN:
        await set_webhook()
    yield


app = FastAPI(title="Центр Красок AI Bot", lifespan=lifespan)


async def set_webhook():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json={"url": f"{WEBHOOK_URL}/webhook"})
        logger.info("Webhook set: %s", r.json())


async def send_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        })


async def send_typing(chat_id: int):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendChatAction"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": chat_id, "action": "typing"})


async def ask_openai(chat_id: int, user_text: str) -> str:
    history = conversation_store.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    if len(history) > MAX_HISTORY * 2:
        history = history[-(MAX_HISTORY * 2):]
        conversation_store[chat_id] = history

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "content-type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": 1024,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        if not resp.is_success:
            logger.error("OpenAI %s: %s", resp.status_code, resp.text)
        resp.raise_for_status()
        data = resp.json()

    assistant_text = data["choices"][0]["message"]["content"]

    # Record actual token usage against the global daily budget
    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", AVG_TOKENS_PER_REQUEST)
    _check_global_budget(tokens_used=total_tokens)
    logger.info("Tokens used this request: %d | Daily total: %d / %d",
                total_tokens, _global_token_budget["tokens"], GLOBAL_MAX_TOKENS_PER_DAY)

    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


@app.get("/")
async def root():
    return {"status": "ok", "bot": "Центр Красок #1 AI Assistant"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/stats")
async def stats():
    """Live view of rate-limit state — protect this endpoint in production."""
    return {
        "global_tokens_today":    _global_token_budget["tokens"],
        "global_token_limit":     GLOBAL_MAX_TOKENS_PER_DAY,
        "global_budget_remaining": max(0, GLOBAL_MAX_TOKENS_PER_DAY - _global_token_budget["tokens"]),
        "date":                   _global_token_budget["date"],
        "user_hourly_limit":      USER_MAX_MESSAGES_PER_HOUR,
        "active_user_sessions":   len(_user_message_times),
    }


@app.post("/webhook")
async def webhook(request: Request):
    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    message = update.get("message") or update.get("edited_message")
    if not message:
        return JSONResponse({"ok": True})

    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip()

    if not text:
        return JSONResponse({"ok": True})

    if text.startswith("/start"):
        await send_message(
            chat_id,
            "👋 Привет! Я AI-ассистент магазина <b>Центр Красок #1</b>.\n\n"
            "Задайте мне любой вопрос о нашей компании, продукции или услугах — "
            "я с радостью помогу! ",
        )
        return JSONResponse({"ok": True})

    if text.startswith("/"):
        return JSONResponse({"ok": True})

    # --- Rate limiting ---
    if not _check_user_rate(chat_id):
        await send_message(
            chat_id,
            f"⏳ Вы отправили слишком много сообщений. "
            f"Лимит: {USER_MAX_MESSAGES_PER_HOUR} сообщений в час.\n"
            "Пожалуйста, попробуйте позже или позвоните нам: +7 (778) 061-50-00"
        )
        return JSONResponse({"ok": True})

    if not _check_global_budget():
        logger.warning("Global daily token budget exhausted!")
        await send_message(
            chat_id,
            "😔 Ассистент временно недоступен — достигнут дневной лимит запросов.\n"
            "Пожалуйста, свяжитесь с нами напрямую: +7 (778) 061-50-00 "
            "или info.online@abis.kz"
        )
        return JSONResponse({"ok": True})
    # --- End rate limiting ---

    await send_typing(chat_id)

    try:
        reply = await ask_openai(chat_id, text)
    except httpx.HTTPStatusError as e:
        logger.error("OpenAI API error: %s", e)
        reply = (
            "Извините, произошла техническая ошибка. "
            "Пожалуйста, попробуйте позже или свяжитесь с нами: "
            "+7 (778) 061-50-00"
        )
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        reply = "Что-то пошло не так. Пожалуйста, попробуйте ещё раз."

    await send_message(chat_id, reply)
    return JSONResponse({"ok": True})


async def poll():
    """Long-polling fallback for local development."""
    offset = 0
    url_base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    logger.info("Starting long-polling...")

    async with httpx.AsyncClient(timeout=35) as client:
        while True:
            try:
                r = await client.get(
                    f"{url_base}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                )
                updates = r.json().get("result", [])
                for upd in updates:
                    offset = upd["update_id"] + 1
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg:
                        continue
                    chat_id = msg["chat"]["id"]
                    text = msg.get("text", "").strip()
                    if not text:
                        continue
                    if text.startswith("/start"):
                        await send_message(
                            chat_id,
                            "👋 Привет! Я AI-ассистент магазина <b>Центр Красок #1</b>.\n\n"
                            "Задайте мне любой вопрос о нашей компании, продукции или услугах — "
                            "я с радостью помогу! 🎨",
                        )
                        continue
                    if text.startswith("/"):
                        continue
                    await send_typing(chat_id)
                    # --- Rate limiting ---
                    if not _check_user_rate(chat_id):
                        await send_message(
                            chat_id,
                            f"⏳ Вы отправили слишком много сообщений. "
                            f"Лимит: {USER_MAX_MESSAGES_PER_HOUR} сообщений в час.\n"
                            "Пожалуйста, попробуйте позже или позвоните нам: +7 (778) 061-50-00"
                        )
                        continue
                    if not _check_global_budget():
                        logger.warning("Global daily token budget exhausted!")
                        await send_message(
                            chat_id,
                            "😔 Ассистент временно недоступен — достигнут дневной лимит запросов.\n"
                            "Пожалуйста, свяжитесь с нами напрямую: +7 (778) 061-50-00 "
                            "или info.online@abis.kz"
                        )
                        continue
                    # --- End rate limiting ---
                    try:
                        reply = await ask_openai(chat_id, text)
                    except Exception as e:
                        logger.error("Error: %s", e)
                        reply = "Ошибка. Попробуйте позже или позвоните: +7 (778) 061-50-00"
                    await send_message(chat_id, reply)
            except Exception as e:
                logger.error("Polling error: %s", e)
                import asyncio; await asyncio.sleep(3)


if __name__ == "__main__":
    import asyncio
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "poll"
    if mode == "poll":
        asyncio.run(poll())
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)