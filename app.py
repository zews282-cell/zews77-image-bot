import base64
import io
import logging
import os
import time
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, Request
from openai import AsyncOpenAI

# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zews77")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Экономичная модель для текста, анализа фото и OCR
OPENAI_TEXT_MODEL = os.getenv(
    "OPENAI_TEXT_MODEL",
    "gpt-5.6-luna"
)

# Модель для генерации и редактирования изображений
OPENAI_IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-2"
)

PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")

# Сколько последних сообщений помнить в одном чате
MAX_HISTORY_MESSAGES = int(
    os.getenv("MAX_HISTORY_MESSAGES", "12")
)

# Минимальный интервал между запросами одного пользователя
RATE_LIMIT_SECONDS = float(
    os.getenv("RATE_LIMIT_SECONDS", "2")
)

# ============================================================
# APP
# ============================================================

app = FastAPI(title="Zews77 AI Assistant")

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)

telegram_base = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
)

# Память диалогов.
# Важно: после перезапуска сервера она очищается.
chat_history = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)

last_request_time = {}


# ============================================================
# TELEGRAM
# ============================================================

async def telegram(method: str, **kwargs):
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{telegram_base}/{method}",
            data=kwargs
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("ok"):
            raise RuntimeError(
                f"Telegram API error: {result}"
            )

        return result


async def download_telegram_file(
    file_id: str
) -> bytes:

    info = await telegram(
        "getFile",
        file_id=file_id
    )

    file_path = info["result"]["file_path"]

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(
            f"https://api.telegram.org/file/bot"
            f"{TELEGRAM_BOT_TOKEN}/{file_path}"
        )

        response.raise_for_status()

        return response.content


# ============================================================
# RATE LIMIT
# ============================================================

def is_rate_limited(chat_id: int) -> bool:

    now = time.monotonic()

    previous = last_request_time.get(chat_id)

    if previous is not None:
        if now - previous < RATE_LIMIT_SECONDS:
            return True

    last_request_time[chat_id] = now

    return False


# ============================================================
# OPENAI — TEXT
# ============================================================

async def ask_ai(
    chat_id: int,
    text: str
) -> str:

    history = list(chat_history[chat_id])

    response = await openai_client.responses.create(
        model=OPENAI_TEXT_MODEL,

        instructions=(
            "Ты полезный AI-ассистент в Telegram. "
            "Отвечай на русском языке, если пользователь пишет "
            "на русском. Будь понятным, практичным и дружелюбным. "
            "Если пользователь просит выполнить задачу, помоги "
            "выполнить её максимально непосредственно. "
            "Не используй Markdown, если он может быть проблемой "
            "для Telegram."
        ),

        input=history + [
            {
                "role": "user",
                "content": text
            }
        ],
    )

    answer = response.output_text.strip()

    if not answer:
        answer = "Не удалось получить ответ."

    # Сохраняем контекст
    chat_history[chat_id].append(
        {
            "role": "user",
            "content": text
        }
    )

    chat_history[chat_id].append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer


# ============================================================
# OPENAI — IMAGE ANALYSIS / OCR
# ============================================================

async def analyze_image(
    chat_id: int,
    image_bytes: bytes,
    instruction: str
) -> str:

    image_b64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    # Telegram может прислать JPEG/PNG/WebP.
    # Для большинства фотографий JPEG — наиболее вероятный вариант.
    image_url = (
        f"data:image/jpeg;base64,{image_b64}"
    )

    if instruction:
        user_text = instruction
    else:
        user_text = (
            "Проанализируй это изображение подробно. "
            "Опиши, что на нём изображено. "
            "Если на изображении есть текст, распознай его "
            "и перепиши максимально точно."
        )

    response = await openai_client.responses.create(
        model=OPENAI_TEXT_MODEL,

        instructions=(
            "Ты умеешь анализировать изображения. "
            "Внимательно рассматривай фотографию. "
            "Если пользователь просит распознать текст, "
            "делай OCR максимально точно. "
            "Не выдумывай детали, которых не видно."
        ),

        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_text
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                ]
            }
        ],
    )

    answer = response.output_text.strip()

    if not answer:
        answer = "Не удалось проанализировать изображение."

    # Сохраняем только текст, а не огромное изображение.
    chat_history[chat_id].append(
        {
            "role": "user",
            "content": (
                f"[Пользователь отправил изображение] "
                f"{user_text}"
            )
        }
    )

    chat_history[chat_id].append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer


# ============================================================
# OPENAI — IMAGE EDIT
# ============================================================

async def edit_image(
    image_bytes: bytes,
    instruction: str
) -> bytes:

    result = await openai_client.images.edit(
        model=OPENAI_IMAGE_MODEL,

        image=io.BytesIO(image_bytes),

        prompt=(
            "Edit the provided image according to the user's "
            "instruction.\n\n"
            "Preserve the original subject, composition, "
            "proportions and important details unless the user "
            "explicitly asks to change them.\n\n"
            "Make only the requested changes.\n\n"
            f"User instruction: {instruction}"
        ),
    )

    if not result.data:
        raise RuntimeError(
            "OpenAI returned no image"
        )

    item = result.data[0]

    image_b64 = getattr(
        item,
        "b64_json",
        None
    )

    if not image_b64:
        raise RuntimeError(
            "OpenAI did not return image data"
        )

    return base64.b64decode(image_b64)


# ============================================================
# TELEGRAM MESSAGES
# ============================================================

async def send_text(
    chat_id: int,
    text: str
):

    # Telegram имеет ограничение около 4096 символов.
    max_length = 4000

    if len(text) <= max_length:
        await telegram(
            "sendMessage",
            chat_id=chat_id,
            text=text
        )
        return

    # Разбиваем длинный ответ
    for start in range(
        0,
        len(text),
        max_length
    ):
        chunk = text[
            start:start + max_length
        ]

        await telegram(
            "sendMessage",
            chat_id=chat_id,
            text=chunk
        )


async def send_typing(
    chat_id: int
):

    try:
        await telegram(
            "sendChatAction",
            chat_id=chat_id,
            action="typing"
        )
    except Exception:
        pass


# ============================================================
# COMMANDS
# ============================================================

async def handle_command(
    chat_id: int,
    text: str
) -> bool:

    command = text.split()[0].lower()

    if command == "/start":

        await send_text(
            chat_id,
            (
                "Привет! 👋\n\n"
                "Я твой AI-ассистент.\n\n"
                "Я умею:\n"
                "• отвечать на вопросы;\n"
                "• вести диалог и помнить контекст;\n"
                "• анализировать фотографии;\n"
                "• распознавать текст на фото (OCR);\n"
                "• редактировать изображения;\n"
                "• менять цвета, фон, предметы и другие элементы "
                "по твоей инструкции.\n\n"
                "Просто напиши сообщение или отправь фотографию."
            )
        )

        return True

    if command == "/help":

        await send_text(
            chat_id,
            (
                "Как пользоваться ботом:\n\n"
                "💬 Напиши обычный вопрос — я отвечу.\n\n"
                "📷 Отправь фото без подписи — я его "
                "проанализирую.\n\n"
                "🔎 Отправь фото и напиши вопрос в подписи — "
                "я отвечу по содержимому фотографии.\n\n"
                "✏️ Чтобы изменить фото, отправь его с "
                "инструкцией, например:\n"
                "«Сделай косметичку нежно-фиолетовой».\n\n"
                "/reset — очистить контекст диалога."
            )
        )

        return True

    if command == "/reset":

        chat_history.pop(
            chat_id,
            None
        )

        await send_text(
            chat_id,
            "Контекст диалога очищен 🧹"
        )

        return True

    return False


# ============================================================
# DETECT IMAGE EDIT REQUEST
# ============================================================

def looks_like_edit_request(
    instruction: str
) -> bool:

    text = instruction.lower()

    edit_words = [
        "сделай",
        "измени",
        "замени",
        "поменяй",
        "убери",
        "добавь",
        "удали",
        "перекрась",
        "покрась",
        "отретушируй",
        "отредактируй",
        "измени цвет",
        "замени цвет",
        "изменить цвет",
        "заменить цвет",
        "фон",
        "цвет",
        "надпись",
        "текст на изображении",
        "косметичк",
        "упаковк",
        "стикер",
        "логотип",
    ]

    return any(
        word in text
        for word in edit_words
    )


# ============================================================
# PROCESS MESSAGE
# ============================================================

async def process_message(
    message: dict
):

    chat = message.get(
        "chat",
        {}
    )

    chat_id = chat.get("id")

    if not chat_id:
        return

    text = (
        message.get("text")
        or ""
    ).strip()

    # --------------------------------------------------------
    # COMMANDS
    # --------------------------------------------------------

    if text.startswith("/"):

        handled = await handle_command(
            chat_id,
            text
        )

        if handled:
            return

    # --------------------------------------------------------
    # RATE LIMIT
    # --------------------------------------------------------

    if is_rate_limited(chat_id):

        await send_text(
            chat_id,
            "Слишком быстро 🙂 Подожди пару секунд."
        )

        return

    # --------------------------------------------------------
    # PHOTO
    # --------------------------------------------------------

    photo = message.get("photo")

    if photo:

        instruction = (
            message.get("caption")
            or ""
        ).strip()

        await send_typing(chat_id)

        try:

            # Берём изображение максимального доступного размера
            file_id = photo[-1]["file_id"]

            image_bytes = (
                await download_telegram_file(
                    file_id
                )
            )

            # ------------------------------------------------
            # EDIT IMAGE
            # ------------------------------------------------

            if instruction and looks_like_edit_request(
                instruction
            ):

                await telegram(
                    "sendChatAction",
                    chat_id=chat_id,
                    action="upload_photo"
                )

                edited = await edit_image(
                    image_bytes,
                    instruction
                )

                files = {
                    "photo": (
                        "edited.png",
                        edited,
                        "image/png"
                    )
                }

                data = {
                    "chat_id": str(chat_id),
                    "caption": "Готово ✨"
                }

                async with httpx.AsyncClient(
                    timeout=180
                ) as client:

                    response = await client.post(
                        f"{telegram_base}/sendPhoto",
                        data=data,
                        files=files
                    )

                    response.raise_for_status()

                return

            # ------------------------------------------------
            # ANALYZE IMAGE
            # ------------------------------------------------

            answer = await analyze_image(
                chat_id,
                image_bytes,
                instruction
            )

            await send_text(
                chat_id,
                answer
            )

            return

        except Exception:

            logger.exception(
                "Image processing failed"
            )

            await send_text(
                chat_id,
                (
                    "Не получилось обработать изображение 😔\n\n"
                    "Попробуй ещё раз или сформулируй инструкцию "
                    "немного иначе."
                )
            )

            return

    # --------------------------------------------------------
    # ORDINARY TEXT
    # --------------------------------------------------------

    if text:

        await send_typing(chat_id)

        try:

            answer = await ask_ai(
                chat_id,
                text
            )

            await send_text(
                chat_id,
                answer
            )

        except Exception:

            logger.exception(
                "Text processing failed"
            )

            await send_text(
                chat_id,
                (
                    "Не получилось получить ответ от AI 😔\n"
                    "Попробуй ещё раз через несколько секунд."
                )
            )

        return

    # --------------------------------------------------------
    # OTHER MESSAGE TYPES
    # --------------------------------------------------------

    await send_text(
        chat_id,
        (
            "Я умею работать с текстом и фотографиями.\n\n"
            "Отправь мне сообщение или изображение 🙂"
        )
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def health():

    return {
        "status": "ok",
        "bot": "Zews77 AI Assistant"
    }


# ============================================================
# TELEGRAM WEBHOOK
# ============================================================

@app.post("/telegram/webhook")
async def webhook(
    request: Request
):

    update = await request.json()

    message = update.get("message")

    if message:

        await process_message(
            message
        )

    return {
        "ok": True
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():

    if not PUBLIC_URL:

        logger.warning(
            "PUBLIC_URL is not configured. "
            "Telegram webhook was not changed."
        )

        return

    webhook_url = (
        f"{PUBLIC_URL}/telegram/webhook"
    )

    try:

        await telegram(
            "setWebhook",
            url=webhook_url
        )

        logger.info(
            "Telegram webhook configured: %s",
            webhook_url
        )

    except Exception:

        logger.exception(
            "Could not configure Telegram webhook"
        )
