"""
Telegram entrypoint. Deliberately dumb: it only knows how to turn a Telegram
message (text / voice / image) into content for ai.get_response(), and send
the reply back. All product logic lives in ai.py.

No slash commands, no inline buttons, no menus — per the brief, every
message type is handled the same conversational way.
"""
import os
import base64
import logging
import tempfile
from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

import db
import ai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("atlas-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = str(update.effective_user.id)
    db.get_or_create_user(telegram_id, name=update.effective_user.first_name or "")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    reply = ai.get_response(telegram_id, update.message.text)
    await update.message.reply_text(reply)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = str(update.effective_user.id)
    db.get_or_create_user(telegram_id, name=update.effective_user.first_name or "")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if not os.getenv("GROQ_API_KEY"):
        await update.message.reply_text(
            "Voice messages need a GROQ_API_KEY set for transcription — text works in the meantime!"
        )
        return

    voice_file = await update.message.voice.get_file()
    local_path = os.path.join(tempfile.gettempdir(), f"{update.message.voice.file_unique_id}.ogg")
    await voice_file.download_to_drive(local_path)

    try:
        from openai import OpenAI
        # Groq's Whisper endpoint is free and OpenAI-compatible, so we reuse
        # the same GROQ_API_KEY already used for chat — no separate OpenAI
        # billing needed.
        groq_audio = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        with open(local_path, "rb") as f:
            transcript = groq_audio.audio.transcriptions.create(model="whisper-large-v3", file=f)
        text = transcript.text
    except Exception as e:
        logger.exception("transcription failed")
        await update.message.reply_text(f"Couldn't transcribe that voice note ({e}). Mind typing it instead?")
        return
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

    reply = ai.get_response(telegram_id, text)
    await update.message.reply_text(reply)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = str(update.effective_user.id)
    db.get_or_create_user(telegram_id, name=update.effective_user.first_name or "")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    photo = update.message.photo[-1]  # highest resolution
    photo_file = await photo.get_file()
    local_path = os.path.join(tempfile.gettempdir(), f"{photo.file_unique_id}.jpg")
    await photo_file.download_to_drive(local_path)

    with open(local_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(local_path)

    caption = update.message.caption or "Take a look at this and tell me what's useful here."
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": caption},
    ]

    reply = ai.get_response(telegram_id, content)
    await update.message.reply_text(reply)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Basic fallback so PDFs/docs don't get silently ignored during a demo.
    await update.message.reply_text(
        "I can see the document — full PDF parsing isn't wired up in this build yet, "
        "but ask me anything about it in the meantime and I'll do my best."
    )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing from .env")

    db.init_db()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Atlas bot starting (long polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()