import logging
import os

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from scipy.io import wavfile
from dotenv import load_dotenv
import secrets
from subprocess import CalledProcessError, run
import numpy as np
import json
import requests
from functools import wraps

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    async def command_func(update, context, *args, **kwargs):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        await func(update, context, *args, **kwargs)

    return command_func


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


async def get_audio_transcript(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_id = update.message.voice.file_id
    new_file = await context.bot.get_file(file_id)
    local_file_id = secrets.token_hex(6)
    await new_file.download_to_drive(f"{local_file_id}.oga")

    # audio = load_audio(f"{local_file_id}.oga")
    try:
        run(["ffmpeg", "-i", f"{local_file_id}.oga", f"{local_file_id}.wav"])
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    samplerate, data = wavfile.read(f"{local_file_id}.wav")
    # logger.info(data.shape)

    headers = {
        "Authorization": "Basic MWIzZDI3OWFkNDEyYTFmN2MxOWM1MDVmZTMyZDgyZTM6NTE5ZGZlMThkNTJiMzU1ZDI4NTRkMmYxMDA0YTM1NWE=",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    d = {
        "audio": {
            "rate": samplerate,
            "data": data.tolist(),
        }
    }

    # logger.info(os.environ["BEAM_ENDPOINT"])
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    r = requests.post(os.environ["BEAM_ENDPOINT"], headers=headers, data=json.dumps(d))

    body = json.loads(r.text)

    run(["rm", f"{local_file_id}.oga", f"{local_file_id}.wav"])
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=body["response"]
    )


if __name__ == "__main__":
    TOKEN = os.environ["TELEGRAM_TOKEN"]
    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler("start", start)
    application.add_handler(start_handler)

    audio_handler = MessageHandler(filters.ALL, get_audio_transcript)
    application.add_handler(audio_handler)

    application.run_polling()
