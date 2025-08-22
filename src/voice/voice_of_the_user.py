
import os
import logging
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from dotenv import load_dotenv
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------ CONFIG ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Tell Pydub exactly where ffmpeg is
AudioSegment.converter = r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"


def record_audio(file_path: str, timeout=30, phrase_time_limit=10):
    """
    Records audio from the microphone and saves it as an MP3 file.

    Args:
        file_path (str): Path to save the recorded audio.
        timeout (int): Maximum wait time to start speaking.
        phrase_time_limit (int): Maximum length of a single phrase.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("🎤 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("🎤 Please start speaking...")

            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("🎤 Recording complete.")

            # Convert to MP3
            wav_data = audio.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")

            logging.info(f"✅ Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"❌ Recording failed: {e}")


def transcribe_audio_gemini(audio_filepath: str, language="en") -> str:
    """
    Transcribe audio using Gemini embeddings (speech-to-text).

    Args:
        audio_filepath (str): Path to the audio file.
        language (str): Language code ('en' for English, 'ne' for Nepali).

    Returns:
        str: Transcribed text.
    """
    try:
        # Open the audio file
        with open(audio_filepath, "rb") as f:
            audio_bytes = f.read()

        # Gemini API transcription call
        transcription = genai.audio.transcribe(
            model="models/whisper-1",  # or your selected Gemini STT model
            audio=audio_bytes,
            language=language
        )
        return transcription["text"]
    except Exception as e:
        logging.error(f"❌ Transcription failed: {e}")
        return ""
