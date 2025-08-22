
import os
import platform
import subprocess
from gtts import gTTS
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ------------------ TEXT TO SPEECH: gTTS ------------------
def text_to_speech_gtts(input_text: str, output_filepath: str, language="en"):
    """
    Convert text to speech using gTTS and save as MP3, then auto-play.

    Args:
        input_text (str): Text to convert.
        output_filepath (str): Path to save MP3 file.
        language (str): Language code ('en' or 'ne').
    """
    try:
        tts = gTTS(text=input_text, lang=language, slow=False)
        tts.save(output_filepath)
        play_audio_file(output_filepath)
    except Exception as e:
        print(f"❌ gTTS TTS failed: {e}")


# ------------------ TEXT TO SPEECH: ElevenLabs ------------------
def text_to_speech_elevenlabs(input_text: str, output_filepath: str, voice_id="9BWtsMINqrJLrRacOk9x"):
    """
    Convert text to speech using ElevenLabs API, save MP3, and auto-play.

    Args:
        input_text (str): Text to convert.
        output_filepath (str): Path to save MP3 file.
        voice_id (str): ElevenLabs voice ID.
    """
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.text_to_speech.convert(
            text=input_text,
            voice_id=voice_id,
            output_format="mp3_22050_32",
            model_id="eleven_turbo_v2_5"
        )

        with open(output_filepath, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        play_audio_file(output_filepath)
    except Exception as e:
        print(f"❌ ElevenLabs TTS failed: {e}")


# ------------------ AUDIO PLAYER ------------------
def play_audio_file(file_path: str):
    """
    Auto-play the saved MP3 file depending on OS.

    Args:
        file_path (str): Path to audio file.
    """
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(["afplay", file_path])
        elif os_name == "Windows":
            # Using ffplay for Windows (ensure ffmpeg installed)
            subprocess.run(
                [r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffplay.exe", "-nodisp", "-autoexit", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
        elif os_name == "Linux":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"❌ Audio playback failed: {e}")
