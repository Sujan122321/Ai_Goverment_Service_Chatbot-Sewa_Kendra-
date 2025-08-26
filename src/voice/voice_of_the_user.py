import streamlit as st
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

def record_audio(file_path="user_audio.wav", timeout=10, phrase_time_limit=30):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.sidebar.info("🎤 वातावरणको आवाज मिलाउँदै...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.sidebar.info("🎤 बोल्न सुरु गर्नुहोस्...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            with open(file_path, "wb") as f:
                f.write(audio.get_wav_data())
            st.sidebar.success(f"✅ आवाज सुरक्षित गरियो: {file_path}")
    except Exception as e:
        st.error(f"❌ रेकर्ड गर्न असफल: {e}")

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="ne-NP")  # Nepali
        return text
    except Exception as e:
        st.sidebar.error(f"❌ ट्रान्सक्रिप्सन असफल: {e}")
        return ""

