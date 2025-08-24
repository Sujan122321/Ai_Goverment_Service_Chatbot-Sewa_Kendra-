# app.py
import streamlit as st
import os
import platform
import subprocess
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
import google.generativeai as genai

from src.text_embedding import retrieve_relevant_chunks
from src.vector_database import load_vector_store

# Set ffmpeg path for PyDub
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"  # <-- update with your ffmpeg path

# ------------------ CONFIG ------------------
st.set_page_config(page_title="🤖 AI सरकारी सेवा चैटबोट", layout="wide")

# Load FAISS vector store
index, metadata = load_vector_store()
if index is None:
    st.error("Vector store not found. Run embeddings first.")
    st.stop()

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ------------------ HELPER FUNCTIONS ------------------
def record_audio(file_path="user_audio.wav", timeout=10, phrase_time_limit=30):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 वातावरणको आवाज मिलाउँदै...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("🎤 बोल्न सुरु गर्नुहोस्...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            with open(file_path, "wb") as f:
                f.write(audio.get_wav_data())
            st.success(f"✅ आवाज सुरक्षित गरियो: {file_path}")
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
        st.error(f"❌ ट्रान्सक्रिप्सन असफल: {e}")
        return ""

def speak_text(text, output_file="ai_response.wav"):
    tts = gTTS(text=text, lang="ne", slow=False)
    tts.save("temp.mp3")
    sound = AudioSegment.from_mp3("temp.mp3")
    sound.export(output_file, format="wav")
    
    os_name = platform.system()
    try:
        if os_name == "Windows":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os_name == "Darwin":
            subprocess.run(["afplay", output_file])
        elif os_name == "Linux":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        st.error(f"❌ अडियो चलाउन असफल: {e}")

def generate_answer_with_rag(question):
    """Use RAG + Gemini to generate Nepali answer."""
    # 1️⃣ Retrieve relevant chunks
    results = retrieve_relevant_chunks(question, index, metadata, top_k=3)
    context_texts = [r[0]["text"] for r in results]
    context = "\n\n".join(context_texts)
    
    # 2️⃣ Gemini prompt
    prompt = f"तपाईंलाई निम्न सन्दर्भको आधारमा नेपालीमा छोटो, सरल र स्पष्ट उत्तर दिनुहोस्:\n\n{context}\n\nप्रश्न: {question}\n\nउत्तर:"
    
    try:
        response = genai.chat.create(
            model="gemini-2.0-turbo",
            messages=[{"author": "user", "content": prompt}]
        )
        return response.last.message.content
    except Exception as e:
        st.error(f"❌ Gemini failed: {e}")
        return "माफ गर्नुहोस्, अहिले उत्तर उपलब्ध छैन।"

# ------------------ UI ------------------
st.title("🤖 AI सरकारी सेवा चैटबोट (नेपाली)")
input_mode = st.radio("इनपुट विधि चयन गर्नुहोस्", ["✍️ टेक्स्ट", "🎤 आवाज"])
voice_output = st.checkbox("आवाजमा प्रतिक्रिया दिनुहोस्", value=True)

# ------------------ GET USER INPUT ------------------
if input_mode == "✍️ टेक्स्ट":
    st.session_state.user_input = st.text_input("तपाईंको प्रश्न लेख्नुहोस्")
else:
    if st.button("🎤 रेकर्ड गर्नुहोस्"):
        audio_file = "user_audio.wav"
        record_audio(audio_file)
        st.session_state.user_input = transcribe_audio(audio_file)
        if st.session_state.user_input:
            st.success(f"तपाईंले भन्नुभयो: {st.session_state.user_input}")

# ------------------ PROCESS USER INPUT ------------------
if st.button("Send") and st.session_state.user_input.strip() != "":
    user_question = st.session_state.user_input
    response_text = generate_answer_with_rag(user_question)

    if voice_output:
        speak_text(response_text)

    st.session_state.chat_history.append({"user": user_question, "bot": response_text})
    st.session_state.user_input = ""

# ------------------ DISPLAY CHAT HISTORY ------------------
st.subheader("वार्तालाप इतिहास")
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**तपाईं:** {chat['user']}")
    st.markdown(f"**बोट:** {chat['bot']}")
    st.markdown("---")
