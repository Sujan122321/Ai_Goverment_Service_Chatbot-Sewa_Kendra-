# app.py
import streamlit as st
import os
from io import BytesIO
import platform
import subprocess

import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

from src.text_embedding import retrieve_relevant_chunks
from src.vector_database import load_vector_store

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Government Service Chatbot", page_icon="🤖", layout="wide")

# Load FAISS vector store
index, metadata = load_vector_store()
if index is None:
    st.error("Vector store not found. Please run precompute_embeddings.py first.")
    st.stop()

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ------------------ HELPER FUNCTIONS ------------------

def record_audio(file_path="user_audio.wav", timeout=10, phrase_time_limit=30):
    """Record audio from microphone and save as WAV."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("🎤 Recording... Speak now!")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            with open(file_path, "wb") as f:
                f.write(audio.get_wav_data())
            st.success(f"✅ Audio saved to {file_path}")
    except Exception as e:
        st.error(f"❌ Recording failed: {e}")

def transcribe_audio(audio_file):
    """Convert recorded WAV audio to text."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return ""

def speak_text(text, output_file="ai_response.wav"):
    """Convert text to speech using gTTS and play audio."""
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save("temp.mp3")
    
    # Convert MP3 → WAV for consistent playback
    sound = AudioSegment.from_mp3("temp.mp3")
    sound.export(output_file, format="wav")

    # Play audio
    os_name = platform.system()
    try:
        if os_name == "Windows":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os_name == "Darwin":
            subprocess.run(["afplay", output_file])
        elif os_name == "Linux":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        st.error(f"❌ Failed to play audio: {e}")

# ------------------ UI ------------------
st.title("🤖 AI Government Service Chatbot")
lang = st.radio("Select Language", ["English", "Nepali"])
input_mode = st.radio("Input Mode", ["Text", "Voice"])
voice_output = st.checkbox("Enable Voice Response", value=True)

# ------------------ GET USER INPUT ------------------
if input_mode == "Text":
    st.session_state.user_input = st.text_input("Type your question")
else:
    if st.button("🎤 Record Voice"):
        audio_file = "user_audio.wav"
        record_audio(audio_file)
        st.session_state.user_input = transcribe_audio(audio_file)
        if st.session_state.user_input:
            st.success(f"You said: {st.session_state.user_input}")

# ------------------ PROCESS USER INPUT ------------------
if st.button("Send") and st.session_state.user_input.strip() != "":
    user_question = st.session_state.user_input

    # Retrieve relevant chunks from FAISS
    results = retrieve_relevant_chunks(user_question, index, metadata, top_k=3)
    context_texts = [r[0]["text"] for r in results]
    response_text = "\n".join(context_texts).strip()
    if not response_text:
        response_text = "Sorry, I could not find relevant information."

    # Optional: voice output
    if voice_output:
        speak_text(response_text)

    # Save chat history
    st.session_state.chat_history.append({"user": user_question, "bot": response_text})
    st.session_state.user_input = ""  # reset input

# ------------------ DISPLAY CHAT HISTORY ------------------
st.subheader("Chat History")
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
