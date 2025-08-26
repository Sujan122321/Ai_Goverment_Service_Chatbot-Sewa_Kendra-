# simple_app.py
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# loading the embedding and the vector_database
from src.vector_database import load_vector_store
from src.text_embedding import retrieve_relevant_chunks

# Loading the Voice models
from src.voice.voice_of_the_user import transcribe_audio, record_audio
from src.voice.voice_of_the_ai import speak_text

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load FAISS vector store
index, metadata = load_vector_store()
if index is None:
    st.error("Vector store not found. Run precompute_embeddings.py first.")
    st.stop()

# ----------------- UI -----------------
st.set_page_config(page_title="🧠 Nepali RAG Chatbot", page_icon="🤖", layout="wide")

# ----------------- SESSION STATE -----------------
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

# ----------------- SIDEBAR -----------------
st.sidebar.title("🛠 Input Options")
input_type = st.sidebar.radio("Select input type:", ("Text", "Audio"))

if input_type == "Text":
    user_query = st.sidebar.text_input("Type your question here:")
    if st.sidebar.button("🚀 Send Text"):
        if user_query.strip() != "":
            st.session_state["user_query"] = user_query

elif input_type == "Audio":
    if st.sidebar.button("🎤 Record Question"):
        audio_file = "user_audio.wav"
        record_audio(audio_file)
        user_query = transcribe_audio(audio_file)
        if user_query:
            st.session_state["user_query"] = user_query
            st.sidebar.success(f"तपाईंले भन्नुभयो: {user_query}")
        else:
            st.sidebar.error("❌ Could not transcribe audio. Please try again.")

# ----------------- MAIN CONTENT -----------------
st.title("🧠 Simple RAG Chatbot (Nepali Input)")
st.write("Get answers in Nepali based on your knowledge base.")

if st.session_state.get("user_query"):
    user_query = st.session_state["user_query"]

    # Show the question while thinking
    st.markdown(f"**तपाईंको प्रश्न:** {user_query}")
    with st.spinner("🤔 Thinking..."):
        # Step 1: Retrieve relevant chunks
        results = retrieve_relevant_chunks(user_query, index, metadata, top_k=3)
        context_texts = [r[0]["text"] for r in results]
        context_text = "\n".join(context_texts)

        # Step 2: Prepare Gemini prompt
        prompt = f"""
        Give me answer in clear and details also give the process in nepali language. Give only in nepali Language,
        and give link if available in context.{context_text} 
        Answer only  based on the context,  if context not available answer 'answer not available in nepali language'

Context:
{context_text}

प्रश्न: {user_query}

उत्तर:
"""

        # Step 3: Call Gemini
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                answer = response.text
            else:
                answer = response.candidates[0].content.parts[0].text

            # # Speak the answer
            speak_text(answer)

        except Exception as e:
            st.error(f"Gemini failed: {e}")
            answer = "माफ गर्नुहोस्, अहिले उत्तर उपलब्ध छैन।"

    # ----------------- SHOW RESPONSE -----------------
    st.subheader("💡 उत्तर (नेपाली):")
    st.success(answer)

    # Clear current query after showing answer
    st.session_state["user_query"] = ""