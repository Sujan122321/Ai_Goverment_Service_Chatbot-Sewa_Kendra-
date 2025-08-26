# simple_app.py
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# loading the embedding and the vector_database
from src.vector_database import load_vector_store
from src.text_embedding import retrieve_relevant_chunks



# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load FAISS vector store
index, metadata = load_vector_store()
if index is None:
    st.error("Vector store not found. Run precompute_embeddings.py first.")
    st.stop()

# Streamlit UI
st.title("🧠 Simple RAG Chatbot")
st.write("Ask me anything from your knowledge base!")

# User input
user_query = st.text_input("Enter your question:")

if user_query:
    # Step 1: Retrieve relevant chunks
    results = retrieve_relevant_chunks(user_query, index, metadata, top_k=3)
    context_texts = [r[0]["text"] for r in results]
    context_text = "\n".join(context_texts)

    # Step 2: Prepare Gemini prompt
    prompt = f"""Give me answer in clear and details also give the process in nepali language. Give only in nepali Language,
    and give link if available in context.{context_text} 
    Answer only  based on the context,  if context not available answer 'answer not available in nepali language'
    
    

प्रश्न: {user_query}

उत्तर:"""

    # Step 3: Call Gemini
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        st.error(f"Gemini failed: {e}")
        answer = "माफ गर्नुहोस्, अहिले उत्तर उपलब्ध छैन।"

    # Step 4: Show response
    st.subheader("💡 उत्तर (नेपाली):")
    st.write(answer)
