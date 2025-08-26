import re
from gtts import gTTS
import streamlit as st

def clean_text_for_tts(text):
    # Remove Markdown formatting like *, **, _, __, ``
    cleaned_text = re.sub(r"[*_`]", "", text)
    return cleaned_text

def speak_text(text, output_file="ai_response.mp3"):
    try:
        # Clean text first
        clean_text = clean_text_for_tts(text)

        # Generate TTS MP3
        tts = gTTS(text=clean_text, lang="ne", slow=False)
        tts.save(output_file)

        # Play in Streamlit
        st.audio(output_file, format="audio/mp3")
        return output_file

    except Exception as e:
        st.error(f"❌ Audio generation failed: {e}")
        return None
