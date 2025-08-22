import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Use relative imports because this file is inside src/
from .document_loader import load_and_chunk_all_pdfs
from .vector_database import build_vector_store, save_vector_store, load_vector_store, search


# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# -------- EMBED CHUNKS + BUILD VECTOR STORE --------
def embed_and_store(chunks, force_recompute=False):
    """
    Embed chunks and save in FAISS vector store.
    If already saved, loads from FAISS index.
    """
    index, metadata = load_vector_store()

    if index is not None and not force_recompute:
        print("✅ Loaded existing FAISS vector store.")
        return index, metadata

    print("⚡ No cache found or recompute=True. Embedding now...")
    embeddings = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]  # extract text from dictionary

        if not text.strip():
            embeddings.append(np.zeros(768))
            continue

        # Limit text length
        text = text[:3000]

        try:
            res = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(res["embedding"])
            print(f"Chunk {i} embedded")  # optional progress print
        except Exception as e:
            print(f"❌ Embedding failed for chunk {i}: {e}")
            embeddings.append(np.zeros(768))

    # Build FAISS index (keep full chunk dict as metadata)
    index, metadata = build_vector_store(embeddings, chunks, use_cosine=True)

    # Save FAISS + metadata
    save_vector_store(index, metadata)
    print("💾 Saved FAISS index and metadata.")

    return index, metadata


# -------- EMBED QUERY --------
def embed_query(query):
    if not query.strip():
        return np.zeros(768)

    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return res["embedding"]

    except Exception as e:
        print(f"❌ Embedding failed for query: {e}")
        return np.zeros(768)


# -------- RETRIEVE RELEVANT CHUNKS --------
def retrieve_relevant_chunks(query, index, metadata, top_k=5):
    query_embedding = embed_query(query)
    results, scores = search(index, query_embedding, metadata, k=top_k, use_cosine=True)

    # results are full chunk dicts
    return list(zip(results, scores))  # return dict + similarity score
