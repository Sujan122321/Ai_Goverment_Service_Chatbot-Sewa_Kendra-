import faiss
import numpy as np
import os
import pickle

INDEX_PATH = "vector_store.index"
META_PATH = "metadata.pkl"

# -------- SAVE & LOAD --------
def save_vector_store(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def load_vector_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# -------- BUILD VECTOR STORE --------
def build_vector_store(embeddings, texts, use_cosine=True):
    """
    embeddings: list of vectors
    texts: list of corresponding text chunks or metadata
    use_cosine: if True, use cosine similarity (normalize vectors + inner product)
    """
    embeddings = np.array(embeddings).astype("float32")

    if use_cosine:
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])  # Euclidean distance

    index.add(embeddings)
    return index, texts


# -------- SEARCH --------
def search(index, query_vector, metadata, k=5, use_cosine=True):
    query_vector = np.array([query_vector]).astype("float32")
    
    if use_cosine:
        faiss.normalize_L2(query_vector)
    
    D, I = index.search(query_vector, k)
    results = [metadata[i] for i in I[0]]
    return results, D[0]  # return texts + distances
