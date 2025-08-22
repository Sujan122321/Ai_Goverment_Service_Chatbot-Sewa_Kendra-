from document_loader import load_and_chunk_all_pdfs
from text_embedding import embed_and_store

DATASET_FOLDER = "datasets\AI_Goverment_Services"  # folder containing your PDFs

def main():
    # Step 1: Load and chunk PDFs
    print("📄 Loading and chunking PDFs...")
    chunks = load_and_chunk_all_pdfs(DATASET_FOLDER, chunk_size=1000, overlap=200)
    print(f"✅ Loaded {len(chunks)} chunks from PDFs.")

    # Step 2: Embed chunks and save to FAISS
    print("⚡ Embedding chunks and building FAISS vector store...")
    index, metadata = embed_and_store(chunks)
    print("💾 Embeddings and FAISS vector store saved successfully!")

if __name__ == "__main__":
    main()
