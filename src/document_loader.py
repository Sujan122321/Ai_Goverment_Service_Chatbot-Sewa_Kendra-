import os
import fitz  # PyMuPDF

# Function to load a single PDF file and extract text
def load_pdf(file_path):
    doc = fitz.open(file_path)  # open PDF directly from path
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Function to load all PDFs in a folder and chunk them
def load_and_chunk_all_pdfs(folder_path, chunk_size=300, overlap=50):
    all_chunks = []
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):  # only process PDF files
            file_path = os.path.join(folder_path, file_name)
            print(f"Loading {file_path} ...")
            
            text = load_pdf(file_path)
            chunks = chunk_text(text, chunk_size, overlap)
            
            # Optionally store with filename info
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "file": file_name,
                    "chunk_id": i,
                    "text": chunk
                })
    
    return all_chunks
