import fitz 

def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype ='pdf')
    text =""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size = 300, overlap=50):
    chunks=[]
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks 
