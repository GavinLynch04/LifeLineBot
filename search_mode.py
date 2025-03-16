import os
import json
import nltk
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb

nltk.download('punkt')

embedder = SentenceTransformer("all-MiniLM-L6-v2")

METADATA_FILE = "./Documents/processed_pdfs_search_mode.json"

# Load ChromaDB for storing and retrieving documents
chroma_client = chromadb.PersistentClient(path="./rag_database_search_mode")
collection = chroma_client.get_or_create_collection(name="survival_knowledge")


def split_into_chunks(text, max_tokens=750):
    """Dynamically split text into chunks at sentence boundaries."""
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())  # Estimate tokens
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_text(text, source_name):
    """Splits text adaptively into chunks, embeds them, and stores them in ChromaDB."""
    chunks = split_into_chunks(text)
    embeddings = embedder.encode(chunks).tolist()

    existing_ids = set(collection.get(ids=[f"{source_name}-{i}" for i in range(len(chunks))])["ids"])

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_name}-{i}"
        if chunk_id in existing_ids:
            continue

        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embeddings[i]]
        )


def load_processed_pdfs():
    """Load the list of processed PDFs from a JSON file."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_processed_pdfs(processed_pdfs):
    """Save the list of processed PDFs to a JSON file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(processed_pdfs, f, indent=4)


def process_pdf(pdf_path, processed_pdfs):
    """Extracts text from a PDF and stores it in ChromaDB."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    if text.strip():
        process_text(text, os.path.basename(pdf_path))
        processed_pdfs[os.path.basename(pdf_path)] = True
        save_processed_pdfs(processed_pdfs)


def ai_search(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []