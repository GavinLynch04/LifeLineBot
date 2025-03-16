import time
from llama_cpp import Llama
import json
import os
import pdfplumber
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tensorflow.python.eager.context import num_gpus

os.environ["USE_TF"] = "0"
METADATA_FILE = "../Documents/processed_pdfs.json"
n_threads = os.cpu_count() // 2

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
	filename="Llama-3.2-3B-Instruct-Q4_0.gguf",
    verbose=False,
    n_ctx=40096,
    n_gpu_layers=10,
    n_threads=n_threads,
    batch_size=64,
)


embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="../rag_database2")
collection = chroma_client.get_or_create_collection(name="survival_knowledge")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def expand_query(user_query):
    """Uses the quantized LLM to rephrase or expand the query for improved retrieval."""
    prompt = f"""
Rewrite the following user question to be clearer and optimized for retrieving survival knowledge quickly. This is NOT a SQL query, but a text based query.
Only print the query, nothing else.
User query: "{user_query}"

Expanded query:
"""
    # Generate using the llama_cpp model
    result = llm(prompt, max_tokens=10000)
    # Assuming the result is a dictionary with key 'text'
    expanded = result.get("choices", "")[0].get("text", "").strip()
    return expanded

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

def process_text(text, source_name):  # ADD OVERLAP TO CHUNKS if needed
    """Splits text into chunks, embeds them, and stores them in ChromaDB without duplicates."""
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks).tolist()

    # Get existing document IDs in ChromaDB
    existing_ids = set(
        collection.get(ids=[f"{source_name}-{i}" for i in range(len(chunks))]).get("ids", [])
    )

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_name}-{i}"
        if chunk_id in existing_ids:
            continue
        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embeddings[i]]
        )

def process_pdf(pdf_path, processed_pdfs):
    """Extracts text from a PDF and stores it in ChromaDB."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    if text.strip():
        process_text(text, os.path.basename(pdf_path))
        processed_pdfs[os.path.basename(pdf_path)] = True
        save_processed_pdfs(processed_pdfs)

def retrieve_relevant_text(query, top_k=3):
    expanded_query = expand_query(query)
    # If any control tags exist, clean them up.
    if "</think>" in expanded_query:
        expanded_query = expanded_query.split("</think>")[-1].strip()
    query_embedding = embedder.encode([expanded_query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    # Return the first set of retrieved documents if available
    return results['documents'][0] if results.get('documents') else []

chat_hist = ""

def generate_response(user_query):
    """Retrieves relevant info and formats a response using the quantized LLM via llama_cpp."""
    retrieved_texts = retrieve_relevant_text(user_query)
    context = " ".join(retrieved_texts)

    # Create a prompt for the LLM with context, chat history, and user query
    prompt = f"""
You are a survival expert. Answer the user's question in a concise, structured manner.
Assume the user is in a survival situation unless otherwise stated.
If you need more information to provide a good recommendation, ask for clarification.
Use only the relevant information from the provided context.
Make sure your response is complete, directed at the user, and includes clear steps to follow.
Respond in a compact, short manner, as the user might be in a dire situation.

Survival Information:
{context}

Chat history:
{chat_hist}

User question: {user_query}

Answer:
"""
    result = llm(prompt, max_tokens=100000)
    response = result.get("choices", "")[0].get("text", "").strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response

if __name__ == "__main__":
    processed_pdfs = load_processed_pdfs()
    for root, _, files in os.walk("../Documents"):
        for file in files:
            if file.endswith(".pdf") and file not in processed_pdfs:
                pdf_path = os.path.join(root, file)
                print(f"Processing new PDF: {pdf_path}")
                process_pdf(pdf_path, processed_pdfs)
            else:
                print(f"Skipping already processed PDF: {file}")

    while True:
        user_input = input("\nAsk a survival question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        start = time.time()
        response = generate_response(user_input)
        chat_hist = response
        print("\nAI Response:", response)
        end = time.time()
        print("\nTotal time:", end - start)
