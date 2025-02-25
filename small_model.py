import json
import os
import pdfplumber
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

os.environ["USE_TF"] = "0"  # Disable TensorFlow
METADATA_FILE = "./Documents/processed_pdfs.json"

# Load Sentence Transformer for embedding queries
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB for storing and retrieving documents
chroma_client = chromadb.PersistentClient(path="./rag_database2")
collection = chroma_client.get_or_create_collection(name="survival_knowledge")

# Set device and dtype for performance
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16 if device == "mps" else torch.float32

print(f"Using device: {device}, dtype: {torch_dtype}")

# Load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Alternative approach
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch_dtype,
)

# Speed up inference with torch.compile
if device == "cuda":
    model = torch.compile(model)


# Load processed PDFs from a JSON file
def load_processed_pdfs():
    return json.load(open(METADATA_FILE, "r")) if os.path.exists(METADATA_FILE) else {}


def save_processed_pdfs(processed_pdfs):
    with open(METADATA_FILE, "w") as f:
        json.dump(processed_pdfs, f, indent=4)


# Process text and store embeddings
def process_text(text, source_name):
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks, batch_size=8).tolist()  # Batch encoding for speed

    existing_ids = set(collection.get(ids=[f"{source_name}-{i}" for i in range(len(chunks))])["ids"])

    new_entries = [
        (f"{source_name}-{i}", chunk, embeddings[i])
        for i, chunk in enumerate(chunks)
        if f"{source_name}-{i}" not in existing_ids
    ]

    if new_entries:
        ids, docs, embs = zip(*new_entries)
        collection.add(ids=list(ids), documents=list(docs), embeddings=list(embs))


# Process PDFs in parallel
def process_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(filter(None, [page.extract_text() for page in pdf.pages]))

    if text.strip():
        process_text(text, os.path.basename(pdf_path))
        return os.path.basename(pdf_path)
    return None


# Retrieve top-k relevant text
def retrieve_relevant_text(query, top_k=3):
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []


chat_hist = ""


def generate_response(user_query):
    """Retrieves relevant info and formats a response."""
    retrieved_texts = retrieve_relevant_text(user_query)
    context = " ".join(retrieved_texts)

    prompt = f"""
        You are a survival expert. Answer the user's question concisely with clear steps.
        Assume they are in a survival situation unless stated otherwise.
        Use only relevant retrieved information.
        Prioritize speed in your response.

        Survival Info: {context}
        Chat history: {chat_hist}
        User question: {user_query}
        Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=600,  # Reduced from 100000 for speed
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.split("</think>")[-1].strip() if "</think>" in response else response


if __name__ == "__main__":
    processed_pdfs = load_processed_pdfs()

    # Parallel PDF processing
    pdf_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk("./Documents")
        for file in files if file.endswith(".pdf") and file not in processed_pdfs
    ]

    with ThreadPoolExecutor() as executor:
        new_pdfs = list(filter(None, executor.map(process_pdf, pdf_paths)))

    for pdf in new_pdfs:
        processed_pdfs[pdf] = True

    save_processed_pdfs(processed_pdfs)

    # User interaction loop
    while True:
        user_input = input("\nAsk a survival question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break

        response = generate_response(user_input)
        chat_hist = response
        print("\nAI Response:", response)
