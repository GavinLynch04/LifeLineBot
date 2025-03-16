import json
import os
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer
from SurvivalBot import expand_query, query_gemini

METADATA_FILE = "./Documents/processed_pdfs.json"

# Load Sentence Transformer for embedding queries
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB for storing and retrieving documents
chroma_client = chromadb.PersistentClient(path="./rag_database2")
collection = chroma_client.get_or_create_collection(name="survival_knowledge")

class UserKnowledgeBase:
    def __init__(self):
        """
        Initializes the user knowledge base with empty datasets for terrain, weather,
        conditions, urgency, and other data.
        """
        self.data = {
            "rescuee_location": "Location Data: ",
            "rescue_weather": "Weather Data: ",
            "rescuee_condition": "Rescuee Condition: ",
            "urgency": "Urgency: ",
            "other_data": "Other Relevant Data: ",
        }

userData = UserKnowledgeBase()

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


def process_text(text, source_name): #ADD OVERLAP TO CHUNKS
    """Splits text into chunks, embeds them, and stores them in ChromaDB without duplicates."""
    chunks = [text[i:i + 100] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks).tolist()

    # Get existing document IDs in ChromaDB
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


def process_pdf(pdf_path, processed_pdfs):
    """Extracts text from a PDF and stores it in ChromaDB."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    if text.strip():
        process_text(text, os.path.basename(pdf_path))
        processed_pdfs[os.path.basename(pdf_path)] = True
        save_processed_pdfs(processed_pdfs)


def update_user_data(message):
    # Prompt gemini to update the user data based on user message
    prompt = (
        "Update the following JSON data based on the user message. If there is no new relevant data, leave JSON as is. Never delete data, only add on."
        "Return only valid JSON without any extra text.\n\n"
        "Current JSON Data:\n"
        f"{json.dumps(userData.data, indent=2)}\n\n"
        "User Message:\n"
        f"{message}"
    )

    response = query_gemini(prompt)
    previous_data = userData.data

    try:
        userData.data = json.loads(response)
        return userData
    except json.JSONDecodeError:
        userData.data = previous_data
        return userData
    except Exception as e:
        return f"Error: {e}"


def retrieve_relevant_text(query, top_k=3):
    expanded_query = expand_query(query)
    if "</think>" in expanded_query:
        expanded_query = expanded_query.split("</think>")[-1].strip()
    query_embedding = embedder.encode([expanded_query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []

def ai_search(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []