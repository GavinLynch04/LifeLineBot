import json
import os
import pdfplumber
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["USE_TF"] = "0"  # Disable TensorFlow
METADATA_FILE = "./Documents/processed_pdfs.json"

# Load Sentence Transformer for embedding queries
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB for storing and retrieving documents
chroma_client = chromadb.PersistentClient(path="./rag_database2")
collection = chroma_client.get_or_create_collection(name="survival_knowledge")

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float32  # MPS does not work well with float16

print(f"Using device: {device}")

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map=device
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


def process_text(text, source_name):
    """Splits text into chunks, embeds them, and stores them in ChromaDB without duplicates."""
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
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

def retrieve_relevant_text(query, top_k=3):
    """Finds the most relevant survival information for a query."""
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []

chat_hist=""
def generate_response(user_query):
    """Retrieves relevant info and formats a response with Mistral 8x7B."""
    retrieved_texts = retrieve_relevant_text(user_query)
    context = " ".join(retrieved_texts)
    #print(f"RAG Generated Info: {context}")

    # Create a user-friendly prompt
    prompt = f"""
        You are a survival expert. Answer the user's question in a concise, and structured manner.
        Assume the user is in a survival situation unless otherwise told so.
        If you need more info from the user to make a good recommendation, ask.
        You do not need to use all the info, only some of it will be relevant, choose what to use.
        Make sure your response is complete and directed at the user, with clear steps to follow.
        Make your response fairly quick, as the user might be in a dire situation.
        Here is the info:

        Survival Information:
        {context}
        
        Also take into account any previous questions (if any) and answers that have occurred to formulate your response.
        Chat history:
        {chat_hist}

        User question: {user_query}

        Answer:
        """
    # Ensure model has a pad token (set it to EOS token if missing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate response with attention mask
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Ensure reliable results
        max_length=100000,
        pad_token_id=tokenizer.pad_token_id  # Avoid padding issues
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


if __name__ == "__main__":
    processed_pdfs = load_processed_pdfs()

    # Load and process only new PDFs
    for root, _, files in os.walk("./Documents"):
        for file in files:
            if file.endswith(".pdf") and file not in processed_pdfs:
                pdf_path = os.path.join(root, file)
                print(f"Processing new PDF: {pdf_path}")
                process_pdf(pdf_path, processed_pdfs)
            else:
                print(f"Skipping already processed PDF: {file}")

    # User interaction loop
    while True:
        user_input = input("\nAsk a survival question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break

        response = generate_response(user_input)
        chat_hist = response
        print("\nAI Response:", response)
