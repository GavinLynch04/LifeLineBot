import os
import pdfplumber
import chromadb
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer for embedding queries
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB for storing and retrieving documents
chroma_client = chromadb.PersistentClient(path="./rag_database")
collection = chroma_client.get_or_create_collection(name="survival_knowledge")

# Load LLaMA model using llama-cpp-python
llm = Llama(model_path="llama-2-7b.Q4_K_M.gguf", n_threads=8, verbose=False)  # Adjust threads for M1 Pro


def process_text(text, source_name):
    """Splits text into chunks, embeds them, and stores in ChromaDB."""
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]  # Chunking text
    embeddings = embedder.encode(chunks).tolist()

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{source_name}-{i}"],
            documents=[chunk],
            embeddings=[embeddings[i]]
        )


def process_pdf(pdf_path):
    """Extracts text from a PDF and stores it in ChromaDB."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    process_text(text, os.path.basename(pdf_path))


def retrieve_relevant_text(query, top_k=3):
    """Finds the most relevant survival information for a query."""
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []


def generate_response(user_query):
    """Retrieves relevant info and formats a response with LLaMA."""
    retrieved_texts = retrieve_relevant_text(user_query)
    context = " ".join(retrieved_texts)
    print(f"RAG Generated Info: {context}")

    prompt = f"Here is some survival information, format it into a user friendly response based on the question that the user asked:\n{context}\n\nUser question: {user_query}\n\nAnswer:"

    # Generate response using Llama.cpp
    output = llm(prompt, max_tokens=500)
    return output["choices"][0]["text"]


if __name__ == "__main__":
    # Load and process PDFs
    '''for file in os.listdir("./Documents"):
        if file.endswith(".pdf"):
            print(f"Processing: {file}")
            process_pdf(os.path.join("./Documents", file))'''

    # User interaction loop
    while True:
        user_input = input("\nAsk a survival question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break

        response = generate_response(user_input)
        print("\nAI Response:", response)
