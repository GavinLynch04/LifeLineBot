import json
import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

METADATA_FILE = "./Documents/processed_pdfs.json"

# Load Sentence Transformer for embedding queries
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB for storing and retrieving documents
chroma_client = chromadb.PersistentClient(path="./rag_database2")
collection = chroma_client.get_or_create_collection(name="survival_knowledge")


class KnowledgeBase:
    @staticmethod
    def load_processed_pdfs():
        return json.load(open(METADATA_FILE, "r")) if os.path.exists(METADATA_FILE) else {}

    @staticmethod
    def save_processed_pdfs(processed_pdfs):
        with open(METADATA_FILE, "w") as f:
            json.dump(processed_pdfs, f, indent=4)

    @staticmethod
    def process_text(text, source_name):
        chunks = [text[i:i + 300] for i in range(0, len(text), 300)]
        embeddings = embedder.encode(chunks, batch_size=8).tolist()

        existing_ids = set(collection.get(ids=[f"{source_name}-{i}" for i in range(len(chunks))])["ids"])

        new_entries = [
            (f"{source_name}-{i}", chunk, embeddings[i])
            for i, chunk in enumerate(chunks)
            if f"{source_name}-{i}" not in existing_ids
        ]

        if new_entries:
            ids, docs, embs = zip(*new_entries)
            collection.add(ids=list(ids), documents=list(docs), embeddings=list(embs))

    @staticmethod
    def process_pdf(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(filter(None, [page.extract_text() for page in pdf.pages]))

        if text.strip():
            KnowledgeBase.process_text(text, os.path.basename(pdf_path))
            return os.path.basename(pdf_path)
        return None

    @staticmethod
    def retrieve_relevant_text(query, top_k=3):
        query_embedding = embedder.encode([query])[0].tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    processed_pdfs = KnowledgeBase.load_processed_pdfs()
    pdf_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk("./Documents")
        for file in files if file.endswith(".pdf") and file not in processed_pdfs
    ]

    with ThreadPoolExecutor() as executor:
        new_pdfs = list(filter(None, executor.map(KnowledgeBase.process_pdf, pdf_paths)))

    for pdf in new_pdfs:
        processed_pdfs[pdf] = True

    KnowledgeBase.save_processed_pdfs(processed_pdfs)
