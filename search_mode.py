import os
import json
import nltk
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
import chromadb
from config import SEARCH_METADATA_FILE, SEARCH_DB_PATH, SEARCH_COLLECTION_NAME

nltk.download('punkt')

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB for storing and retrieving documents
chroma_client = chromadb.PersistentClient(path=SEARCH_DB_PATH)
collection = chroma_client.get_or_create_collection(name=SEARCH_COLLECTION_NAME)


# Change this to the directory where the model and other needed files are stored
model_name = f"sentence_compression_{os.name}"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.generation_config = GenerationConfig()


def summarize_text_bart(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True,
                             decoder_start_token_id=tokenizer.bos_token_id)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def combine_short_sentences(sentences, target_length=15):
    combined_sentences = []
    current_combination = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        # If the sentence is short, try to combine it
        if sentence_length < target_length:
            if current_length + sentence_length <= target_length:
                current_combination.append(sentence)
                current_length += sentence_length
            else:
                # Add the combined short sentences to the final list
                combined_sentences.append(" ".join(current_combination))
                # Start a new combination with the current short sentence
                current_combination = [sentence]
                current_length = sentence_length
        else:
            # If the sentence is long enough, add it as is
            if current_combination:
                combined_sentences.append(" ".join(current_combination))
                current_combination = []
                current_length = 0
            combined_sentences.append(sentence)

    # Append any leftover short sentence combinations
    if current_combination:
        combined_sentences.append(" ".join(current_combination))

    return combined_sentences


def split_into_chunks(text, chunk_size=300, overlap_interval=5):
    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)
    chunks = []

    # Step 2: Split any sentence longer than 40 words into chunks of 20 words
    new_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 40:
            # Split sentence into smaller chunks of 20 words
            for i in range(0, len(words), 20):
                new_sentences.append(" ".join(words[i:i + 20]))
        else:
            new_sentences.append(sentence)

    new_sentences = combine_short_sentences(new_sentences, target_length=15)

    # Now loop through the sentences and create chunks
    sentences = new_sentences

    # Loop over different starting points for overlap
    for start_idx in range(0, len(sentences), overlap_interval):
        current_chunk = []
        current_length = 0
        i = start_idx  # Restart at the current starting index

        # Start combining sentences
        while i < len(sentences) and current_length + len(sentences[i].split()) <= chunk_size:
            current_chunk.append(sentences[i])
            current_length += len(sentences[i].split())
            i += 1

        #print(f"New chunk length: {current_length} words (starting at sentence {start_idx})")

        # Add the current chunk to the list of chunks
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
    if os.path.exists(SEARCH_METADATA_FILE):
        with open(SEARCH_METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_processed_pdfs(processed_pdfs):
    """Save the list of processed PDFs to a JSON file."""
    with open(SEARCH_METADATA_FILE, "w") as f:
        json.dump(processed_pdfs, f, indent=4)


def process_pdf(pdf_path, processed_pdfs):
    """Extracts text from a PDF and stores it in ChromaDB."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    if text.strip():
        process_text(text, os.path.basename(pdf_path))
        processed_pdfs[os.path.basename(pdf_path)] = True
        save_processed_pdfs(processed_pdfs)


def ai_search(query):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    top_result = results['documents'][0][0] if results['documents'] and results['documents'][0] else None

    if top_result is None:
        return "Please expand your search query to retrieve a better result"

    return
