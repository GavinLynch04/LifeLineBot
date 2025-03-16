import importlib
import time
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import KnowledgeBase
from search_mode import ai_search
from dotenv import load_dotenv
import chromadb
from config import (
    AVAILABLE_MODES,
    CHAT_METADATA_FILE, SEARCH_METADATA_FILE,  # Metadata file paths
    CHAT_DB_PATH, SEARCH_DB_PATH,  # ChromaDB paths
    CHAT_COLLECTION_NAME, SEARCH_COLLECTION_NAME  # ChromaDB collection names
)

# Initialize ChromaDB clients and collections
chat_chroma_client = chromadb.PersistentClient(path=CHAT_DB_PATH)
search_chroma_client = chromadb.PersistentClient(path=SEARCH_DB_PATH)

chat_collection = chat_chroma_client.get_or_create_collection(name=CHAT_COLLECTION_NAME)
search_collection = search_chroma_client.get_or_create_collection(name=SEARCH_COLLECTION_NAME)

load_dotenv()

base = 0
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ["USE_TF"] = "0"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16

print(f"Using device: {device}")

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map=device,
).eval()

# Compile Model
model = torch.compile(model)


def query_gemini(prompt, model="gemini-2.0-flash"):
    """Query Google Gemini API and return response."""
    try:
        response = genai.GenerativeModel(model).generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini API failed: {e}. Falling back to local model.")
        return None


def expand_query(user_query):
    """Uses LLM to rephrase or expand the query to improve retrieval."""
    prompt = f"""
    Rewrite the following user query to be clearer and optimized for retrieving survival knowledge from a database quickly.
    Only print the query.
    User query: "{user_query}"
    """
    expanded_query = query_gemini(prompt)
    if expanded_query:
        base = KnowledgeBase.update_user_data(user_query)
        return expanded_query

    print("🔄 Falling back to local expansion model...")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    print("Generating first output")
    start = time.time()
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        max_length=200,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Less randomness
        top_p=0.9,  # Less likely to generate long rambling responses
        do_sample=True,
    )
    end = time.time()
    print("Time taken to generate output:", end - start)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


chat_hist = ""


def generate_response(user_query):
    """Retrieves relevant info and formats a response with Mistral 8x7B."""
    retrieved_texts = KnowledgeBase.retrieve_relevant_text(user_query)
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

    gemini_response = query_gemini(prompt)
    if gemini_response:
        return gemini_response

    print("🔄 Falling back to local model...")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=10000,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


def load_module_functions(mode):
    """Dynamically load functions from the correct mode-based module."""
    if mode not in AVAILABLE_MODES:
        raise ValueError(f"Invalid mode: {mode}")

    module_name = AVAILABLE_MODES[mode]
    module = importlib.import_module(module_name)  # Dynamically import module

    return module.load_processed_pdfs, module.process_pdf  # Return functions


if __name__ == "__main__":
    # Process PDFs for each mode
    for mode in AVAILABLE_MODES.keys():
        load_processed_pdfs, process_pdf = load_module_functions(mode)  # Get correct functions

        processed_pdfs = load_processed_pdfs()  # Call the correct function dynamically
        for root, _, files in os.walk("./Documents"):
            for file in files:
                if file.endswith(".pdf") and file not in processed_pdfs:
                    pdf_path = os.path.join(root, file)
                    print(f"[{mode.upper()}] Processing new PDF: {pdf_path}")
                    process_pdf(pdf_path, processed_pdfs)  # Call correct function
                else:
                    # print(f"[{mode.upper()}] Skipping already processed PDF: {file}")
                    pass

    output_mode = "chat"

    while True:
        user_input = input(f"\nAsk a survival question ({output_mode} mode, type 'exit', 'search', or 'chat'): ")

        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "search":
            output_mode = "search"
            print("Switched to search mode.")
            continue
        elif user_input.lower() == "chat":
            output_mode = "chat"
            print("Switched to chat mode.")
            continue

        if output_mode == "chat":
            response = generate_response(user_input)
            chat_hist = response
            print("\nAI Response:", response)
        elif output_mode == "search":
            search_result = ai_search(user_input)
            for result in search_result:
                print("\nSearch Result:", result)
