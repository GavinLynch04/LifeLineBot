import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from knowledge import KnowledgeBase

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

# Set device and dtype for performance
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16 if device == "mps" else torch.float32

print(f"Using device: {device}, dtype: {torch_dtype}")

# Load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch_dtype,
    offload_buffers=True
)

# Speed up inference with torch.compile
if device == "cuda":
    model = torch.compile(model)

chat_hist = ""


def generate_response(user_query):
    """Retrieves relevant info and formats a response."""
    retrieved_texts = KnowledgeBase.retrieve_relevant_text(user_query)
    context = " ".join(retrieved_texts)

    prompt = f"""
        You are a survival expert. Answer the user's question concisely with clear steps.
        Assume they are in a survival situation unless stated otherwise.
        Use only relevant retrieved information.

        Survival Info: {context}
        Chat history: {chat_hist}
        User question: {user_query}
        Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=450,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.split("</think>")[-1].strip() if "</think>" in response else response


if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a survival question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break

        response = generate_response(user_input)
        chat_hist = response
        print("\nAI Response:", response)
