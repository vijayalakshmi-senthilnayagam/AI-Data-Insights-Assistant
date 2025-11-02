from huggingface_hub import InferenceClient

# 🔹 Initialize client with your token
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    token="REMOVED"
)

# 🔹 Test query
prompt = "Explain what correlation means in data analysis."
response = client.chat_completion(
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=150,
)
print("🔹 Llama 3 Response:\n", response)


# response = client.text_generation(prompt, max_new_tokens=150)
