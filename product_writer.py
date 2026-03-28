from groq import Groq
import os

# BLOCK 1 — Connect to Groq API
# Groq serves open-source LLMs (like Llama 3) at very high speed
# This is identical in concept to calling Claude or GPT — just a different provider
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# BLOCK 2 — Define your product
# This data gets loaded into the context window with every call
product = {
    "name": "Extra Virgin Olive Oil",
    "category": "Cooking Oils",
    "origin": "Spain",
    "pack_size": "5L tin"
}

# BLOCK 3 — Build your prompt
# This is your prompt engineering in action
# You're designing exactly what goes into the context window
prompt = f"""
Write a compelling product description for a HoReCa supplier catalog.

Product: {product['name']}
Category: {product['category']}
Origin: {product['origin']}
Pack size: {product['pack_size']}

Keep it under 80 words. Professional tone. Focus on quality and use case.
"""

# BLOCK 4 — Make the API call
# model: we're using Llama 3 — an open source LLM hosted by Groq
# temperature: 0.7 — creative but controlled (remember Day 1?)
# max_tokens: caps our output so we control cost and length
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=200,
    messages=[
        {
            "role": "system",
            "content": "You are an expert copywriter for HoReCa and F&B suppliers."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# BLOCK 5 — Print the output
print("=== PRODUCT DESCRIPTION ===")
print(response.choices[0].message.content)

# BLOCK 6 — Token usage — your live cost dashboard
# This is exactly what we discussed in Day 1 — tokens = cost + capacity
print("\n=== TOKEN USAGE ===")
print(f"Input tokens:  {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens:  {response.usage.total_tokens}")

print("\n=== COST ===")
print(f"Model: Llama 3 8B via Groq")
print(f"Cost this call: $0.00 (free tier)")