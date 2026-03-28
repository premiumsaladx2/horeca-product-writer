import streamlit as st
from groq import Groq
import os

# PAGE SETUP
# This is what appears in the browser tab
st.set_page_config(page_title="HoReCa Product Writer", page_icon="🫒")

# HEADER
# st.title and st.write render text on the web page
st.title("🫒 HoReCa Product Description Generator")
st.write("Generate professional supplier catalog copy in seconds using AI.")

# INPUT FORM
# These create the text boxes users fill in
# This data goes into your context window — just like we discussed in Day 1
st.subheader("Enter Product Details")

product_name = st.text_input("Product Name", placeholder="e.g. Extra Virgin Olive Oil")
category = st.text_input("Category", placeholder="e.g. Cooking Oils")
origin = st.text_input("Origin", placeholder="e.g. Spain")
pack_size = st.text_input("Pack Size", placeholder="e.g. 5L tin")

# TONE SELECTOR
# This changes the temperature and system prompt based on user choice
tone = st.radio("Tone", ["Professional", "Premium & Luxurious", "Simple & Direct"])

# MAP TONE TO TEMPERATURE
# Remember Day 1 — temperature controls creativity
tone_map = {
    "Professional": 0.7,
    "Premium & Luxurious": 0.9,
    "Simple & Direct": 0.3
}
temperature = tone_map[tone]

# GENERATE BUTTON
if st.button("✨ Generate Description"):

    # Check all fields are filled
    if not all([product_name, category, origin, pack_size]):
        st.warning("Please fill in all fields before generating.")
    else:
        # Show a spinner while the API call happens
        with st.spinner("Writing your description..."):

            # Connect to Groq — same as product_writer.py
            client = Groq(api_key=os.environ["GROQ_API_KEY"])

            # Build the prompt — same context window design as before
            prompt = f"""
Write a compelling product description for a HoReCa supplier catalog.

Product: {product_name}
Category: {category}
Origin: {origin}
Pack size: {pack_size}
Tone: {tone}

Keep it under 80 words. Focus on quality, use case, and appeal to 
professional kitchen buyers.
"""

            # API call — same structure as product_writer.py
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=temperature,
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

            description = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        # DISPLAY OUTPUT
        st.subheader("Your AI-Generated Description")
        st.success(description)

        # TOKEN DASHBOARD — Day 1 concepts made visible
        st.subheader("Token Usage")
        col1, col2, col3 = st.columns(3)
        col1.metric("Input Tokens", input_tokens)
        col2.metric("Output Tokens", output_tokens)
        col3.metric("Total Tokens", input_tokens + output_tokens)

        st.caption(f"Temperature used: {temperature} ({tone})")
        st.caption("Powered by Llama 3 via Groq | Free tier")