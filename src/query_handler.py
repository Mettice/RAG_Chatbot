import openai
import streamlit as st
import os
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# Set the OpenAI API key using st.secrets for deployment, or .env for local testing
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Optional: Debugging check for API key
if not openai.api_key:
    st.error("OpenAI API key not found. Ensure it's set in secrets.toml for deployment or in the .env file locally.")

def generate_response(query, vectorized_data):
    """Generate a response using ChatGPT based on the most similar FAQ entry retrieved."""
    query_embedding = get_embedding(query)
    retrieved_document = find_most_similar(query_embedding, vectorized_data)[0]
    answer = retrieved_document["answer"]

    # ChatGPT API call
    messages = [
        {"role": "system", "content": "You are a helpful customer support assistant."},
        {"role": "user", "content": f"A user asked: '{query}'."},
        {"role": "user", "content": f"The most relevant information we have is: '{answer}'."}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100
    )
    return response['choices'][0]['message']['content'].strip()
