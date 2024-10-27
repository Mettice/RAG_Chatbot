import openai
import numpy as np
import os
from scipy.spatial.distance import cosine
from data_processing import load_data  # Assuming you have a data loading function
import streamlit as st
from dotenv import load_dotenv  # Import dotenv to load environment variables

# Load environment variables from .env for local development
load_dotenv()

# Set the OpenAI API key from st.secrets for deployment, or .env for local
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Debugging to check where the API key is being loaded from
if st.secrets.get("OPENAI_API_KEY"):
    st.write("API Key found in st.secrets.")
elif os.getenv("OPENAI_API_KEY"):
    st.write("API Key found in .env file.")
else:
    st.error("OpenAI API key not found. Ensure it's set in secrets.toml for deployment or in .env locally.")

# Stop execution if API key is missing
if not openai.api_key:
    st.stop()

def get_embedding(text):
    """Generate an embedding for a given text using OpenAI's API."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Ensure model compatibility
        input=text
    )
    return np.array(response['data'][0]['embedding'])

def vectorize_data(data):
    """Vectorizes the questions in the dataset."""
    embeddings = []
    for entry in data:
        question = entry["question"]
        embedding = get_embedding(question)
        embeddings.append({
            "question": question,
            "answer": entry["answer"],
            "embedding": embedding
        })
    return embeddings

def find_most_similar(query_embedding, embeddings, top_n=1):
    """Find the most similar question based on cosine similarity."""
    query_embedding = np.ravel(query_embedding)  # Ensure query embedding is 1-D
    similarities = []
    for item in embeddings:
        item_embedding = np.ravel(item["embedding"])  # Ensure item embedding is 1-D
        similarity = 1 - cosine(query_embedding, item_embedding)
        similarities.append((item, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in similarities[:top_n]]

if __name__ == "__main__":
    # Load the FAQ data
    data = load_data('../frontend/data/faq_data.json')

    # Vectorize the data
    vectorized_data = vectorize_data(data)

    # Save embeddings as .npy file
    np.save("../frontend/embeddings/vectorized_faqs.npy", vectorized_data)
    print("Embeddings created and stored successfully.")
