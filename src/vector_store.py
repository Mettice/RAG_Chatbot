import openai
import numpy as np
import streamlit as st
from scipy.spatial.distance import cosine
from data_processing import load_data

# Load API key from Streamlit secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Debugging to confirm API key loading
if openai.api_key:
    st.write("API Key successfully loaded from Streamlit secrets.")
else:
    st.error("OpenAI API key not found. Ensure it’s set in Streamlit Cloud's Secrets settings.")
    st.stop()  # Stop execution if API key is missing

def get_embedding(text):
    """Generate an embedding for a given text using OpenAI's API."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
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
    query_embedding = np.ravel(query_embedding)
    similarities = []
    for item in embeddings:
        item_embedding = np.ravel(item["embedding"])
        similarity = 1 - cosine(query_embedding, item_embedding)
        similarities.append((item, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in similarities[:top_n]]

if __name__ == "__main__":
    data = load_data('../frontend/data/faq_data.json')
    vectorized_data = vectorize_data(data)
    np.save("../frontend/embeddings/vectorized_faqs.npy", vectorized_data)
    print("Embeddings created and stored successfully.")
