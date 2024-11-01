import os
import sys
import numpy as np
import streamlit as st

# Add the src directory to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(SRC_DIR)

# Import functions from src
from query_handler import generate_response
from data_processing import load_data
from vector_store import vectorize_data, find_most_similar, get_embedding

# Define paths within the frontend directory
EMBEDDINGS_PATH = os.path.join(CURRENT_DIR, "embeddings", "vectorized_faqs.npy")
DATA_PATH = os.path.join(CURRENT_DIR, "data", "faq_data.json")

# Check if the FAQ data file exists
if not os.path.exists(DATA_PATH):
    st.error("FAQ data file not found. Please ensure 'faq_data.json' is located in the 'data' directory.")
else:
    # Generate embeddings if the file doesn't exist
    if not os.path.exists(EMBEDDINGS_PATH):
        st.write("Embeddings file not found. Generating embeddings...")
        data = load_data(DATA_PATH)  # Load data from JSON
        vectorized_data = vectorize_data(data)  # Vectorize the data
        np.save(EMBEDDINGS_PATH, vectorized_data)  # Save as .npy file in the embeddings folder
    else:
        vectorized_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)  # Load existing embeddings

    # Streamlit app code
    st.title("RAG Chatbot")
    st.write("Ask a question about our e-commerce platform.")

    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if query:
            query_embedding = get_embedding(query)  # Generate embedding for the user query
            similar_items = find_most_similar(query_embedding, vectorized_data, top_n=1)
            response = similar_items[0]["answer"]  # Get the most similar answer
            st.write("**Response:**", response)
        else:
            st.write("Please enter a question.")
