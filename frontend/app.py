import streamlit as st
import numpy as np
import os
import sys

# Adjust the path to import query_handler from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from query_handler import generate_response  # Import after setting the path

# Load vectorized data
vectorized_data = np.load("../embeddings/vectorized_faqs.npy", allow_pickle=True)

# Streamlit app title
st.title("RAG Chatbot")
st.write("Ask a question about our e-commerce platform.")

# Input box for user query
query = st.text_input("Enter your question:")

# Button to get an answer
if st.button("Get Answer"):
    if query:
        response = generate_response(query, vectorized_data)
        st.write("**Response:**", response)
    else:
        st.write("Please enter a question.")
