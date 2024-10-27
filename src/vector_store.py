import openai
import numpy as np
import os
from scipy.spatial.distance import cosine
from data_processing import load_data  # Assuming you have a data loading function

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    """Generate an embedding for a given text using OpenAI's API."""
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
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
    data = load_data('../data/faq_data.json')

    # Vectorize the data
    vectorized_data = vectorize_data(data)

    # Save embeddings as .npy file
    np.save("../embeddings/vectorized_faqs.npy", vectorized_data)
    print("Embeddings created and stored successfully.")