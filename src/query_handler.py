import openai
import numpy as np
import os
from vector_store import find_most_similar, get_embedding

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_response(query, vectorized_data):
    """
    Generate a response using ChatGPT based on the most similar FAQ entry retrieved.
    """
    # Generate the query embedding
    query_embedding = get_embedding(query)

    # Retrieve the most similar FAQ entry
    retrieved_document = find_most_similar(query_embedding, vectorized_data)[0]
    question = retrieved_document["question"]
    answer = retrieved_document["answer"]

    # Prepare messages for the ChatGPT API call
    messages = [
        {"role": "system", "content": "You are a helpful and concise customer support assistant."},
        {"role": "user", "content": f"A user asked: '{query}'."},
        {"role": "user", "content": f"The most relevant information we have is: '{answer}'."},
        {"role": "user",
         "content": "Please provide an accurate and specific response, focusing only on the essential details to address the user's question directly."}
    ]

    # Call ChatGPT API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100
    )
    return response['choices'][0]['message']['content'].strip()


if __name__ == "__main__":
    # Load vectorized data
    vectorized_data = np.load("../frontend/embeddings/vectorized_faqs.npy", allow_pickle=True)

    # List of test queries
    test_queries = [
        "How do I return a product?",
        "What payment methods can I use?",
        "How long is shipping?",
        "Can I get a refund?",
        "Is there a warranty on items?"
    ]

    # Iterate through each query and print the response
    for query in test_queries:
        print(f"Query: {query}")
        response = generate_response(query, vectorized_data)
        print("Response:", response)
        print("-" * 50)
