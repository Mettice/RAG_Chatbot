import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    # Load data
    data = load_data('../frontend/data/faq_data.json')
    print("Loaded Data:", data)
