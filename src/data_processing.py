import json
import os


def load_data(file_path):
    """Load JSON data from a file and handle file-related errors."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError("The JSON file is malformed. Please ensure it has a valid JSON format.")

    return data


if __name__ == "__main__":
    # Define the path to your JSON file
    data_path = '../frontend/data/faq_data.json'

    # Attempt to load data
    try:
        data = load_data(data_path)
        print("Loaded Data:", data)
    except (FileNotFoundError, ValueError) as e:
        print("Error:", e)
