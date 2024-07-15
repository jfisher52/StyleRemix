import json
import os

def save_to_jsonl(data, filename):
    """
    Saves the data to a JSONL (JSON Lines) file, creating directories if they don't exist.

    Each item in the data list is serialized to JSON and written as a single line in the file.

    Args:
        data (list): A list of dictionaries or serializable objects to be saved.
        filename (str): The path to the output JSONL file.

    Example:
        data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        save_to_jsonl(data, 'output/data.jsonl')

    This will save the following content to 'output/data.jsonl':
        {"name": "Alice", "age": 30}
        {"name": "Bob", "age": 25}
    """
    # Extract the directory part from the filename
    directory = os.path.dirname(filename)
    
    # If the directory part is not empty and does not exist, create it
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Proceed to save the data to the file
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_record = json.dumps(item, ensure_ascii=False)
            f.write(json_record + '\n')

def read_from_jsonl(filename):
    """
    Reads data from a JSONL (JSON Lines) file and returns a list of items.

    Each line in the file is deserialized from JSON and added to the list.

    Args:
        filename (str): The path to the input JSONL file.

    Returns:
        list: A list of dictionaries or objects read from the JSONL file.

    Example:
        data = read_from_jsonl('output/data.jsonl')

    If 'output/data.jsonl' contains:
        {"name": "Alice", "age": 30}
        {"name": "Bob", "age": 25}

    The returned list will be:
        [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
