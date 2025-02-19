import json

def load_result(file_path):
    def custom_decoder(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if value == "Infinity":
                    obj[key] = np.inf
                elif isinstance(value, dict):
                    obj[key] = custom_decoder(value)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if value == "Infinity":
                    obj[i] = np.inf
        
                elif isinstance(value, dict):
                    obj[i] = custom_decoder(value)
        return obj

    # Loading the JSON file with custom decoding
    with open(file_path, "r") as json_file:
        loaded_results = json.load(json_file, object_hook=custom_decoder)

    return loaded_results

json_data = load_result("../CleanData/aggregated_summary.json")

def extract_keys_and_shape(data):
    """
    Extracts the keys and the overall shape of a JSON-like dictionary.
    
    Parameters:
    data (dict): The JSON-like dictionary to analyze.
    
    Returns:
    tuple: A tuple containing the keys and the shape of the dictionary.
    """
    keys = set()
    shape = []

    def traverse(obj, path=[]):
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.add('.'.join(path + [key]))
                traverse(value, path + [key])
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                traverse(item, path + [str(index)])
        else:
            keys.add('.'.join(path))

    traverse(data)
    shape = [len(data["runs"])]

    return keys, tuple(shape)

keys, shape = extract_keys_and_shape(json_data)

print("Keys:")
for key in keys:
    print(key)

print("\nShape:", shape)