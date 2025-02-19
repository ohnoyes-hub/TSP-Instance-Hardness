import os
import json

def convert_tsp_to_json(file_path):
    """
    Converts Levi Koppenhol's TSP instances from "Exactly characterizable parameter settings in a crossoverless evolutionary algorithm" to JSON format.
    """
    with open(file_path, "r") as tsp_file:
        lines = tsp_file.readlines()
    
    tsp_json = {}
    cities = []

    for line in lines:
        line = line.strip()

        if line.startswith("AME"): # I assume this is a typo and should be "NAME"
            tsp_json["name"] = line.split(":")[1].strip()
        elif line.startswith("COMMENT"):
            continue
        elif line.startswith("TYPE"):
            tsp_json["type"] = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            tsp_json["DIMENSION"] = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            tsp_json["EDGE_WEIGHT_TYPE"] = line.split(":")[1].strip()
        elif line == "NODE_COORD_SECTION":
            # Next lines contain city coordinates
            continue
        elif line == "EOF":
            break
        else:
            # City coordinates
            parts = line.split(" ")
            city = {"id": int(parts[0]), "x": int(parts[1]), "y": int(parts[2])}
            cities.append(city)

    tsp_json["cities"] = cities
    return tsp_json

def process_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'): # process all .txt files
            file_path = os.path.join(input_directory, filename)
            tsp_json = convert_tsp_to_json(file_path)
            
            # Save the JSON file in the output directory
            json_filename = filename.replace('.txt', '.json')
            json_path = os.path.join(output_directory, json_filename)
            
            with open(json_path, 'w') as json_file:
                json.dump(tsp_json, json_file, indent=4)
            
            print(f"Converted {filename} to {json_filename} and saved in {output_directory}")
            


if __name__ == "__main__":
    input_directory = "./PPATSPParameters/Data"
    output_directory = "./TSPHardener/data/euclidean_tsp/levi_koppenhol"
    process_directory(input_directory, output_directory)

