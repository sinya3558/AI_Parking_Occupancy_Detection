import os
import json

# Directory of JSON files
base_dir = '/../AI_Parking_Occupancy_Detection/datasets/labels/indoor/mid_small'

# Traverse all subdirectories under 'mid_small', where all data(.json & .jpg) is located
for root, _, files in os.walk(base_dir):
    for filename in files:
        if filename.endswith('.json'):
            file_path = os.path.join(root, filename)

            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Keep only the 'segmentation' field if it exists
            filtered_data = {'segmentation': data.get('segmentation', [])}

            # Save the modified JSON file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(filtered_data, file, indent=4)

print("Filtering is done!")
