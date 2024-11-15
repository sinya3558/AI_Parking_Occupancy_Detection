import json
import os
from collections import Counter
from glob import glob

# Initialize a Counter to keep track of class occurrences
class_counts_seg = Counter()
class_counts_3d = Counter()

# Path to the root of your dataset containing subdirectories with .json files
dataset_path = '/../AI_Parking_Occupancy_Detection/datasets/labels/indoor/mid_small'

# Recursively find all .json files within the dataset directory
json_files = glob(os.path.join(dataset_path, '**', 'label/*.json'), recursive=True)

# Loop through each .json file
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # Extract class labels from each JSON file
        # Adjust 'annotations' and 'class' based on your JSON structure
        for item in data.get("bbox3d", []):
            class_label_3d = item.get("name")
            if class_label_3d:
                class_counts_3d[class_label_3d] += 1

        for item in data.get("segmentation", []):
            class_label_seg = item.get("name")
            if class_label_seg:
                class_counts_seg[class_label_seg] += 1

# Display the results
for class_label_3d, count in class_counts_3d.items():
    print(f"Class '{class_label_3d}' in 3dbox: {count}")

for class_label_seg, count in class_counts_seg.items():
    print(f"Class '{class_label_seg}' in seg: {count}")
