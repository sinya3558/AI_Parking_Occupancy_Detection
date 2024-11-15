import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directories for JSON files, images, and where to save output projections
json_dir = '/../AI_Parking_Occupancy_Detection'
image_dir = '/../AI_Parking_Occupancy_Detection'
output_dir = '/../AI_Parking_Occupancy_Detection'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def project_segmentation(json_path, image_path, output_path):
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract the segmentation polygons
    segmentation = data.get("segmentation", [])
    if not segmentation:
        print(f"No segmentation data found in {json_path}.")
        return

    # Open the image
    image = Image.open(image_path)

    # Create a plot to display the image with the polygon overlay
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw each polygon in the segmentation data
    for segment in segmentation:
        polygon = segment.get("polygon", [])
        if polygon:
            # Convert polygon to a list of (x, y) tuples for matplotlib
            polygon_tuples = [tuple(coord) for coord in polygon]

            # Determine color based on the 'name' field
            name = segment.get("name", "Unknown")
            if name == "Driveable Space":
                color = 'blue'
            elif name == "Parking Space":
                color = 'green'
            else:
                color = 'red'  # Default color for other categories

            # Create the polygon patch
            poly = patches.Polygon(polygon_tuples, closed=True, edgecolor=color, fill=False, linewidth=2)
            ax.add_patch(poly)

            # Annotate with the polygon's name
            centroid_x = sum([pt[0] for pt in polygon_tuples]) / len(polygon_tuples)
            centroid_y = sum([pt[1] for pt in polygon_tuples]) / len(polygon_tuples)
            ax.text(centroid_x, centroid_y, name, color='black', fontsize=7, ha='center')

    plt.axis('off')  # Hide axes

    # Save the figure instead of showing it
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)  # Close the plot to free memory

# Loop through JSON files and find corresponding images
for json_filename in os.listdir(json_dir):
    if json_filename.endswith('.json'):
        json_path = os.path.join(json_dir, json_filename)
        image_filename = json_filename.replace('.json', '.jpg')  # Assuming corresponding image has the same name but with .jpg!
        image_path = os.path.join(image_dir, image_filename)

        if os.path.exists(image_path):
            output_path = os.path.join(output_dir, f"projection_{image_filename}")
            project_segmentation(json_path, image_path, output_path)
        else:
            print(f"Image file {image_filename} not found.")
