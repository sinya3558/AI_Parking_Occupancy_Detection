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

def project_segmentation_and_bbox(json_path, image_path, output_path):
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract segmentation polygons and bounding boxes
    segmentation = data.get("segmentation", [])
    bbox2d = data.get("bbox2d", [])

    # Open the image
    image = Image.open(image_path)

    # Create a plot to display the image with overlays
    fig, ax = plt.subplots()
    ax.imshow(image)

    # To track used positions and avoid overlap
    used_positions = []

    # Function to adjust text position to avoid overlaps
    def adjust_position(x, y):
        offset_step = 5  # Pixels to adjust per step
        for _ in range(20):  # Limit the number of adjustments
            if not any(abs(x - px) < offset_step and abs(y - py) < offset_step for px, py in used_positions):
                used_positions.append((x, y))
                return x, y
            y -= offset_step  # Move the text upwards to avoid overlap
        return x, y  # Default to current position if no space found

    # Draw each polygon in the segmentation data
    for segment in segmentation:
        polygon = segment.get("polygon", [])
        if polygon:
            # Convert polygon to a list of (x, y) tuples for matplotlib
            polygon_tuples = [tuple(coord) for coord in polygon]

            # Determine color based on the name field
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
            centroid_x, centroid_y = adjust_position(centroid_x, centroid_y)
            ax.text(centroid_x, centroid_y, name, color=color, fontsize=10, ha='center')

    # Draw each bounding box in the bbox2d data
    for box in bbox2d:
        bbox = box.get("bbox", [])
        if bbox and len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            # Determine color based on the name field
            name = box.get("name", "Unknown")
            if name == "Parking Block":
                color = 'orange'
            elif name == "Disabled Parking Space":
                color = 'purple'
            elif name == "Parking Sign":
                color = 'cyan'
            else:
                color = 'yellow'  # Default color for other categories

            # Create the rectangle patch
            rect = patches.Rectangle((x_min, y_min), width, height, edgecolor=color, fill=False, linewidth=1)
            ax.add_patch(rect)

            # Annotate with the box's name (smaller font size for bbox2d)
            text_x, text_y = adjust_position(x_min + width / 2, y_min - 5)
            ax.text(text_x, text_y, name, color='black', fontsize=5, ha='center')

    plt.axis('off')  # Hide axes

    # Save the figure instead of showing it
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)  # Close the plot to free memory

# Example usage: loop through JSON files and find corresponding images
for json_filename in os.listdir(json_dir):
    if json_filename.endswith('.json'):
        json_path = os.path.join(json_dir, json_filename)
        image_filename = json_filename.replace('.json', '.jpg')  # Assuming corresponding image has the same name but with .jpg
        image_path = os.path.join(image_dir, image_filename)

        if os.path.exists(image_path):
            output_path = os.path.join(output_dir, f"seg_2d_{image_filename}")
            project_segmentation_and_bbox(json_path, image_path, output_path)
        else:
            print(f"Image file {image_filename} not found.")
