import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Directory containing JSON and images
json_dir = '/../AI_Parking_Occupancy_Detection'
image_dir = '/../AI_Parking_Occupancy_Detection'
output_dir = '/../AI_Parking_Occupancy_Detection'
os.makedirs(output_dir, exist_ok=True)

def rotation_matrix_from_angles(angles):
    """Compute rotation matrix from given angles in radians"""
    rx, ry, rz = angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

# Transform and project 3D points onto a 2D image
def project_points_3d_to_2d(points_3d, intrinsic_matrix, rotation_matrix, translation_vector):
    points_2d = []
    for point in points_3d:
        # Apply rotation and translation
        transformed_point = rotation_matrix @ point + translation_vector

        # Apply intrinsic projection
        projected_point = intrinsic_matrix @ transformed_point

        # Convert to 2D by normalizing
        x = projected_point[0] / projected_point[2]
        y = projected_point[1] / projected_point[2]
        points_2d.append([x, y])
    return points_2d

# Function to get 3D bounding box corners
def get_bbox_corners(dimensions, location, rotation_z):
    dx, dy, dz = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2
    corners = np.array([
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz]
    ])
    rotation_matrix_z = np.array([
        [np.cos(rotation_z), -np.sin(rotation_z), 0],
        [np.sin(rotation_z),  np.cos(rotation_z), 0],
        [0, 0, 1]
    ])
    rotated_corners = corners @ rotation_matrix_z.T + location
    return rotated_corners

def project_3d_bboxes(json_path, image_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load calibration data
    intrinsic = data["calib"]["intrinsic"]
    intrinsic_matrix = np.array(intrinsic)
    rotation_angles = data["calib"]["rotation"]
    translation_vector = np.array(data["calib"]["translation"])

    # Compute the full rotation matrix
    rotation_matrix = rotation_matrix_from_angles(rotation_angles)

    bboxes_3d = data.get("bbox3d", [])
    if not bboxes_3d:
        print("No 3D bounding boxes found.")
        return

    # Open the image and set up for plotting
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox in bboxes_3d:
        dimensions = bbox["dimensions"]
        location = bbox["location"]
        rotation_z = bbox["rotation_z"]

        # Get 3D corners
        bbox_corners_3d = get_bbox_corners(dimensions, location, rotation_z)

        # Project 3D corners to 2D
        bbox_corners_2d = project_points_3d_to_2d(bbox_corners_3d, intrinsic_matrix, rotation_matrix, translation_vector)

        # Draw the projected 2D bounding box
        for i in range(4):
            ax.plot([bbox_corners_2d[i][0], bbox_corners_2d[(i+1)%4][0]],
                    [bbox_corners_2d[i][1], bbox_corners_2d[(i+1)%4][1]], 'r-')
            ax.plot([bbox_corners_2d[i+4][0], bbox_corners_2d[(i+1)%4+4][0]],
                    [bbox_corners_2d[i+4][1], bbox_corners_2d[(i+1)%4+4][1]], 'r-')
            ax.plot([bbox_corners_2d[i][0], bbox_corners_2d[i+4][0]],
                    [bbox_corners_2d[i][1], bbox_corners_2d[i+4][1]], 'r-')

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Example usage
for json_filename in os.listdir(json_dir):
    if json_filename.endswith('.json'):
        json_path = os.path.join(json_dir, json_filename)
        image_filename = json_filename.replace('.json', '.jpg')
        image_path = os.path.join(image_dir, image_filename)

        if os.path.exists(image_path):
            output_path = os.path.join(output_dir, f"3D_{image_filename}")
            project_3d_bboxes(json_path, image_path, output_path)
        else:
            print(f"Image file {image_filename} not found.")
