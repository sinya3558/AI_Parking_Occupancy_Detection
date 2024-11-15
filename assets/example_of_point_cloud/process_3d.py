import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

# Function to generate the 3D bounding box coordinates from the dimensions, location, and rotation
def generate_3d_bbox(dimensions, location, rotation_z):
    # Bounding box dimensions: [length, width, height]
    length, width, height = dimensions
    # Location: [x, y, z]
    x, y, z = location

    # Rotation: Only around the z-axis for this example
    rotation_matrix = np.array([
        [math.cos(rotation_z), -math.sin(rotation_z), 0],
        [math.sin(rotation_z), math.cos(rotation_z), 0],
        [0, 0, 1]
    ])

    # 8 corners of the bounding box before rotation
    corners = np.array([[-length/2, -width/2, -height/2],
                        [ length/2, -width/2, -height/2],
                        [ length/2,  width/2, -height/2],
                        [-length/2,  width/2, -height/2],
                        [-length/2, -width/2, height/2],
                        [ length/2, -width/2, height/2],
                        [ length/2,  width/2, height/2],
                        [-length/2,  width/2, height/2]])

    # Apply rotation to corners
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Translate the bounding box
    rotated_corners += np.array(location)

    return rotated_corners

# Load the JSON file
json_data = '''{
    "image": "00000010.jpg",
    "point_cloud": "00000010.pcd",
    "meta": {
        "time": "Day",
        "environment": "Indoor",
        "weather": "None",
        "place": "Small and Medium",
        "city": "Gwangju",
        "terrain": "Urban",
        "road_type": "ge 4m",
        "road_material": "Paved"
    },
    "calib": {
        "intrinsic": [
            [ 1397.991371 , 0.0 , 960.0 ],
            [ 0.0 , 1397.991371 , 540.0 ],
            [ 0.0 , 0.0 , 1.0 ]
        ],
        "rotation": [ -0.96517669 , -1.502906215 , 2.560259924 ],
        "translation": [ 0.13620099 , 1.34389899 , 1.35197656 ]
    },
    "bbox2d": [],
    "segmentation": [],
    "bbox3d": [
        {
            "name": "Car",
            "dimensions": [ 4.97 , 1.91 , 1.73 ],
            "location": [ 5.16 , 3.35 , 0.9 ],
            "rotation_z": -1.69 },
        {
            "name": "Car",
            "dimensions": [ 4.61 , 1.82 , 1.73 ],
            "location": [ -7.05 , 5.1 , 0.88 ],
            "rotation_z": -1.81 },
        {
            "name": "Car",
            "dimensions": [ 4.79 , 1.79 , 1.6 ],
            "location": [ -13.18 , -5.01 , 0.82 ],
            "rotation_z": 1.45 },
        {
            "name": "Car",
            "dimensions": [ 4.64 , 1.94 , 1.73 ],
            "location": [ 7.64 , 2.99 , 0.97 ],
            "rotation_z": -1.69 },
        {
            "name": "Car",
            "dimensions": [ 4.45 , 1.88 , 1.73 ],
            "location": [ 0.11 , 3.71 , 0.91 ],
            "rotation_z": -1.69 }
    ]
}'''

data = json.loads(json_data)

# Load the .pcd (Point Cloud Data) file
pcd_path = data["point_cloud"]
pcd = o3d.io.read_point_cloud(pcd_path)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Generate 3D bounding boxes and add them to the point cloud visualization
for bbox in data["bbox3d"]:
    dimensions = bbox["dimensions"]
    location = bbox["location"]
    rotation_z = bbox["rotation_z"]

    corners = generate_3d_bbox(dimensions, location, rotation_z)

    # Create a mesh to visualize the bounding box (=line segments connecting the corners)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for bounding box edges

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the bounding box along with the point cloud
    o3d.visualization.draw_geometries([pcd, line_set])
