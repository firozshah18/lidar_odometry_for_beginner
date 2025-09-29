#!/usr/bin/env python3
"""
@file      visualize_lidar.py
@brief     LiDAR point cloud visualization using Open3D
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 1: LiDAR Point Cloud Visualization
This code visualizes two PCD files using Open3D:
- 00000.pcd: displayed in green
- 00010.pcd: displayed in yellow
"""

import open3d as o3d
import numpy as np
import os

def load_and_colorize_pcd(pcd_path, color):
    """
    Load a PCD file and apply the specified color.
    
    Args:
        pcd_path (str): Path to the PCD file
        color (list): RGB color values [R, G, B] (range 0-1)
    
    Returns:
        o3d.geometry.PointCloud: Colored point cloud
    """
    # Load PCD file
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Check number of points
    num_points = len(pcd.points)
    print(f"Loaded points ({os.path.basename(pcd_path)}): {num_points}")
    
    # Apply the same color to all points
    colors = np.tile(color, (num_points, 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def main():
    # Set data path
    data_dir = "data"
    pcd1_path = os.path.join(data_dir, "00000.pcd")
    pcd2_path = os.path.join(data_dir, "00010.pcd")
    
    # Check file existence
    if not os.path.exists(pcd1_path):
        print(f"Error: Cannot find {pcd1_path} file.")
        return
    
    if not os.path.exists(pcd2_path):
        print(f"Error: Cannot find {pcd2_path} file.")
        return
    
    print("Loading PCD files and applying colors...")
    
    # Load PCD files and apply colors
    # 00000.pcd in green
    pcd1 = load_and_colorize_pcd(pcd1_path, [0.0, 1.0, 0.0])  # Pure green
    
    # 00010.pcd in yellow  
    pcd2 = load_and_colorize_pcd(pcd2_path, [1.0, 1.0, 0.0])  # Yellow
    
    # Combine two point clouds
    combined_pcd = pcd1 + pcd2
    
    print("\nStarting visualization...")
    print("- Green: 00000.pcd")
    print("- Yellow: 00010.pcd")
    print("- Mouse: rotate/zoom available")
    print("- ESC or close window to exit")
    
    # Visualization with navy background
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="LiDAR Point Cloud Visualization - Chapter 1",
        width=1200,
        height=800,
        left=100,
        top=100
    )
    
    # Set dark navy background color (RGB: 0, 0, 0.2)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.2])  # Darker navy blue
    
    vis.add_geometry(combined_pcd)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
