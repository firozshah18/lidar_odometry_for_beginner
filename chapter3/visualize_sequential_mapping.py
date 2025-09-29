#!/usr/bin/env python3
"""
@file      visualize_sequential_mapping.py
@brief     Open3D Visualization for Sequential Local Mapping
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 3: Sequential Local Mapping Visualization with Open3D
Run mapping first, then visualize trajectory and local map with Open3D.
"""

import numpy as np
import open3d as o3d
import time
from typing import List

from se3_utils import SE3
from point_cloud_utils import PointCloud
from run_sequential_local_mapping import SequentialLocalMapper, load_pcd_sequence


def run_mapping() -> tuple:
    """Run sequential local mapping and return results"""
    
    print("="*80)
    print("Running Sequential Local Mapping...")
    print("="*80)
    
    # Load data
    data_dir = "data"
    point_clouds = load_pcd_sequence(data_dir, max_frames=11)
    
    if len(point_clouds) < 1:
        print("Error: Need at least 1 point cloud")
        return None, None, None
    
    print(f"Loaded {len(point_clouds)} point clouds")
    
    # Process all frames
    mapper = SequentialLocalMapper()
    results = []
    
    start_time = time.time()
    for i, cloud in enumerate(point_clouds):
        success, pose, result = mapper.process_frame(i, cloud)
        results.append(result)
        
        # Show progress without full details
        status = "✓" if success else "✗"
        print(f"Frame {i:2d}: {status} | Map: {result['map_size']:5d} pts | Time: {result['computation_time_ms']:6.1f}ms")
    
    total_time = time.time() - start_time
    
    print(f"\nMapping completed in {total_time:.2f} seconds")
    print(f"Final map size: {mapper.local_map.size()} points")
    print(f"Success rate: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    return mapper, results, mapper.get_trajectory()


def create_open3d_visualization(mapper: SequentialLocalMapper, trajectory: List[SE3]):
    """Create Open3D visualization"""
    
    print("\nCreating Open3D visualization...")
    
    geometries = []
    
    # 1. Create local map point cloud
    if not mapper.local_map.feature_map.empty():
        map_points = mapper.local_map.feature_map.points
        map_pcd = o3d.geometry.PointCloud()
        map_pcd.points = o3d.utility.Vector3dVector(map_points)
        
        # Color map points by height (Z coordinate)
        z_values = map_points[:, 2]
        z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
        colors = np.zeros((len(map_points), 3))
        colors[:, 0] = z_normalized  # Red channel for height
        colors[:, 1] = 0.3  # Fixed green
        colors[:, 2] = 1.0 - z_normalized  # Blue inverse of height
        map_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        geometries.append(map_pcd)
        print(f"Added local map: {len(map_points)} points")
    
    # 2. Create trajectory line
    if len(trajectory) > 1:
        trajectory_points = np.array([pose.translation for pose in trajectory])
        trajectory_lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
        
        trajectory_line_set = o3d.geometry.LineSet()
        trajectory_line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        trajectory_line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)
        
        # Color trajectory from green (start) to red (end)
        line_colors = []
        for i in range(len(trajectory_lines)):
            t = i / (len(trajectory_lines) - 1) if len(trajectory_lines) > 1 else 0
            color = [t, 1.0 - t, 0.0]  # Green to red gradient
            line_colors.append(color)
        trajectory_line_set.colors = o3d.utility.Vector3dVector(line_colors)
        
        # Make trajectory line thicker by adding cylinder representation
        trajectory_cylinder = o3d.geometry.LineSet()
        trajectory_cylinder.points = o3d.utility.Vector3dVector(trajectory_points)
        trajectory_cylinder.lines = o3d.utility.Vector2iVector(trajectory_lines)
        trajectory_cylinder.colors = o3d.utility.Vector3dVector(line_colors)
        
        geometries.append(trajectory_line_set)
        print(f"Added trajectory: {len(trajectory_points)} poses")
    
    # 3. Create pose markers (coordinate frames)
    frame_size = 0.3  # Increased from 0.15 to make axes more visible
    for i, pose in enumerate(trajectory[::2]):  # Every 2nd frame to avoid clutter
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose.to_matrix())
        geometries.append(frame)
    
    print(f"Added {len(trajectory[::2])} coordinate frames")
    
    # 4. Create start and end markers
    if len(trajectory) > 0:
        # Start marker (green sphere)
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)  # Increased size
        start_sphere.paint_uniform_color([0, 1, 0])  # Green
        start_translation = trajectory[0].translation.reshape(3, 1)
        start_sphere.translate(start_translation)
        geometries.append(start_sphere)
        
        # End marker (red sphere)
        if len(trajectory) > 1:
            end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)  # Increased size
            end_sphere.paint_uniform_color([1, 0, 0])  # Red
            end_translation = trajectory[-1].translation.reshape(3, 1)
            end_sphere.translate(end_translation)
            geometries.append(end_sphere)
    
    return geometries


def print_trajectory_summary(trajectory: List[SE3]):
    """Print trajectory summary"""
    
    print(f"\n{'='*60}")
    print("TRAJECTORY SUMMARY")
    print(f"{'='*60}")
    
    if not trajectory:
        print("No trajectory to display")
        return
    
    positions = np.array([pose.translation for pose in trajectory])
    
    # Calculate total distance
    total_distance = 0.0
    for i in range(1, len(positions)):
        total_distance += np.linalg.norm(positions[i] - positions[i-1])
    
    # Calculate bounding box
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    
    print(f"Total poses: {len(trajectory)}")
    print(f"Total distance: {total_distance:.3f} m")
    print(f"Bounding box:")
    print(f"  X: {min_pos[0]:.3f} to {max_pos[0]:.3f} m ({max_pos[0]-min_pos[0]:.3f} m)")
    print(f"  Y: {min_pos[1]:.3f} to {max_pos[1]:.3f} m ({max_pos[1]-min_pos[1]:.3f} m)")
    print(f"  Z: {min_pos[2]:.3f} to {max_pos[2]:.3f} m ({max_pos[2]-min_pos[2]:.3f} m)")
    
    # Print some poses
    print(f"\nKey poses:")
    print(f"{'Frame':<6} {'X':<8} {'Y':<8} {'Z':<8}")
    print("-" * 35)
    indices_to_show = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, len(trajectory)-1]
    for i in indices_to_show:
        if i < len(trajectory):
            pos = trajectory[i].translation
            print(f"{i:<6} {pos[0]:<8.3f} {pos[1]:<8.3f} {pos[2]:<8.3f}")


def main():
    """Main function"""
    
    # Step 1: Run mapping
    mapper, results, trajectory = run_mapping()
    
    if mapper is None:
        return
    
    # Step 2: Print summary
    print_trajectory_summary(trajectory)
    
    # Step 3: Create visualization
    geometries = create_open3d_visualization(mapper, trajectory)
    
    # Step 4: Launch Open3D viewer
    print(f"\n{'='*60}")
    print("LAUNCHING OPEN3D VIEWER")
    print(f"{'='*60}")
    print("Visualization includes:")
    print("  • Local map point cloud (colored by height)")
    print("  • Trajectory path (green→red gradient)")
    print("  • Coordinate frames at poses")
    print("  • Start (green) and end (red) markers")
    print("  • Reference ground plane")
    print("\nControls:")
    print("  • Mouse drag: Rotate view")
    print("  • Mouse wheel: Zoom in/out")
    print("  • Ctrl+Mouse drag: Pan view")
    print("  • 'H': Show help")
    print("  • 'Q' or close window: Exit")
    
    input("\nPress Enter to launch Open3D viewer...")
    
    # Configure viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Sequential Local Mapping - Chapter 3", 
                      width=1400, height=900, left=100, top=100)
    
    # Add all geometries
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Set nice viewing angle
    view_control = vis.get_view_control()
    view_control.set_up([0, 0, 1])  # Z up (correct orientation)
    view_control.set_front([0.5, 0.5, 0.7])  # Look from above-front angle (positive Z)
    view_control.set_lookat([0.6, 0.1, 0])  # Look at middle of trajectory
    view_control.set_zoom(0.7)  # Zoom out a bit for better overview
    
    # Run viewer
    vis.run()
    vis.destroy_window()
    
    print("\nVisualization completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
