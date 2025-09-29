#!/usr/bin/env python3
"""
@file      run_icp.py
@brief     Main script to run ICP on PCD files
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 2: ICP Demo
Load PCD files 00000.pcd and 00010.pcd and perform ICP alignment.
Demonstrates point-to-plane ICP implementation with Open3D visualization
of alignment results before and after registration.
"""

import numpy as np
import time
import open3d as o3d

from se3_utils import SE3
from point_cloud_utils import PointCloud, load_pcd_file
from point_to_plane_icp import PointToPlaneICP, ICPConfig
from gauss_newton_solver import HuberLoss

import numpy as np
import time
import open3d as o3d

from se3_utils import SE3
from point_cloud_utils import PointCloud, load_pcd_file
from point_to_plane_icp import PointToPlaneICP, ICPConfig
from gauss_newton_solver import HuberLoss




def load_and_preprocess_data():
    """Load and preprocess PCD files"""
    
    # Load PCD files
    print("Loading PCD files...")
    source_cloud = load_pcd_file("data/00000.pcd")
    target_cloud = load_pcd_file("data/00010.pcd")
    
    print(f"Source cloud: {source_cloud.size()} points")
    print(f"Target cloud: {target_cloud.size()} points")
    
    # Downsample for faster processing
    voxel_size = 0.4
    print(f"Downsampling with voxel size: {voxel_size}")
    
    source_cloud = source_cloud.downsample_voxel(voxel_size)
    target_cloud = target_cloud.downsample_voxel(voxel_size)
    
    print(f"Downsampled source: {source_cloud.size()} points")
    print(f"Downsampled target: {target_cloud.size()} points")
    
    return source_cloud, target_cloud


def visualize_icp_results(source_cloud: PointCloud, 
                          target_cloud: PointCloud, 
                          aligned_cloud: PointCloud,
                          initial_pose: SE3,
                          final_pose: SE3):
    """Visualize ICP results using Open3D"""
    
    print("Creating visualization...")
    
    # Convert PointCloud objects to Open3D format
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source_cloud.points)
    source_o3d.paint_uniform_color([1, 0, 0])  # Red for source
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_cloud.points)
    target_o3d.paint_uniform_color([0, 1, 0])  # Green for target
    
    aligned_o3d = o3d.geometry.PointCloud()
    aligned_o3d.points = o3d.utility.Vector3dVector(aligned_cloud.points)
    aligned_o3d.paint_uniform_color([0, 0, 1])  # Blue for aligned
    
    # Create coordinate frames for poses
    initial_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    initial_frame.transform(initial_pose.to_matrix())
    
    final_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    final_frame.transform(final_pose.to_matrix())
    
    # Show initial alignment (source + target)
    print("Showing initial alignment (Red: Source, Green: Target)")
    print("Press 'Q' to continue to next visualization...")
    o3d.visualization.draw_geometries([source_o3d, target_o3d, initial_frame],
                                     window_name="Initial Alignment",
                                     width=1200, height=800)
    
    # Show final alignment (aligned source + target)  
    print("Showing final alignment (Blue: Aligned Source, Green: Target)")
    print("Press 'Q' to continue...")
    o3d.visualization.draw_geometries([aligned_o3d, target_o3d, final_frame],
                                     window_name="Final Alignment",
                                     width=1200, height=800)
    
    # Show comparison (all clouds)
    print("Showing comparison (Red: Original Source, Blue: Aligned Source, Green: Target)")
    print("Press 'Q' to close...")
    # Make source cloud semi-transparent for better comparison
    source_o3d_transparent = o3d.geometry.PointCloud()
    source_o3d_transparent.points = o3d.utility.Vector3dVector(source_cloud.points)
    source_o3d_transparent.paint_uniform_color([1, 0.5, 0.5])  # Light red
    
    o3d.visualization.draw_geometries([source_o3d_transparent, aligned_o3d, target_o3d, final_frame],
                                     window_name="ICP Comparison",
                                     width=1200, height=800)


def print_pose_info(pose: SE3, name: str):
    """Print pose information"""
    print(f"\n{name}:")
    print(f"  Translation: [{pose.translation[0]:.4f}, {pose.translation[1]:.4f}, {pose.translation[2]:.4f}]")
    
    # Convert rotation matrix to Euler angles for better understanding
    from scipy.spatial.transform import Rotation as R
    euler = R.from_matrix(pose.rotation).as_euler('xyz', degrees=True)
    print(f"  Rotation (Euler XYZ deg): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")


def main():
    """Main function to run ICP"""
    
    print("="*60)
    print("Point-to-Plane ICP Implementation")
    print("="*60)
    
    # Load and preprocess data
    source_cloud, target_cloud = load_and_preprocess_data()
    
    # Configure ICP
    config = ICPConfig()
    config.max_iterations = 50  # Reduced from 50
    config.translation_tolerance = 1e-3
    config.rotation_tolerance = 1e-3
    config.max_correspondence_distance = 1.0
    config.min_correspondence_points = 100
    config.outlier_rejection_ratio = 0.8
    config.use_robust_loss = True
    config.robust_loss_delta = 0.1
    
    print(f"\nICP Configuration:")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Translation tolerance: {config.translation_tolerance}")
    print(f"  Rotation tolerance: {config.rotation_tolerance}")
    print(f"  Max correspondence distance: {config.max_correspondence_distance}")
    print(f"  Outlier rejection ratio: {config.outlier_rejection_ratio}")
    print(f"  Use robust loss: {config.use_robust_loss}")
    
    # Create ICP solver
    icp = PointToPlaneICP(config)
    
    # Initial pose (identity - no initial guess)
    initial_pose = SE3()
    print_pose_info(initial_pose, "Initial Pose")
    
    # Run ICP
    print("\n" + "="*60)
    print("Running ICP...")
    start_time = time.time()
    
    success, final_pose = icp.align(source_cloud, target_cloud, initial_pose)
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Get statistics
    stats = icp.get_statistics()
    
    # Print results
    print("="*60)
    print("ICP Results:")
    print(f"  Success: {success}")
    print(f"  Converged: {stats.converged}")
    print(f"  Iterations used: {stats.iterations_used}")
    print(f"  Initial cost: {stats.initial_cost:.6f}")
    print(f"  Final cost: {stats.final_cost:.6f}")
    print(f"  Cost reduction: {stats.initial_cost - stats.final_cost:.6f}")
    print(f"  Correspondences: {stats.correspondences_count}")
    print(f"  Inliers: {stats.inlier_count}")
    print(f"  Match ratio: {stats.match_ratio:.3f}")
    print(f"  Computation time: {elapsed_time:.2f} ms")
    
    print_pose_info(final_pose, "Final Pose")
    
    # Calculate pose difference
    pose_diff = initial_pose.inverse() * final_pose
    translation_norm = np.linalg.norm(pose_diff.translation)
    rotation_angle = np.linalg.norm(pose_diff.log()[3:])  # Rotation part of log
    
    print(f"\nPose Change:")
    print(f"  Translation distance: {translation_norm:.4f} m")
    print(f"  Rotation angle: {np.degrees(rotation_angle):.2f} degrees")
    
    # Create aligned source cloud for visualization
    aligned_cloud = PointCloud(source_cloud.points.copy())
    aligned_cloud.transform(final_pose)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_icp_results(source_cloud, target_cloud, aligned_cloud, initial_pose, final_pose)
    
    # Save results if needed
    print("\nICP alignment completed!")
    
    return success, final_pose, stats


if __name__ == "__main__":
    try:
        success, final_pose, stats = main()
        
        if success:
            print("\n✓ ICP alignment successful!")
        else:
            print("\n✗ ICP alignment failed!")
            
    except Exception as e:
        print(f"\nError during ICP: {e}")
        import traceback
        traceback.print_exc()
