#!/usr/bin/env python3
"""
@file      point_to_plane_icp.py
@brief     Point-to-Plane ICP Implementation
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 2: Point-to-Plane ICP
Python version of the C++ ICP algorithm using Gauss-Newton optimization.
Implements iterative closest point algorithm with point-to-plane error metric
for accurate 3D point cloud registration.
"""

import numpy as np
from typing import List, Tuple, Optional
from se3_utils import SE3
from point_cloud_utils import (
    PointCloud, 
    find_correspondences_point_to_plane,
    load_pcd_file
)
from gauss_newton_solver import GaussNewtonSolver, RobustGaussNewtonSolver, HuberLoss


class ICPConfig:
    """Configuration for ICP algorithm"""
    
    def __init__(self):
        self.max_iterations = 50
        self.translation_tolerance = 1e-4
        self.rotation_tolerance = 1e-4
        self.max_correspondence_distance = 1.0
        self.min_correspondence_points = 10
        self.outlier_rejection_ratio = 0.9
        self.use_robust_loss = True
        self.robust_loss_delta = 0.1


class ICPStatistics:
    """Statistics for ICP convergence"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.iterations_used = 0
        self.initial_cost = 0.0
        self.final_cost = 0.0
        self.converged = False
        self.correspondences_count = 0
        self.inlier_count = 0
        self.match_ratio = 0.0


class PointToPlaneICP:
    """Point-to-plane ICP using Gauss-Newton optimization"""
    
    def __init__(self, config: ICPConfig = None):
        """
        Initialize ICP
        
        Args:
            config: ICP configuration
        """
        self.config = config or ICPConfig()
        self.statistics = ICPStatistics()
        
        # Initialize solvers
        if self.config.use_robust_loss:
            huber_loss = HuberLoss(delta=self.config.robust_loss_delta)
            self.solver = RobustGaussNewtonSolver(
                max_iterations=10,  # Inner iterations
                tolerance=1e-6,
                loss_function=huber_loss
            )
        else:
            self.solver = GaussNewtonSolver(max_iterations=10, tolerance=1e-6)
    
    def align(self, source: PointCloud, target: PointCloud, 
              initial_guess: SE3 = None) -> Tuple[bool, SE3]:
        """
        Align source to target point cloud
        
        Args:
            source: Source point cloud
            target: Target point cloud  
            initial_guess: Initial pose estimate
            
        Returns:
            Tuple of (converged, final_pose)
        """
        if source.empty() or target.empty():
            print("Error: Input clouds are empty")
            return False, SE3()
        
        # Initialize
        self.statistics.reset()
        current_pose = initial_guess or SE3()
        
        print(f"Starting ICP with {source.size()} source and {target.size()} target points")
        
        prev_cost = float('inf')
        
        for iteration in range(self.config.max_iterations):
            # Transform source cloud with current pose
            transformed_source = source.copy()
            transformed_source.transform(current_pose)
            
            # Find correspondences
            correspondences = find_correspondences_point_to_plane(
                transformed_source,
                target,
                max_distance=self.config.max_correspondence_distance,
                k_neighbors=5
            )
            
            num_correspondences = len(correspondences)
            self.statistics.correspondences_count = num_correspondences
            
            if num_correspondences < self.config.min_correspondence_points:
                print(f"Insufficient correspondences: {num_correspondences} < {self.config.min_correspondence_points}")
                break
            
            # Reject outliers
            num_inliers = self._reject_outliers(correspondences)
            self.statistics.inlier_count = num_inliers
            self.statistics.match_ratio = num_inliers / source.size()
            
            if num_inliers < self.config.min_correspondence_points:
                print(f"Insufficient inliers: {num_inliers} < {self.config.min_correspondence_points}")
                break
            
            # Convert correspondences back to original source points (untransformed)
            # This is important for the solver which expects source points in original frame
            original_correspondences = []
            for i, corr in enumerate(correspondences):
                if corr['valid']:
                    # Find corresponding original source point
                    # We need to map back from transformed correspondence to original source
                    original_corr = corr.copy()
                    # The source point should be the original untransformed point
                    # We can find it by inverse transforming or by tracking indices
                    # For now, we'll use a simple approach - find the closest original point
                    original_corr['source_point'] = self._find_original_source_point(
                        corr['source_point'], source, current_pose
                    )
                    original_correspondences.append(original_corr)
            
            # Optimize pose using Gauss-Newton
            optimized_pose, solver_stats = self.solver.solve_point_to_plane_icp(
                original_correspondences, current_pose
            )
            
            if iteration == 0:
                self.statistics.initial_cost = solver_stats['initial_cost']
            
            # Check convergence
            converged = self._check_convergence(current_pose, optimized_pose)
            
            # Calculate cost change
            current_cost = solver_stats['final_cost']
            cost_change = abs(prev_cost - current_cost)
            
            print(f"ICP Iteration {iteration}: correspondences={num_correspondences}, "
                  f"inliers={num_inliers}, cost={current_cost:.6f}, "
                  f"match_ratio={self.statistics.match_ratio:.3f}")
            
            if converged or (iteration > 0 and cost_change < 1e-6):
                self.statistics.converged = True
                break
            
            # Update pose and continue
            current_pose = optimized_pose
            prev_cost = current_cost
            self.statistics.iterations_used = iteration + 1
        
        self.statistics.final_cost = prev_cost if prev_cost != float('inf') else 0.0
        
        print(f"ICP completed: iterations={self.statistics.iterations_used}, "
              f"converged={self.statistics.converged}, final_cost={self.statistics.final_cost:.6f}")
        
        return self.statistics.converged, current_pose
    
    def _find_original_source_point(self, transformed_point: np.ndarray, 
                                  source: PointCloud, pose: SE3) -> np.ndarray:
        """
        Find the original source point corresponding to a transformed point
        
        Args:
            transformed_point: Point in transformed space
            source: Original source cloud
            pose: Current pose
            
        Returns:
            Original source point
        """
        # Inverse transform the point
        pose_inv = pose.inverse()
        original_point = pose_inv * transformed_point
        
        # Find closest point in original source cloud
        distances = np.linalg.norm(source.points - original_point, axis=1)
        closest_idx = np.argmin(distances)
        
        return source.points[closest_idx]
    
    def _reject_outliers(self, correspondences: List[dict]) -> int:
        """
        Reject outliers based on distance
        
        Args:
            correspondences: List of correspondences
            
        Returns:
            Number of inliers
        """
        if not correspondences:
            return 0
        
        # Sort by distance
        correspondences.sort(key=lambda x: x['distance'])
        
        # Keep only the best correspondences
        num_keep = int(len(correspondences) * self.config.outlier_rejection_ratio)
        
        # Mark outliers as invalid
        for i in range(len(correspondences)):
            correspondences[i]['valid'] = i < num_keep
            if correspondences[i]['valid']:
                correspondences[i]['weight'] = 1.0
        
        return num_keep
    
    def _check_convergence(self, prev_pose: SE3, current_pose: SE3) -> bool:
        """
        Check if ICP has converged
        
        Args:
            prev_pose: Previous pose
            current_pose: Current pose
            
        Returns:
            True if converged
        """
        # Calculate pose difference
        pose_diff = prev_pose.inverse() * current_pose
        
        # Translation change
        translation_change = np.linalg.norm(pose_diff.translation)
        
        # Rotation change (magnitude of rotation vector)
        rotation_vector = pose_diff.to_tangent()[3:]  # Get rotation part
        rotation_change = np.linalg.norm(rotation_vector)
        
        translation_converged = translation_change < self.config.translation_tolerance
        rotation_converged = rotation_change < self.config.rotation_tolerance
        
        return translation_converged and rotation_converged
    
    def get_statistics(self) -> ICPStatistics:
        """Get ICP statistics"""
        return self.statistics


def run_icp_example():
    """Example of running ICP on two point clouds"""
    
    # Load point clouds
    print("Loading point clouds...")
    source_cloud = load_pcd_file("/home/eugene/source/lidar_odometry_from_scratch/chapter2/data/00000.pcd")
    target_cloud = load_pcd_file("/home/eugene/source/lidar_odometry_from_scratch/chapter2/data/00010.pcd")
    
    if source_cloud.empty() or target_cloud.empty():
        print("Error: Could not load point clouds")
        return
    
    print(f"Source cloud: {source_cloud.size()} points")
    print(f"Target cloud: {target_cloud.size()} points")
    
    # Downsample for faster processing
    print("Downsampling point clouds...")
    voxel_size = 0.1
    source_cloud = source_cloud.downsample_voxel(voxel_size)
    target_cloud = target_cloud.downsample_voxel(voxel_size)
    
    print(f"After downsampling - Source: {source_cloud.size()}, Target: {target_cloud.size()}")
    
    # Configure ICP
    config = ICPConfig()
    config.max_iterations = 50
    config.max_correspondence_distance = 1.0
    config.use_robust_loss = True
    config.robust_loss_delta = 0.1
    
    # Create ICP instance
    icp = PointToPlaneICP(config)
    
    # Set initial guess (identity transform)
    initial_guess = SE3()
    
    # Run ICP
    print("\nRunning ICP...")
    converged, final_pose = icp.align(source_cloud, target_cloud, initial_guess)
    
    # Print results
    print(f"\nICP Results:")
    print(f"Converged: {converged}")
    print(f"Final pose:")
    print(f"  Translation: {final_pose.translation}")
    print(f"  Rotation matrix:")
    print(f"{final_pose.rotation}")
    
    stats = icp.get_statistics()
    print(f"\nStatistics:")
    print(f"  Iterations: {stats.iterations_used}")
    print(f"  Initial cost: {stats.initial_cost:.6f}")
    print(f"  Final cost: {stats.final_cost:.6f}")
    print(f"  Final match ratio: {stats.match_ratio:.3f}")
    
    return final_pose, stats


if __name__ == "__main__":
    run_icp_example()
