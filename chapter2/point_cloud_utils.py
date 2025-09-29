#!/usr/bin/env python3
"""
@file      point_cloud_utils.py
@brief     Point Cloud Utilities for LiDAR processing
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 2: Point Cloud Utilities
Provides functions for loading, processing, and manipulating point clouds.
Includes correspondence finding, outlier rejection, and geometric operations
for point-to-plane ICP implementation.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional
from scipy.spatial import KDTree


class PointCloud:
    """Simple point cloud class"""
    
    def __init__(self, points=None):
        """
        Initialize point cloud
        
        Args:
            points: Nx3 numpy array of points
        """
        if points is None:
            self.points = np.empty((0, 3))
        else:
            self.points = np.array(points, dtype=np.float64)
    
    def size(self):
        """Get number of points"""
        return len(self.points)
    
    def empty(self):
        """Check if point cloud is empty"""
        return self.size() == 0
    
    def transform(self, transformation):
        """Transform point cloud with SE3 transformation"""
        from se3_utils import SE3
        
        if isinstance(transformation, SE3):
            if not self.empty():
                # Apply transformation to all points
                transformed_points = []
                for point in self.points:
                    transformed_point = transformation * point
                    transformed_points.append(transformed_point)
                self.points = np.array(transformed_points)
        elif isinstance(transformation, np.ndarray) and transformation.shape == (4, 4):
            # Handle 4x4 matrix directly
            if not self.empty():
                homogeneous_points = np.hstack([self.points, np.ones((len(self.points), 1))])
                transformed_homogeneous = (transformation @ homogeneous_points.T).T
                self.points = transformed_homogeneous[:, :3]
        else:
            raise TypeError("Transformation must be SE3 or 4x4 matrix")
    
    def downsample_voxel(self, voxel_size):
        """Downsample point cloud using voxel grid"""
        if self.empty():
            return PointCloud()
        
        # Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # Downsample
        downsampled = pcd.voxel_down_sample(voxel_size)
        
        # Convert back
        return PointCloud(np.asarray(downsampled.points))
    
    def copy(self):
        """Create a copy of the point cloud"""
        return PointCloud(self.points.copy())


def load_pcd_file(filepath: str) -> PointCloud:
    """
    Load point cloud from PCD file
    
    Args:
        filepath: Path to PCD file
        
    Returns:
        PointCloud object
    """
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        return PointCloud(points)
    except Exception as e:
        print(f"Error loading PCD file {filepath}: {e}")
        return PointCloud()


def find_correspondences_knn(source: PointCloud, 
                            target: PointCloud, 
                            k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find correspondences using K-nearest neighbors
    
    Args:
        source: Source point cloud
        target: Target point cloud
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (source_indices, target_indices, distances)
    """
    if source.empty() or target.empty():
        return np.array([]), np.array([]), np.array([])
    
    # Build KD-tree for target cloud
    kdtree = KDTree(target.points)
    
    source_indices = []
    target_indices = []
    distances = []
    
    for i, point in enumerate(source.points):
        dists, indices = kdtree.query(point, k=min(k, len(target.points)))
        
        # If we get multiple neighbors, take the closest one
        if isinstance(indices, np.ndarray):
            closest_idx = indices[0]
            closest_dist = dists[0]
        else:
            closest_idx = indices
            closest_dist = dists
            
        source_indices.append(i)
        target_indices.append(closest_idx)
        distances.append(closest_dist)
    
    return np.array(source_indices), np.array(target_indices), np.array(distances)


def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to a set of points using SVD
    
    Args:
        points: Nx3 array of points
        
    Returns:
        Tuple of (plane_normal, plane_point)
    """
    if len(points) < 3:
        return np.array([0, 0, 1]), np.mean(points, axis=0)
    
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # SVD to find the normal
    _, _, Vt = np.linalg.svd(centered_points)
    normal = Vt[-1, :]  # Last row corresponds to smallest singular value
    
    # Normalize
    normal = normal / np.linalg.norm(normal)
    
    return normal, centroid


def is_collinear(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if three points are collinear
    
    Args:
        p1, p2, p3: 3D points
        threshold: Collinearity threshold
        
    Returns:
        True if points are collinear
    """
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Check if cross product is small
    cross = np.cross(v1, v2)
    cross_magnitude = np.linalg.norm(cross)
    
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    if v1_mag < 1e-6 or v2_mag < 1e-6:
        return True
    
    normalized_cross = cross_magnitude / (v1_mag * v2_mag)
    return normalized_cross < threshold


def find_correspondences_point_to_plane(source: PointCloud, 
                                       target: PointCloud, 
                                       max_distance: float = 1.0,
                                       k_neighbors: int = 5) -> List[dict]:
    """
    Find point-to-plane correspondences
    
    Args:
        source: Source point cloud
        target: Target point cloud
        max_distance: Maximum correspondence distance
        k_neighbors: Number of neighbors for plane fitting
        
    Returns:
        List of correspondence dictionaries
    """
    if source.empty() or target.empty():
        return []
    
    # Build KD-tree for target cloud
    kdtree = KDTree(target.points)
    
    correspondences = []
    
    for i, source_point in enumerate(source.points):
        # Find K nearest neighbors in target cloud
        distances, indices = kdtree.query(source_point, k=min(k_neighbors, len(target.points)))
        
        if len(indices) < 5:
            continue
            
        # Get neighbor points
        neighbor_points = target.points[indices]
        
        # Select up to 5 non-collinear points for plane fitting
        selected_points = []
        selected_indices = []
        
        non_collinear_found = False
        for j, (idx, point) in enumerate(zip(indices, neighbor_points)):
            if len(selected_points) < 2:
                selected_points.append(point)
                selected_indices.append(idx)
            elif not non_collinear_found:
                # Check collinearity for the third point
                if is_collinear(selected_points[0], selected_points[1], point, 0.5):
                    continue
                else:
                    non_collinear_found = True
                    selected_points.append(point)
                    selected_indices.append(idx)
            else:
                selected_points.append(point)
                selected_indices.append(idx)
                
            if len(selected_points) >= 5:
                break
        
        if len(selected_points) < 5:
            continue
            
        selected_points = np.array(selected_points)
        
        # Fit plane to selected points
        plane_normal, plane_point = fit_plane_to_points(selected_points)
        
        # Validate plane by checking distance of all points to plane
        plane_valid = True
        for pt in selected_points:
            dist_to_plane = abs(np.dot(plane_normal, pt - plane_point))
            if dist_to_plane > max_distance:
                plane_valid = False
                break
        
        if not plane_valid:
            continue
            
        # Calculate point-to-plane residual
        residual = abs(np.dot(plane_normal, source_point - plane_point))
        
        if residual > max_distance * 3.0:
            continue
            
        # Create correspondence
        correspondence = {
            'source_point': source_point.copy(),
            'target_point': plane_point.copy(),
            'plane_normal': plane_normal.copy(),
            'distance': residual,
            'weight': 1.0,
            'valid': True
        }
        
        correspondences.append(correspondence)
    
    return correspondences
