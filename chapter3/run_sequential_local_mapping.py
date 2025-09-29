#!/usr/bin/env python3
"""
@file      run_sequential_local_mapping.py
@brief     Sequential Local Mapping Demo - Build local map incrementally
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 3: Sequential Local Mapping Demo
Process frames sequentially while building and maintaining a local map.
All frames are treated as keyframes for maximum map density.
"""

import numpy as np
import time
import os
import glob
from typing import List, Tuple, Optional

from se3_utils import SE3
from point_cloud_utils import PointCloud, load_pcd_file
from point_to_plane_icp import PointToPlaneICP, ICPConfig


class LocalMap:
    """Local map management class"""
    
    def __init__(self, map_voxel_size: float = 0.2, max_range: float = 50.0):
        self.map_voxel_size = map_voxel_size
        self.max_range = max_range
        self.feature_map = PointCloud()
        self.keyframes = []
        
    def add_keyframe(self, cloud: PointCloud, pose: SE3) -> None:
        """Add new keyframe to local map"""
        if cloud.empty():
            print("Warning: Empty cloud for keyframe")
            return
            
        # Transform cloud to world coordinates
        world_cloud = cloud.copy()
        world_cloud.transform(pose.to_matrix())
        
        # Add to feature map
        self.feature_map.add_points_from(world_cloud)
        
        # Store keyframe info
        keyframe_info = {
            'pose': pose,
            'cloud_size': cloud.size(),
            'world_cloud_size': world_cloud.size()
        }
        self.keyframes.append(keyframe_info)
        
        # Apply voxel grid downsampling
        before_voxel = self.feature_map.size()
        self.feature_map = self.feature_map.downsample_voxel(self.map_voxel_size)
        after_voxel = self.feature_map.size()
        
        # Apply crop box filter around current pose
        current_pos = pose.translation
        crop_radius = self.max_range * 1.2
        
        min_bound = current_pos - crop_radius
        max_bound = current_pos + crop_radius
        
        self.feature_map = self.feature_map.crop_box(min_bound, max_bound)
        after_crop = self.feature_map.size()
        
        print(f"  Local map updated: {before_voxel} -> {after_voxel} -> {after_crop} points")
    
    def get_local_features_in_frame(self, current_pose: SE3) -> PointCloud:
        """Get local map features transformed to current frame coordinates"""
        if self.feature_map.empty():
            return PointCloud()
            
        # Transform from world to current frame
        local_features = self.feature_map.copy()
        
        # T_lw = T_wl^-1
        T_lw = current_pose.inverse().to_matrix()
        local_features.transform(T_lw)
        
        return local_features
    
    def size(self) -> int:
        """Get current map size"""
        return self.feature_map.size()
    
    def num_keyframes(self) -> int:
        """Get number of keyframes"""
        return len(self.keyframes)


class SequentialLocalMapper:
    """Sequential local mapping processor"""
    
    def __init__(self):
        # Configuration
        self.voxel_size = 0.4
        self.map_voxel_size = 0.2
        self.max_range = 50.0
        
        # ICP configuration
        self.icp_config = ICPConfig()
        self.icp_config.max_iterations = 30
        self.icp_config.translation_tolerance = 1e-3
        self.icp_config.rotation_tolerance = 1e-3
        self.icp_config.max_correspondence_distance = 1.0
        self.icp_config.min_correspondence_points = 100
        self.icp_config.outlier_rejection_ratio = 0.8
        self.icp_config.use_robust_loss = True
        self.icp_config.robust_loss_delta = 0.1
        
        # State
        self.local_map = LocalMap(self.map_voxel_size, self.max_range)
        self.trajectory = []
        self.current_pose = SE3()  # World pose
        self.velocity = SE3()      # Velocity model
        self.initialized = False
        
        # Statistics
        self.total_processing_time = 0.0
        self.total_icp_time = 0.0
        self.frame_count = 0
    
    def preprocess_cloud(self, cloud: PointCloud) -> PointCloud:
        """Preprocess point cloud with downsampling"""
        if cloud.empty():
            return cloud
        
        # Downsample using voxel grid
        processed_cloud = cloud.downsample_voxel(self.voxel_size)
        return processed_cloud
    
    def process_frame(self, frame_idx: int, raw_cloud: PointCloud) -> Tuple[bool, SE3, dict]:
        """Process single frame with local mapping"""
        
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing Frame {frame_idx:05d}")
        print(f"{'='*60}")
        print(f"Raw points: {raw_cloud.size()}")
        
        # Step 1: Preprocess frame
        feature_cloud = self.preprocess_cloud(raw_cloud)
        print(f"Preprocessing: {raw_cloud.size()} -> {feature_cloud.size()} points")
        
        if feature_cloud.empty():
            print("Error: Empty processed cloud")
            return False, SE3(), {}
        
        if not self.initialized:
            return self._initialize_first_frame(frame_idx, feature_cloud, start_time)
        
        # Step 2: Motion prediction using velocity model
        predicted_pose = self.current_pose * self.velocity
        
        print(f"Motion prediction applied:")
        print(f"  Translation: {np.linalg.norm(self.velocity.translation):.4f} m")
        print(f"  Rotation: {np.degrees(np.linalg.norm(self.velocity.log()[3:])):.2f} deg")
        
        # Step 3: Get local map features in current frame coordinates
        local_map_features = self.local_map.get_local_features_in_frame(predicted_pose)
        
        if local_map_features.empty():
            print("Warning: No local map features available")
            # Just add current frame as keyframe
            self.current_pose = predicted_pose
            self._add_keyframe(feature_cloud)
            self.trajectory.append(self.current_pose)
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            result = {
                'success': True,
                'converged': True,
                'iterations': 0,
                'initial_cost': 0.0,
                'final_cost': 0.0,
                'cost_reduction': 0.0,
                'correspondences': 0,
                'inliers': 0,
                'match_ratio': 0.0,
                'computation_time_ms': total_time,
                'translation_distance': np.linalg.norm(predicted_pose.translation),
                'rotation_angle_deg': np.degrees(np.linalg.norm(predicted_pose.log()[3:])),
                'map_size': self.local_map.size(),
                'keyframes': self.local_map.num_keyframes()
            }
            
            return True, self.current_pose, result
        
        print(f"Local map: {local_map_features.size()} points")
        
        # Step 4: ICP alignment
        icp_start = time.time()
        success, relative_pose, icp_stats = self._run_icp(feature_cloud, local_map_features)
        icp_end = time.time()
        icp_time = (icp_end - icp_start) * 1000  # milliseconds
        
        self.total_icp_time += icp_time
        
        if not success:
            print("Warning: ICP failed, using motion prediction")
            final_pose = predicted_pose
        else:
            # Convert relative pose to world pose
            final_pose = predicted_pose * relative_pose
        
        # Step 5: Update velocity model (exponential smoothing)
        if len(self.trajectory) > 0:
            previous_pose = self.trajectory[-1]
            current_velocity = previous_pose.inverse() * final_pose
            
            # Smooth velocity update
            alpha = 0.7
            if len(self.trajectory) > 1:
                # Smoothing for translation
                prev_trans = self.velocity.translation
                curr_trans = current_velocity.translation
                smooth_trans = alpha * curr_trans + (1.0 - alpha) * prev_trans
                
                # Smoothing for rotation (in tangent space)
                prev_rot_tangent = self.velocity.log()[3:]
                curr_rot_tangent = current_velocity.log()[3:]
                smooth_rot_tangent = alpha * curr_rot_tangent + (1.0 - alpha) * prev_rot_tangent
                
                # Combine smoothed components
                smooth_tangent = np.concatenate([smooth_trans, smooth_rot_tangent])
                self.velocity = SE3.from_tangent(smooth_tangent)
            else:
                self.velocity = current_velocity
        
        # Step 6: Update pose and trajectory
        self.current_pose = final_pose
        self.trajectory.append(self.current_pose)
        
        # Step 7: Add as keyframe (all frames are keyframes)
        self._add_keyframe(feature_cloud)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # milliseconds
        self.total_processing_time += total_time
        self.frame_count += 1
        
        # Create result statistics
        result = dict(icp_stats)  # Copy ICP stats
        result.update({
            'computation_time_ms': total_time,
            'icp_time_ms': icp_time,
            'translation_distance': np.linalg.norm(final_pose.translation),
            'rotation_angle_deg': np.degrees(np.linalg.norm(final_pose.log()[3:])),
            'map_size': self.local_map.size(),
            'keyframes': self.local_map.num_keyframes()
        })
        
        print(f"Frame processed: pose translation=({final_pose.translation[0]:.2f}, {final_pose.translation[1]:.2f}, {final_pose.translation[2]:.2f})")
        print(f"Statistics: time={total_time:.1f}ms, map_size={self.local_map.size()}")
        
        return success, final_pose, result
    
    def _initialize_first_frame(self, frame_idx: int, feature_cloud: PointCloud, start_time: float) -> Tuple[bool, SE3, dict]:
        """Initialize with first frame"""
        print("Initializing first frame...")
        
        # Set identity pose for first frame
        self.current_pose = SE3()
        self.velocity = SE3()
        
        # Add first keyframe
        self._add_keyframe(feature_cloud)
        self.trajectory.append(self.current_pose)
        
        self.initialized = True
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        result = {
            'success': True,
            'converged': True,
            'iterations': 0,
            'initial_cost': 0.0,
            'final_cost': 0.0,
            'cost_reduction': 0.0,
            'correspondences': 0,
            'inliers': 0,
            'match_ratio': 0.0,
            'computation_time_ms': total_time,
            'icp_time_ms': 0.0,
            'translation_distance': 0.0,
            'rotation_angle_deg': 0.0,
            'map_size': self.local_map.size(),
            'keyframes': self.local_map.num_keyframes()
        }
        
        print(f"First frame initialized: {feature_cloud.size()} features, map_size={self.local_map.size()}")
        
        return True, self.current_pose, result
    
    def _run_icp(self, source_cloud: PointCloud, target_cloud: PointCloud) -> Tuple[bool, SE3, dict]:
        """Run ICP between source and target clouds"""
        
        # Create ICP solver
        icp = PointToPlaneICP(self.icp_config)
        
        # Initial guess (identity)
        initial_pose = SE3()
        
        print(f"Running ICP: {source_cloud.size()} source, {target_cloud.size()} target points")
        
        # Run ICP
        success, final_pose = icp.align(source_cloud, target_cloud, initial_pose)
        
        # Get statistics
        stats = icp.get_statistics()
        
        # Create result dictionary
        result = {
            'success': success,
            'converged': stats.converged,
            'iterations': stats.iterations_used,
            'initial_cost': stats.initial_cost,
            'final_cost': stats.final_cost,
            'cost_reduction': stats.initial_cost - stats.final_cost if stats.initial_cost > 0 else 0,
            'correspondences': stats.correspondences_count,
            'inliers': stats.inlier_count,
            'match_ratio': stats.match_ratio
        }
        
        print(f"ICP result: success={success}, iterations={stats.iterations_used}, cost={stats.final_cost:.6f}")
        
        return success, final_pose, result
    
    def _add_keyframe(self, feature_cloud: PointCloud) -> None:
        """Add current frame as keyframe to local map"""
        self.local_map.add_keyframe(feature_cloud, self.current_pose)
    
    def get_trajectory(self) -> List[SE3]:
        """Get full trajectory"""
        return self.trajectory.copy()
    
    def get_statistics(self) -> dict:
        """Get processing statistics"""
        if self.frame_count == 0:
            return {}
        
        return {
            'total_frames': self.frame_count,
            'total_keyframes': self.local_map.num_keyframes(),
            'map_size': self.local_map.size(),
            'avg_processing_time_ms': self.total_processing_time / self.frame_count,
            'avg_icp_time_ms': self.total_icp_time / max(self.frame_count - 1, 1),  # Exclude first frame
            'total_processing_time_s': self.total_processing_time / 1000,
            'trajectory_length': len(self.trajectory)
        }


def load_pcd_sequence(data_dir: str, max_frames: int = 11) -> List[PointCloud]:
    """Load sequence of PCD files"""
    pcd_pattern = os.path.join(data_dir, "*.pcd")
    pcd_files = sorted(glob.glob(pcd_pattern))
    
    if not pcd_files:
        print(f"Error: No PCD files found in {data_dir}")
        return []
    
    print(f"Loading {min(len(pcd_files), max_frames)} PCD files...")
    
    point_clouds = []
    for i, pcd_file in enumerate(pcd_files[:max_frames]):
        basename = os.path.basename(pcd_file)
        print(f"  Loading {basename}...", end="")
        
        cloud = load_pcd_file(pcd_file)
        if cloud.empty():
            print(f" FAILED")
            continue
        
        print(f" {cloud.size()} points")
        point_clouds.append(cloud)
    
    return point_clouds


def main():
    """Main function for sequential local mapping demo"""
    
    print("="*80)
    print("Sequential Local Mapping Demo - Chapter 3")
    print("All frames are treated as keyframes")
    print("="*80)
    
    # Load PCD sequence
    data_dir = "data"
    point_clouds = load_pcd_sequence(data_dir, max_frames=11)
    
    if len(point_clouds) < 1:
        print("Error: Need at least 1 point cloud")
        return
    
    print(f"\nLoaded {len(point_clouds)} point clouds")
    
    # Create sequential mapper
    mapper = SequentialLocalMapper()
    
    # Process frames sequentially
    results = []
    total_start_time = time.time()
    
    for i, cloud in enumerate(point_clouds):
        success, pose, result = mapper.process_frame(i, cloud)
        results.append(result)
        
        status = "SUCCESS ✓" if success else "FAILED ✗"
        print(f"\nFrame {i:05d}: {status}")
        print(f"  Map size: {result['map_size']} points")
        print(f"  Keyframes: {result['keyframes']}")
        print(f"  Processing time: {result['computation_time_ms']:.1f} ms")
        if 'icp_time_ms' in result:
            print(f"  ICP time: {result['icp_time_ms']:.1f} ms")
        
        print(f"{'='*60}")
        
        # Small delay for readability
        time.sleep(0.2)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL LOCAL MAPPING SUMMARY")
    print(f"{'='*80}")
    
    successful_frames = sum(1 for r in results if r['success'])
    failed_frames = len(results) - successful_frames
    
    print(f"Total frames processed: {len(results)}")
    print(f"Successful frames: {successful_frames}")
    print(f"Failed frames: {failed_frames}")
    print(f"Success rate: {successful_frames/len(results)*100:.1f}%")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    # Get mapper statistics
    stats = mapper.get_statistics()
    if stats:
        print(f"\nProcessing Statistics:")
        print(f"  Total keyframes: {stats['total_keyframes']}")
        print(f"  Final map size: {stats['map_size']} points")
        print(f"  Average processing time: {stats['avg_processing_time_ms']:.1f} ms/frame")
        print(f"  Average ICP time: {stats['avg_icp_time_ms']:.1f} ms/frame")
        print(f"  Total trajectory length: {stats['trajectory_length']} poses")
    
    # Print trajectory
    trajectory = mapper.get_trajectory()
    print(f"\nTrajectory Summary:")
    print(f"{'Frame':<6} {'X':<8} {'Y':<8} {'Z':<8} {'Distance':<10}")
    print("-" * 50)
    
    for i, pose in enumerate(trajectory):
        trans = pose.translation
        distance = np.linalg.norm(trans) if i > 0 else 0.0
        print(f"{i:<6} {trans[0]:<8.3f} {trans[1]:<8.3f} {trans[2]:<8.3f} {distance:<10.3f}")
    
    # Calculate total trajectory distance
    total_distance = 0.0
    for i in range(1, len(trajectory)):
        diff = trajectory[i].translation - trajectory[i-1].translation
        total_distance += np.linalg.norm(diff)
    
    print(f"\nTotal trajectory distance: {total_distance:.3f} m")
    print(f"Demo completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
