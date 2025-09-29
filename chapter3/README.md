# Chapter 3: Sequential Local Mapping

This chapter implements a complete LiDAR odometry system using sequential local mapping, building upon the ICP foundation from Chapter 2 to create a robust pose estimation pipeline.

## Learning Objectives

- Build a complete LiDAR odometry pipeline
- Implement local mapping for improved robustness
- Learn keyframe-based mapping strategies
- Master sequential pose estimation with motion models

## What You'll Learn

### 1. Sequential Local Mapping
- Local map construction and management
- Keyframe selection and maintenance
- Voxel-based map optimization
- Motion prediction models

### 2. Complete Odometry Pipeline
- Frame-to-map registration vs. frame-to-frame
- Adaptive correspondence thresholds
- Robust outlier rejection
- Real-time performance considerations

### 3. Advanced Visualization
- 3D trajectory visualization
- Local map point cloud display
- Coordinate frame representation
- Interactive Open3D viewer

## Files in This Chapter

- `run_sequential_local_mapping.py` - Complete local mapping system
- `visualize_sequential_mapping.py` - Advanced Open3D visualization
- `point_to_plane_icp.py` - Enhanced ICP implementation
- `se3_utils.py` - SE(3) pose utilities (from Chapter 2)
- `point_cloud_utils.py` - Point cloud processing utilities
- `gauss_newton_solver.py` - Gauss-Newton optimization
- `data/` - Extended KITTI sequence (11 frames)

## How to Run

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run complete local mapping system
python run_sequential_local_mapping.py

# Launch 3D visualization
python visualize_sequential_mapping.py
```

## System Architecture

### LocalMap Class
- **Feature Map**: Accumulated point cloud from all keyframes
- **Voxel Downsampling**: 0.1m resolution for efficiency
- **Crop Box**: Local region management for scalability
- **Add/Update**: Keyframe integration methods

### SequentialLocalMapper Class
- **Motion Prediction**: Velocity-based initial guess
- **Frame Processing**: ICP registration against local map
- **Map Management**: Keyframe addition and optimization
- **Trajectory Tracking**: Complete pose history

## Key Features

### 1. Local Mapping Strategy
- **All-Keyframe Approach**: Every frame becomes a keyframe
- **Incremental Updates**: Efficient map growth
- **Voxel Grid Filtering**: Memory and computation optimization
- **Spatial Cropping**: Focus on relevant local region

### 2. Robust Registration
- **Frame-to-Map**: Register against accumulated features
- **Motion Prediction**: Use velocity model for initialization
- **Adaptive Parameters**: Dynamic threshold adjustment
- **Convergence Monitoring**: Quality assurance checks

### 3. Advanced Visualization
- **Height-Colored Point Cloud**: Z-axis color mapping
- **Trajectory Path**: Green-to-red gradient visualization
- **Coordinate Frames**: Pose orientation display
- **Start/End Markers**: Clear trajectory boundaries

## Expected Results

### Processing Output:
```
================================================================================
Running Sequential Local Mapping...
================================================================================
Loaded 11 point clouds

Frame  0: ✓ | Map:  3084 pts | Time:   45.2ms
Frame  1: ✓ | Map:  5932 pts | Time:   52.1ms
Frame  2: ✓ | Map:  8756 pts | Time:   48.7ms
...
Frame 10: ✓ | Map: 32438 pts | Time:   61.3ms

Mapping completed in 1.25 seconds
Final map size: 32438 points
Success rate: 11/11
```

### Trajectory Summary:
```
============================================================
TRAJECTORY SUMMARY
============================================================
Total poses: 11
Total distance: 1.267 m
Bounding box:
  X: -0.013 to 1.083 m (1.096 m)
  Y: -0.189 to 0.125 m (0.314 m)
  Z: -0.008 to 0.031 m (0.039 m)
```

## Performance Metrics

- **Success Rate**: 100% (11/11 frames)
- **Average Processing Time**: ~50ms per frame
- **Final Map Size**: ~32K points
- **Memory Footprint**: Optimized with voxel filtering
- **Trajectory Accuracy**: Smooth, consistent motion

## Visualization Features

### 3D Scene Components:
- **Local Map**: Height-colored point cloud (32K+ points)
- **Trajectory**: Smooth path with color gradient
- **Pose Markers**: Coordinate frames every 2nd frame
- **Boundary Markers**: Green start, red end spheres
- **Interactive Controls**: Full Open3D viewer capabilities

### Viewer Controls:
- **Mouse Drag**: Rotate 3D view
- **Mouse Wheel**: Zoom in/out
- **Ctrl+Drag**: Pan view
- **'H'**: Show help menu
- **'Q'**: Exit viewer

## Technical Implementation

### Motion Model:
```python
# Velocity-based prediction
if len(trajectory) >= 2:
    velocity = trajectory[-1] * trajectory[-2].inverse()
    predicted_pose = velocity * trajectory[-1]
```

### Local Map Updates:
```python
# Voxel filtering and integration
new_points = transform_and_downsample(scan, pose)
local_map.add_points(new_points)
local_map.crop_to_region(current_pose, radius=20.0)
```

### ICP Registration:
```python
# Frame-to-map alignment
success, pose, info = icp.align(
    source=current_scan,
    target=local_map.feature_map,
    initial_guess=predicted_pose
)
```

## Algorithm Advantages

1. **Robustness**: Local map provides more features than single frames
2. **Accuracy**: Accumulated evidence improves registration quality
3. **Consistency**: Reduces drift through map-based constraints
4. **Scalability**: Voxel filtering maintains computational efficiency

## Troubleshooting

**Performance Issues:**
- Adjust voxel grid size for memory/accuracy trade-off
- Modify crop box radius for computational efficiency
- Tune max correspondence distance for robustness

**Registration Failures:**
- Check motion prediction parameters
- Verify point cloud quality and density
- Adjust ICP convergence criteria

## Next Steps

This completes the basic LiDAR odometry pipeline! For production systems, consider:

1. **Loop Closure**: Detect and correct accumulated drift
2. **Global Optimization**: Bundle adjustment or pose graph optimization
3. **Real-time Performance**: C++ implementation for speed
4. **Sensor Fusion**: IMU integration for robustness

## Comparison with Chapter 2

| Feature | Chapter 2 (ICP) | Chapter 3 (Local Mapping) |
|---------|-----------------|---------------------------|
| Registration | Frame-to-frame | Frame-to-map |
| Robustness | Moderate | High |
| Accuracy | Good | Excellent |
| Computational | Lower | Higher |
| Memory Usage | Minimal | Moderate |
| Drift Accumulation | Higher | Lower |

The local mapping approach provides significantly improved robustness and accuracy at the cost of increased computational complexity.
