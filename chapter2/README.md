# Chapter 2: Point-to-Plane ICP Implementation

This chapter implements the Iterative Closest Point (ICP) algorithm using point-to-plane distance minimization for accurate pose estimation between LiDAR scans.

## Learning Objectives

- Understand the ICP algorithm and its variants
- Implement point-to-plane distance minimization
- Learn SE(3) pose representation and optimization
- Master Gauss-Newton optimization for pose estimation

## What You'll Learn

### 1. ICP Algorithm Fundamentals
- Point-to-point vs. point-to-plane ICP
- Correspondence finding and outlier rejection
- Iterative optimization process

### 2. SE(3) Pose Mathematics
- Rotation matrices and translation vectors
- SE(3) group operations and transformations
- Pose composition and inversion

### 3. Optimization Theory
- Gauss-Newton method for nonlinear least squares
- Jacobian computation for pose parameters
- Convergence criteria and numerical stability

## Files in This Chapter

- `run_icp.py` - Main ICP demonstration script
- `point_to_plane_icp.py` - Core ICP implementation
- `se3_utils.py` - SE(3) pose utilities
- `point_cloud_utils.py` - Point cloud processing utilities
- `gauss_newton_solver.py` - Gauss-Newton optimization
- `requirements.txt` - Python dependencies
- `data/` - Sample KITTI point cloud sequences (11 frames)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run single ICP alignment
python run_icp.py

# Run sequential ICP processing
python run_sequential_icp.py
```

## Key Components

### Point-to-Plane ICP (`point_to_plane_icp.py`)
- Correspondence finding using KD-tree
- Normal vector computation for planes
- Point-to-plane distance calculation
- Iterative alignment optimization

### SE(3) Utilities (`se3_utils.py`)
- SE(3) class for pose representation
- Transformation matrix operations
- Pose composition and inversion
- Translation and rotation extraction

### Gauss-Newton Solver (`gauss_newton_solver.py`)
- Nonlinear least squares optimization
- Jacobian computation
- Step size control and convergence checking

## Algorithm Overview

1. **Initialize**: Start with identity transformation
2. **Find Correspondences**: Match points between source and target clouds
3. **Compute Residuals**: Calculate point-to-plane distances
4. **Optimize**: Use Gauss-Newton to minimize residuals
5. **Iterate**: Repeat until convergence

## Expected Results

### Single Frame ICP:
```
=== ICP Results ===
Converged: Yes
Iterations: 15
Final RMSE: 0.089 m
Translation: [-0.032, 0.145, -0.008] m
Rotation: Small incremental rotation
```

### Sequential Processing:
```
Frame  0 -> 1: ✓ | RMSE: 0.089m | 15 iter
Frame  1 -> 2: ✓ | RMSE: 0.076m | 12 iter
Frame  2 -> 3: ✓ | RMSE: 0.094m | 14 iter
...
Success Rate: 100% (10/10)
```

## Performance Parameters

- **Max Iterations**: 50
- **Convergence Threshold**: 1e-6
- **Max Correspondence Distance**: 1.0m
- **Voxel Grid Size**: 0.1m (for downsampling)

## Troubleshooting

**Common Issues:**
- **Poor Convergence**: Adjust max correspondence distance or initial guess
- **Slow Performance**: Use voxel downsampling to reduce point count
- **Memory Issues**: Process smaller point cloud chunks

**Parameter Tuning:**
- Increase max_iterations for difficult alignments
- Adjust correspondence_threshold for noisy data
- Modify convergence_tolerance for precision/speed trade-offs

## Next Steps

After mastering ICP alignment, Chapter 3 will show you how to build a complete sequential local mapping system by chaining multiple ICP operations together.

## Mathematical Background

The point-to-plane ICP minimizes the objective function:
```
E = Σ [(pi - (R*qi + t)) · ni]²
```

Where:
- `pi`: Points in target cloud
- `qi`: Corresponding points in source cloud  
- `R, t`: Rotation and translation to estimate
- `ni`: Normal vectors at target points

## Visualization Features

- Source cloud (red) and target cloud (green)
- Correspondence lines between matched points
- Transformation visualization with coordinate frames
- Convergence plots showing RMSE over iterations
