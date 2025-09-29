# Chapter 1: LiDAR Point Cloud Visualization

This chapter covers how to visualize LiDAR point cloud data using Open3D.

## File Structure
```
chapter1/
├── data/
│   ├── 00000.pcd  # First LiDAR scan data
│   └── 00010.pcd  # Second LiDAR scan data
├── visualize_lidar.py  # Main visualization script
├── requirements.txt    # Required Python packages
└── README.md          # This file
```

## Installation and Usage

### 1. Install Required Packages
```bash
pip3 install -r requirements.txt
```

### 2. Run the Script
```bash
cd chapter1
python3 visualize_lidar.py
```

## Features
- **00000.pcd**: Visualized in green color
- **00010.pcd**: Visualized in yellow color
- Display both point clouds simultaneously
- Optional individual point cloud visualization

## Controls
- **Mouse Drag**: Rotate point cloud
- **Mouse Wheel**: Zoom in/out
- **ESC key or Close Window**: Exit program

## Learning Objectives
1. Learn how to use the Open3D library
2. Understand PCD file loading and processing
3. Apply colors to point clouds and visualize them
4. Combine multiple point clouds

## Next Chapter Preview
Chapter 2 will cover how to calculate transformations between these two point clouds.
