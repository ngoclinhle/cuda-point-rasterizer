# Point Cloud Rasterizer

A high-performance point cloud visualization application using CUDA acceleration and real-time rasterization.

## Overview

This project implements a point cloud rasterization pipeline that transforms 3D point clouds into 2D images using GPU acceleration. The application supports interactive navigation and real-time visualization of large point cloud datasets.

## Features

### Core Functionality
- **Multi-format Point Cloud Loading**: Support for PLY, PCD, and LAS file formats
- **GPU-Accelerated Rasterization**: CUDA-based point cloud rendering pipeline
- **Real-time Visualization**: Interactive 2D image display with configurable resolution
- **Camera Control**: Two navigation modes (Model and Orbit) with mouse interaction

### Technical Features
- Transform pipeline: World → Camera → Image coordinates
- Frustum culling for efficient rendering
- Depth testing with atomic operations
- Point rejection using cone aperture 
- Color assignment 

### References
- [Real-time Rendering of Massive Unstructured Raw Point Clouds using Screen-space Operators](http://www.crs4.it/vic/data/papers/vast2011-pbr.pdf)
- [Rendering Point Clouds with Compute Shaders and Vertex Order Optimization](https://www.cg.tuwien.ac.at/research/publications/2021/SCHUETZ-2021-PCC/)

## Technical Requirements
### Dependencies
- **C++17** or higher
- **CUDA Toolkit**
- **CMake 3.18+**
- **Eigen 3.4**
- **PDAL**
- **ImGui**
- **OpenGL** 
- **GLFW** 

### Supported Platforms
- Linux 
- Windows (not yet tested)

## Building

```bash
# Clone the repository
git clone <repository-url>
cd pcd_rasterizer

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)
```


## Usage

```bash
# Run the application
./build/pcd_rasterizer [point_cloud_file]
```

### Camera Controls

#### 🌍 **ORBIT MODE** (Camera rotates around a target point)

**Mouse Controls:**
- **Left Click + Drag**: Orbit around target (yaw and pitch control)
- **Right Click + Drag**: Pan target point (move orbit center)
- **Scroll Wheel**: Zoom in/out (distance from target)

#### 🚶 **MODEL MODE** (First-person/free-roam camera)

**Mouse Controls:**
- **Left Click + Drag**: Look around (pitch/yaw rotation)
- **Right Click + Drag**: Strafe left/right, up/down
- **Scroll Wheel**: move forward/backward

# TODO
- improve point cloud loading time