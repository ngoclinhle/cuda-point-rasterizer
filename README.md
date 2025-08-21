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
- Transform pipeline: Model → World → Camera → Image coordinates
- Frustum culling for efficient rendering
- Depth testing with atomic operations
- Color assignment and background handling

### User Interface
- **Settings Window**: Camera resolution, field of view, navigation mode, background color
- **Debug Window**: Point cloud statistics, bounding box, transformation matrices
- **Performance Window**: Frame rate, memory usage, GPU utilization

## Technical Requirements

### Dependencies
- **C++17** or higher
- **CUDA Toolkit** (for GPU acceleration)
- **CMake 3.18+** (build system)
- **PDAL** (point cloud I/O)
- **ImGui** (user interface)
- **OpenGL** (rendering backend)
- **GLFW** (window management)

### Supported Platforms
- Linux (primary)
- Windows (with CUDA support)

## Building

### Prerequisites
Ensure you have CUDA Toolkit, PDAL, and other dependencies installed.

### Build Steps
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

### Quick Build Script
Alternatively, use the provided build script:
```bash
./build.sh
```

## Usage

```bash
# Run the application
./build/pcd_rasterizer [point_cloud_file]
```

### Camera Controls

#### 🌍 **ORBIT MODE** (Camera rotates around a target point)

**Mouse Controls:**
- **Left Click + Drag**: Orbit around target (azimuth/elevation)
- **Right Click + Drag**: Pan target point (move orbit center)
- **Middle Click + Drag**: Zoom in/out (distance from target)
- **Scroll Wheel**: Zoom in/out (alternative)
- **Left Double-Click**: Auto-fit all points in view / Reset to default orbit

**Behavior:**
- Target point remains fixed in space
- Camera maintains distance while rotating around target
- Smooth interpolation for all movements
- Auto-calculate bounding box center as default target

#### 🚶 **MODEL MODE** (First-person/free-roam camera)

**Mouse Controls:**
- **Left Click + Drag**: Look around (pitch/yaw rotation)
- **Right Click + Drag**: Strafe left/right, forward/backward
- **Middle Click + Drag**: Move up/down + forward/backward
- **Scroll Wheel**: Adjust movement speed
- **Left Double-Click**: Quick 180° turn

**Behavior:**
- Camera moves freely through 3D space
- No fixed target point
- Speed adjustable via scroll wheel
- Momentum/inertia for smooth movement

#### 🔄 **MODE SWITCHING**
- **Tab**: Toggle between Orbit ↔ Model modes
- **M**: Quick switch to Model mode
- **O**: Quick switch to Orbit mode

## Architecture

### Rasterization Pipeline
1. **Transform**: Model → World → Camera → Image coordinates
2. **Frustum Culling**: Remove points outside view frustum
3. **Depth Test**: Atomic depth buffer operations
4. **Color Assignment**: Final pixel color determination

### Coordinate System
- **RDF Convention**: +X right, -Y up, +Z forward
- **4x4 Transformation Matrices**: [R|t] format with homogeneous coordinates
- **3x3 Projection Matrix**: Perspective camera model

## Development Status

This project is currently under active development using Task Master AI for project management.

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable] 