#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <Eigen/Dense>

/**
 * Main rasterization function for point cloud rendering
 * 
 * @param positions Array of 3D point positions in device memory
 * @param colors Array of point colors in device memory
 * @param npoints Number of points in the arrays
 * @param F Combined transformation matrix (model-view matrix)
 * @param K Camera intrinsic matrix for projection
 * @param w Output image width in pixels
 * @param h Output image height in pixels
 * @param background_color Background color for out-of-bounds pixels
 * @param visible_filter Whether to filter out points that are not visible
 * @param cone_threshold visible threshold
 * @param visible_mask Visible mask array in device memory
 * @param packed_depth Output depth buffer in device memory (w*h*sizeof(float), can be nullptr)
 * @param frame Output color frame buffer in device memory (w*h*sizeof(uchar3))
 * 
 * This function orchestrates the complete rasterization pipeline:
 * 1. Point transformation and projection
 * 2. Depth testing with atomic operations
 * 3. Visibility filtering
 * 4. Final color and depth assignment
 */
void rasterization(float3* positions, uchar3* colors, size_t npoints, 
                  Eigen::Matrix4f F, Eigen::Matrix3f K, int w, int h, 
                  uchar4 background_color,
                  bool visible_filter,
                  float cone_threshold,
                  bool* visible_mask,
                  uint64_t* packed_depth,
                  uchar4* frame);


/**
 * Projection kernel - transforms points and performs depth testing
 * @param positions Input point positions
 * @param npoints Number of points
 * @param F Combined transformation matrix
 * @param K Camera intrinsic matrix
 * @param w Image width
 * @param h Image height
 * @param packed_depth Output depth buffer in device memory (w*h*sizeof(float), can be nullptr)
 */
__global__ void projection(float3* positions, size_t npoints, Eigen::Matrix4f F, Eigen::Matrix3f K, 
                          int w, int h, uint64_t* packed_depth);


/**
 * @brief Point rejection kernel
 * 
 * @param positions Point positions
 * @param camera_pos Camera position
 * @param packed_depth Packed depth buffer result from projection kernel
 * @param w Image width
 * @param h Image height
 * @param threshold Visible threshold
 * @param visible Visible flag array
 * @return __global__ 
 */
__global__ void point_rejection(float3* positions,
                                float3 camera_pos,
                                uint64_t* packed_depth,
                                int w,
                                int h,
                                float threshold,
                                bool* visible);

/**
 * 
 * Resolve kernel - assigns final colors and depth values
 * @param colors Point colors
 * @param packed_depth Packed depth buffer from projection kernel
 * @param w Image width
 * @param h Image height
 * @param frame Output color frame buffer
 * @param background_color Background color for out-of-bounds pixels
 * @param visible_mask Visible mask array in device memory
 */
__global__ void resolve(uchar3* colors, uint64_t* packed_depth, int w, int h, 
                       uchar4* frame, uchar4 background_color, bool visible_filter, float visible_threshold, bool* visible_mask); 