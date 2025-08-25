#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include "rasterization.h"
#include "helper_cuda.h"
#include "point_cloud_2.h"

#define BATCH_IDX(point_idx) (point_idx / MAX_BATCH_SIZE)
#define BATCH_OFFSET(point_idx) (point_idx % MAX_BATCH_SIZE)

__device__ float3 get_position(PointBatch* batches, int point_idx) {
    int batch_idx = BATCH_IDX(point_idx);
    int batch_offset = BATCH_OFFSET(point_idx);
    PointBatch* batch = &batches[batch_idx];
    return batch->get_positions()[batch_offset];
}

__device__ uchar3 get_color(PointBatch* batches, int point_idx) {
    int batch_idx = BATCH_IDX(point_idx);
    int batch_offset = BATCH_OFFSET(point_idx);
    PointBatch* batch = &batches[batch_idx];
    return batch->get_colors()[batch_offset];
}

__global__ void projection(PointBatch* batches, 
                           size_t npoints, 
                           Eigen::Matrix4f F, 
                           Eigen::Matrix3f K, 
                           int w, 
                           int h, 
                           uint64_t* packed_depth) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= npoints) return;

    // Transform point using combined matrix F
    float3 p = get_position(batches, point_idx);
    Eigen::Vector4f p_homo(p.x, p.y, p.z, 1.0f);
    Eigen::Vector4f p_transformed = F * p_homo;
    Eigen::Vector3f p_cam = p_transformed.head<3>();
    
    // Frustum culling - behind camera
    if (p_cam.z() <= 0.1f) return;
    
    // Project using intrinsic matrix K
    Eigen::Vector3f cam_for_projection(p_cam.x() / p_cam.z(), p_cam.y() / p_cam.z(), 1.0f);
    Eigen::Vector3f projected = K * cam_for_projection;
    
    // Convert to pixel coordinates
    int u = __float2int_rd(projected.x());
    int v = __float2int_rd(projected.y());
    
    // Frustum culling - outside image bounds
    if (u < 0 || u >= w || v < 0 || v >= h) return;
    
    // Depth testing with atomic operations
    int pixel_id = v * w + u;
    uint32_t depth_bits = __float_as_uint(p_cam.z());
    uint64_t new_packed = ((uint64_t)depth_bits << 32) | (uint32_t)point_idx;
    uint64_t old_packed = packed_depth[pixel_id];
    
    if (new_packed < old_packed) {
        atomicMin((unsigned long long*)&packed_depth[pixel_id], new_packed);
    }
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __constant__ int8_t sectors[7*7] = {7, 7, 7, 0, 0, 0, 1, 
                                               6, 7, 7, 0, 0, 1, 1, 
                                               6, 6, 7, 0, 1, 1, 1, 
                                               6, 6, 6,-1, 2, 2, 2, 
                                               5, 5, 5, 4, 3, 2, 2, 
                                               5, 5, 4, 4, 3, 3, 2, 
                                               5, 4, 4, 4, 3, 3, 3};
__device__ __forceinline__ int8_t sector_lookup(int i, int j) {
    if (i>=-3 && i<=3 && j>=-3 && j<=3) {
        return sectors[i+3 + (j+3)*7];
    }
    return -1;
}

__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __forceinline__ float3 normalize(float3 a) {
    float norm = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    return make_float3(a.x/norm, a.y/norm, a.z/norm);
}

__global__ void point_rejection(PointBatch* batches,
                                float3 camera_pos,
                                uint64_t* packed_depth,
                                int w,
                                int h,
                                float threshold,
                                bool* visible) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int pixel_id = y * w + x;
    uint64_t packed = packed_depth[pixel_id];
    uint32_t point_index = packed & 0xFFFFFFFF;
    if (point_index == 0xFFFFFFFF) return;

    float3 point = get_position(batches, point_index);
    float3 pc = camera_pos - point;
    float3 pc_norm = normalize(pc);
    float max_cos[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    for (int j=-3; j<=3; j++) {
        for (int i=-3; i<=3; i++) {
            if (x+i < 0 || x+i >= w || y+j < 0 || y+j >= h) continue;
            int8_t sector = sector_lookup(i, j);

            if (sector != -1) {
                int pix_id = (y+j)*w + (x+i);
                uint64_t packed = packed_depth[pix_id];
                uint32_t pt_id = packed & 0xFFFFFFFF;
                if (pt_id == 0xFFFFFFFF) continue;
                float3 other_point = get_position(batches, pt_id);
                float3 pp = other_point - point;
                float3 pp_norm = normalize(pp);
                float cos_cone = dot(pc_norm, pp_norm);
                max_cos[sector] = max(max_cos[sector], cos_cone);
            }
        }
    }
    float cos_sum = 0;
    for (int i=0; i<8; i++) {
        cos_sum += max_cos[i];
    }
    if (cos_sum < threshold) {
        visible[pixel_id] = true;
    }
}

__global__ void resolve(PointBatch* batches, 
                        uint64_t* packed_depth, 
                        int w, 
                        int h, 
                        uchar4* frame, 
                        uchar4 background_color,
                        bool visible_filter,
                        bool* visible_mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= w || y >= h) return;
    
    int pixel_id = y * w + x;
    uint64_t packed = packed_depth[pixel_id];
    
    // Extract point index from packed value
    uint32_t point_index = packed & 0xFFFFFFFF;
    bool point_visible = !visible_filter || visible_mask[pixel_id];

    if (point_visible && point_index != 0xFFFFFFFF) {
        uchar3 color = get_color(batches, point_index);
        frame[pixel_id] = make_uchar4(color.x, color.y, color.z, 255);
    } else {
        frame[pixel_id] = background_color;
    }
}

void rasterization(PointBatch* batches, 
                    size_t npoints, 
                    Eigen::Matrix4f F, 
                    Eigen::Matrix3f K, 
                    int w, 
                    int h, 
                    uchar4 background_color,
                    bool visible_filter,
                    float cone_threshold,
                    bool* visible_mask,
                    uint64_t* packed_depth, 
                    uchar4* frame) {    
    // Calculate grid and block dimensions
    int block_size = 256;
    int num_blocks = (npoints + block_size - 1) / block_size;
    
    dim3 projection_blocks(num_blocks);
    dim3 projection_threads(block_size);
    
    // For resolve kernel, use 2D grid
    dim3 resolve_block(16, 16);
    dim3 resolve_grid((w + 15) / 16, (h + 15) / 16);
    
    // Launch projection kernel
    projection<<<projection_blocks, projection_threads>>>(
        batches, npoints, F, K, w, h, packed_depth
    );
    
    getLastCudaError("Projection kernel launch failed");

    if (visible_filter) {
        Eigen::Matrix3f R = F.block<3, 3>(0, 0);
        Eigen::Vector3f t = F.block<3, 1>(0, 3);
        Eigen::Vector3f p = -R.transpose() * t;
        float3 camera_pos = make_float3(p.x(), p.y(), p.z());

        point_rejection<<<resolve_grid, resolve_block>>>(
            batches, camera_pos, packed_depth, w, h, cone_threshold, visible_mask
        );
        getLastCudaError("Point rejection kernel launch failed");
    }
    
    // Launch resolve kernel
    resolve<<<resolve_grid, resolve_block>>>(
        batches, packed_depth, w, h, frame, background_color, visible_filter, visible_mask
    );
    
    getLastCudaError("Resolve kernel launch failed");
    
    // Final synchronization
    cudaDeviceSynchronize();
}