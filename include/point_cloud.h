#pragma once

#include <vector>
#include <cstdint>
#include <cfloat>
#include <string>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

typedef std::pair<float3, float3> BoundingBox; // min and max points

class PointCloud {
public:
    PointCloud(std::vector<float3>&& positions, std::vector<uchar3>&& colors, size_t num_points);
    ~PointCloud();
    
    PointCloud(const PointCloud& other) = delete;
    PointCloud& operator=(const PointCloud& other) = delete;
    
    // Single file loading method that auto-detects format
    static std::unique_ptr<PointCloud> from_file(const std::string& filename);

    size_t get_num_points() const { return num_points; }
    float3* get_positions() const { return d_positions; }
    uchar3* get_colors() const { return d_colors; }
    BoundingBox get_bounding_box() const { return bounding_box; }

private:
    size_t num_points;
    // GPU data
    float3* d_positions;            // positions on device
    uchar3* d_colors;               // colors on device
    
    // Host data
    std::vector<float3> h_positions; // positions on host
    std::vector<uchar3> h_colors; // colors on host
    
    BoundingBox bounding_box;

private:
    BoundingBox _calculate_bounding_box();
}; 