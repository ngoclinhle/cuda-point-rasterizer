#pragma once

#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <future>
#include <cuda_runtime.h>
#include "cuda_vector.h"
#include <pdal/PointRef.hpp>

// Forward declarations for PDAL
namespace pdal {
    class PointView;
    using PointId = uint64_t;
}
typedef std::pair<float3, float3> BoundingBox; // min and max points
#define MAX_BATCH_SIZE (256*1024)

struct PointBatch {
    PointBatch();
    ~PointBatch();
    // no copy
    PointBatch(const PointBatch&) = delete;
    PointBatch& operator=(const PointBatch&) = delete;
    // move constructor
    PointBatch(PointBatch&& other) noexcept;
    // move assignment operator
    PointBatch& operator=(PointBatch&& other) noexcept;

    __host__ __device__ float3* get_positions() const { return positions; }
    __host__ __device__ uchar3* get_colors() const { return colors; }
private:
    float3* positions;
    uchar3* colors;
};

class PointCloud2 {
public:
    PointCloud2();
    ~PointCloud2();

    void load_from_file_async(const std::string& filename);
    void add_points(const std::vector<float3>& positions, const std::vector<uchar3>& colors);
    
    size_t get_num_points() const { return num_points_; }
    BoundingBox get_bounding_box() const { return bounding_box_; }
    PointBatch* get_batches() const { return batches_.data(); }
    bool is_loaded() const;
    std::mutex load_mutex_;

private:
    void add_points(const std::vector<pdal::PointRef>& points);
    bool pdal_read(const std::string& filename);
    
    int color_shift=-1;
    size_t num_points_;
    CudaVector<PointBatch> batches_;
    BoundingBox bounding_box_;
    std::future<bool> load_future_;
};