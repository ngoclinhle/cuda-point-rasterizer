#include "point_cloud_2.h"
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

// PDAL includes
#include <pdal/pdal.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/Stage.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Reader.hpp>
#include <pdal/Filter.hpp>
#include <pdal/Streamable.hpp>
#include <iostream>

PointBatch::PointBatch() {
    checkCudaErrors(cudaMallocManaged(&positions, sizeof(float3)*MAX_BATCH_SIZE));
    checkCudaErrors(cudaMallocManaged(&colors, sizeof(uchar3)*MAX_BATCH_SIZE));
}

PointBatch::~PointBatch() {
    if (positions) {
        checkCudaErrors(cudaFree(positions));
    }
    if (colors) {
        checkCudaErrors(cudaFree(colors));
    }
}

PointBatch::PointBatch(PointBatch&& other) noexcept : positions(other.positions), colors(other.colors) {
    other.positions = nullptr;
    other.colors = nullptr;
}

PointBatch& PointBatch::operator=(PointBatch&& other) noexcept {
    if (this != &other) {
        if (positions) {
            checkCudaErrors(cudaFree(positions));
        }
        if (colors) {
            checkCudaErrors(cudaFree(colors));
        }
        positions = other.positions;
        colors = other.colors;
        other.positions = nullptr;
        other.colors = nullptr;
    }
    return *this;
}

PointCloud2::PointCloud2() : num_points_(0) {
}

PointCloud2::~PointCloud2() {
}

void PointCloud2::add_points(const std::vector<float3>& positions, const std::vector<uchar3>& colors) {
    size_t new_points = positions.size();
    if (new_points == 0) {
        return;
    }

    // copy the new points to the batches_
    size_t batch_index, batch_offset;
    for (size_t i = 0; i < new_points; i++) {
        batch_index = (num_points_ + i) / MAX_BATCH_SIZE;
        batch_offset = (num_points_ + i) % MAX_BATCH_SIZE;
        if (batch_offset == 0) {
            batches_.push_back(PointBatch());
        }
        batches_[batch_index].get_positions()[batch_offset] = positions[i];
        batches_[batch_index].get_colors()[batch_offset] = colors[i];
    }

    num_points_ += new_points;
}

#define UB(x) (x >> 8)
#define LB(x) (x & 0xFF)
#define IS_U16(x) (UB(x)!=0)
#define IS_U8(x) (UB(x)==0 && LB(x)!=0)

int get_color_shift(const std::vector<pdal::PointRef>& points) {
    int u16_cnt = 0;
    int u8_cnt = 0;
    for (auto& point : points) {
        uint16_t r = point.getFieldAs<uint16_t>(pdal::Dimension::Id::Red);
        uint16_t g = point.getFieldAs<uint16_t>(pdal::Dimension::Id::Green);
        uint16_t b = point.getFieldAs<uint16_t>(pdal::Dimension::Id::Blue);
        if (IS_U16(r) && IS_U16(g) && IS_U16(b)) {
            u16_cnt++;
        } else if (IS_U8(r) && IS_U8(g) && IS_U8(b)) {
            u8_cnt++;
        }
    }
    if (u16_cnt > u8_cnt) {
        return 8;
    } else {
        return 0;
    }
}

void PointCloud2::add_points(const std::vector<pdal::PointRef>& points) {
    size_t batch_index, batch_offset;
    if (color_shift < 0) color_shift = get_color_shift(points);
    
    for (size_t i = 0; i < points.size(); i++) {
        batch_index = (num_points_ + i) / MAX_BATCH_SIZE;
        batch_offset = (num_points_ + i) % MAX_BATCH_SIZE;
        if (batch_offset == 0) {
            batches_.emplace_back();
        }
        auto& point = points[i];
        float x = point.getFieldAs<float>(pdal::Dimension::Id::X);
        float y = point.getFieldAs<float>(pdal::Dimension::Id::Y);
        float z = point.getFieldAs<float>(pdal::Dimension::Id::Z);
        uint16_t r = point.getFieldAs<uint16_t>(pdal::Dimension::Id::Red);
        uint16_t g = point.getFieldAs<uint16_t>(pdal::Dimension::Id::Green);
        uint16_t b = point.getFieldAs<uint16_t>(pdal::Dimension::Id::Blue);
        if (color_shift != 0) {
            r >>= color_shift;
            g >>= color_shift;
            b >>= color_shift;
        }
        float3 pos = make_float3(x, y, z);
        uchar3 color = make_uchar3(r, g, b);
        
        batches_[batch_index].get_positions()[batch_offset] = pos;
        batches_[batch_index].get_colors()[batch_offset] = color;
    }
    num_points_ += points.size();
}

class PdalFilter : public pdal::Filter, public pdal::Streamable
{
public:
    PdalFilter()
    {}

    std::string getName() const { return "filters.test"; }
    void set_cb(std::function<void(pdal::PointRef&)> cb) {
        cb_ = cb;
    }

private:
    std::function<void(pdal::PointRef&)> cb_;
    virtual bool processOne(pdal::PointRef& point)
    {
        if (cb_) {
            cb_(point);
        }
        return true;
    }
};

bool PointCloud2::pdal_read(const std::string& filename) {
    try {
        pdal::StageFactory factory;
        auto readerType = factory.inferReaderDriver(filename);
        if (readerType.empty()) {
            throw std::runtime_error("Failed to infer PDAL reader driver for format: " + filename);
        }
        auto reader = factory.createStage(readerType);
        pdal::Options readerOptions;
        readerOptions.add("filename", filename);
        reader->setOptions(readerOptions);

        pdal::QuickInfo info = reader->preview();
        if (info.valid()) {
            auto bb = info.m_bounds;
            auto min = make_float3(bb.minx, bb.miny, bb.minz);
            auto max = make_float3(bb.maxx, bb.maxy, bb.maxz);
            bounding_box_ = BoundingBox(min, max);
        }

        int chunk_read = 0;
        std::vector<pdal::PointRef> point_cache;
        auto cb = [this, &chunk_read, &point_cache](pdal::PointRef& point) {
            chunk_read++;
            point_cache.push_back(point);
            if (chunk_read >= MAX_BATCH_SIZE) {
                std::cout << "adding chunk " << chunk_read << std::endl;
                add_points(point_cache);
                chunk_read = 0;
                point_cache.clear();
            }
        };


        PdalFilter filter;
        filter.set_cb(cb);
        filter.setInput(*reader);

        pdal::FixedPointTable table(MAX_BATCH_SIZE*2);
        filter.prepare(table);
        filter.execute(table);

        if (chunk_read > 0) {
            add_points(point_cache);
        }
        
        std::cout << "PDAL loaded " << num_points_ << " points from " << filename << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading file: " << e.what() << std::endl;
        return false;
    }
}

void PointCloud2::load_from_file_async(const std::string& filename) {
    load_future_ = std::async(std::launch::async, [this, filename]() {
        return pdal_read(filename);
    });
}

bool PointCloud2::is_loaded() const {
    return load_future_.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
}
