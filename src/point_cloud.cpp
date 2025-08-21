#include "point_cloud.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <iomanip>
#include <pdal/pdal.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/Stage.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Reader.hpp>

#define TIMER_START(name) auto name##_start = std::chrono::high_resolution_clock::now();
#define TIMER_END(name) do { \
    auto name##_end = std::chrono::high_resolution_clock::now(); \
    auto name##_duration = std::chrono::duration_cast<std::chrono::microseconds>(name##_end - name##_start); \
    std::cout << #name << " time: " << std::fixed << std::setprecision(2) << name##_duration.count() / 1000.0 << " ms" << std::endl; \
} while (0)

// Constructor with positions and colors
PointCloud::PointCloud(std::vector<float3>&& positions, std::vector<uchar3>&& colors, size_t num_points) 
    : d_positions(nullptr), d_colors(nullptr), num_points(num_points), h_positions(std::move(positions)), h_colors(std::move(colors)) {
    
    if (num_points == 0 || h_positions.empty()) {
        return;
    }

    TIMER_START(gpu);

    bounding_box = _calculate_bounding_box();
    
    // Allocate GPU memory for positions
    checkCudaErrors(cudaMalloc(&d_positions, num_points * sizeof(float3)));
    
    // Allocate GPU memory for colors  
    checkCudaErrors(cudaMalloc(&d_colors, num_points * sizeof(uchar3)));
    
    // Copy data to GPU
    checkCudaErrors(cudaMemcpy(d_positions, h_positions.data(), num_points * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colors, h_colors.data(), num_points * sizeof(uchar3), cudaMemcpyHostToDevice));
    
    TIMER_END(gpu);
    
    std::cout << "PointCloud created with " << num_points << " points uploaded to GPU" << std::endl;
    std::cout << "Bounding box: \n  x: " << bounding_box.first.x << ":" << bounding_box.second.x 
            << "\n  y: " << bounding_box.first.y << ":" << bounding_box.second.y 
            << "\n  z: " << bounding_box.first.z << ":" << bounding_box.second.z << std::endl;
}

BoundingBox PointCloud::_calculate_bounding_box() {
    float3 min_point = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max_point = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);

    for (size_t i = 0; i < num_points; i++) {
        float3 point = h_positions[i];
        min_point.x = std::min(min_point.x, point.x);
        min_point.y = std::min(min_point.y, point.y);
        min_point.z = std::min(min_point.z, point.z);
        max_point.x = std::max(max_point.x, point.x);
        max_point.y = std::max(max_point.y, point.y);
        max_point.z = std::max(max_point.z, point.z);
    }
    
    return BoundingBox(min_point, max_point);
}

// Destructor - cleanup GPU memory
PointCloud::~PointCloud() {
    if (d_positions) {
        checkCudaErrors(cudaFree(d_positions));
    }
    if (d_colors) {
        checkCudaErrors(cudaFree(d_colors));
    }
}

// Extract point data using PDAL
void pdal_read(const std::string& filename, 
                std::vector<float3>& positions, 
                std::vector<uchar3>& colors,
                size_t& pointCount) {
    try {
        TIMER_START(pdal_setup);
        
        // Create PDAL pipeline
        pdal::StageFactory factory;
        std::string readerType = pdal::StageFactory::inferReaderDriver(filename);
        if (readerType.empty()) {
            throw std::runtime_error("Failed to infer PDAL reader driver for format: " + filename);
        }
        else {
            std::cout << "PDAL reader driver: " << readerType << std::endl;
        }
        // Create reader stage
        auto reader = factory.createStage(readerType);
        
        // Set options
        pdal::Options readerOptions;
        readerOptions.add("filename", filename);
        reader->setOptions(readerOptions);

        
        // Create point table and prepare
        // pdal::PointTable table;
        pdal::FixedPointTable table(100);
        reader->prepare(table);
        
        TIMER_END(pdal_setup);

        // Execute pipeline (file reading + parsing)
        TIMER_START(pdal_execute);
        // pdal::PointViewSet viewSet = reader->execute(table);
        reader->execute(table);
        TIMER_END(pdal_execute);

        // if (viewSet.empty()) {
        //     throw std::runtime_error("No point data found in file");
        // }
        
        // // Get the first (and typically only) point view
        // pdal::PointViewPtr view = *viewSet.begin();
        // pointCount = view->size();
        
        // if (pointCount == 0) {
        //     throw std::runtime_error("Point cloud contains no points");
        // }
        
        std::cout << "PDAL loaded " << pointCount << " points from " << filename << std::endl;
        
        // Start timing the vector copy phase
        TIMER_START(vector_copy);
        
        // Reserve space for data
        positions.reserve(pointCount);
        colors.reserve(pointCount);

        return;
#if 0        
        // Check what dimensions are available
        bool hasColors = table.layout()->hasDim(pdal::Dimension::Id::Red) &&
                        table.layout()->hasDim(pdal::Dimension::Id::Green) &&
                        table.layout()->hasDim(pdal::Dimension::Id::Blue);
        
        std::cout << "File has color data: " << (hasColors ? "Yes" : "No") << std::endl;

        // Extract point data
        for (pdal::PointId i = 0; i < pointCount; ++i) {
            // Extract positions (X, Y, Z)
            double x = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
            double y = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
            double z = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
            float3 position = make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
            
            positions.push_back(position);
            
            // Extract colors if available
            if (hasColors) {
                uint16_t r = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Red, i);
                uint16_t g = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Green, i);
                uint16_t b = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Blue, i);
                uint8_t r8 = static_cast<uint8_t>(r >> 8);
                uint8_t g8 = static_cast<uint8_t>(g >> 8);
                uint8_t b8 = static_cast<uint8_t>(b >> 8);
                
                uchar3 color = make_uchar3(r8, g8, b8);
                colors.push_back(color);
            } else {
                // Default gray color if no color data
                uchar3 color = make_uchar3(128, 128, 128);
                colors.push_back(color);
            }
        }

        TIMER_END(vector_copy);
        
        std::cout << "Successfully extracted " << pointCount << " points with " 
                  << (hasColors ? "colors" : "default colors") << std::endl;
                  
#endif
    } catch (const pdal::pdal_error& e) {
        throw std::runtime_error("PDAL error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error reading file: " + std::string(e.what()));
    }
}

// Static method to load any supported file format
std::unique_ptr<PointCloud> PointCloud::from_file(const std::string& filename) {
    std::cout << "Loading point cloud file: " << filename << std::endl;
    
    TIMER_START(total);
    
    try {
        std::vector<float3> positions;
        std::vector<uchar3> colors;
        size_t pointCount;

        TIMER_START(pdal_read);
        pdal_read(filename, positions, colors, pointCount);
        TIMER_END(pdal_read);

        TIMER_START(construction);
        auto point_cloud = std::make_unique<PointCloud>(std::move(positions), std::move(colors), pointCount);
        TIMER_END(construction);

        TIMER_END(total);
        
        return point_cloud;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load point cloud file '" + filename + "': " + e.what());
    }
}

