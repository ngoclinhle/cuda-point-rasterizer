#include "renderer.h"
#include "rasterization.h"
#include "point_cloud.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <GL/gl.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Renderer::Renderer(int w, int h) 
    : pcd(nullptr), camera(nullptr), d_frame_buffer(nullptr), d_visible_mask(nullptr), textureId(0),
      width(0), height(0), buffer_size(0), cuda_texture_resource(nullptr),
      d_packed_depth(nullptr), depth_buffer_size(0){
    
    visible_filter = true;
    visible_threshold = 4.0f;
    
    if (w > 0 && h > 0) {
        initialize_resources(w, h);
    }
}


Renderer::~Renderer() {
    cleanup_resources();
}


bool Renderer::set_point_cloud(const std::string& filename) {
    try {
        std::cout << "Renderer loading point cloud: " << filename << std::endl;
        pcd = PointCloud::from_file(filename);
        
        if (pcd) {
            std::cout << "Renderer loaded point cloud with " 
                      << pcd->get_num_points() << " points" << std::endl;
            return true;
        }
        
        std::cerr << "Failed to load point cloud in renderer" << std::endl;
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << "Renderer error loading point cloud: " << e.what() << std::endl;
        pcd.reset();
        return false;
    }
}


void Renderer::set_camera(Camera* cam) {
    this->camera = cam;
    float v_fov;
    int w, h;
    camera->get_intrinsics(v_fov, w, h);
    camera->set_intrinsics(v_fov, width, height);
}

void Renderer::set_visible_filter(bool visible_filter, float threshold) {
    this->visible_filter = visible_filter;
    this->visible_threshold = threshold;
}

void Renderer::get_visible_filter(bool& visible_filter, float& threshold) {
    visible_filter = this->visible_filter;
    threshold = this->visible_threshold;
}

void Renderer::render() {
    if (!pcd || !camera || !d_frame_buffer || !d_packed_depth) {
        return;
    }
    
    Eigen::Matrix3f K = camera->get_K();
    Eigen::Matrix4f F = camera->get_F();
    
    checkCudaErrors(cudaMemset(d_packed_depth, 0xFF, depth_buffer_size));
    checkCudaErrors(cudaMemset(d_frame_buffer, 0, buffer_size));
    checkCudaErrors(cudaMemset(d_visible_mask, 0, mask_size));
    
    rasterization(
        pcd->get_positions(),          // Point positions
        pcd->get_colors(),          // Point colors
        pcd->get_num_points(),          // Number of points
        F,                           // Extrinsic matrix
        K,                           // Intrinsic matrix
        width, height,               // Image dimensions
        make_uchar4(0, 0, 0, 255),       // Background color (black)
        visible_filter,
        visible_threshold,
        d_visible_mask,
        d_packed_depth,              // Depth buffer
        d_frame_buffer                 // Output framebuffer
    );
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    update_texture();
}

// Resize the framebuffer and update camera intrinsics
GLuint Renderer::resize(int w, int h) {
    if (w <= 0 || h <= 0) {
        return textureId; // Invalid dimensions
    }
    
    // Update camera intrinsics if camera is set
    if (camera) {
        float v_fov;
        int old_w, old_h;
        camera->get_intrinsics(v_fov, old_w, old_h);
        camera->set_intrinsics(v_fov, w, h);
    }
    
    // Reallocate buffers with new dimensions
    initialize_resources(w, h);
    
    return textureId;
}

// Get the OpenGL texture ID
GLuint Renderer::getTexture() {
    return textureId;
}

void Renderer::initialize_resources(int w, int h) {
    if (w <= 0 || h <= 0) {
        return;
    }

    
    // Clean up existing resources if they exist
    cleanup_resources();
    
    // Update dimensions
    width = w;
    height = h;
    
    size_t pixels = w * h;
    buffer_size = pixels * sizeof(uchar4);
    depth_buffer_size = pixels * sizeof(uint64_t);
    mask_size = pixels * sizeof(bool);
    
    checkCudaErrors(cudaMalloc((void**)&d_frame_buffer, buffer_size));
    checkCudaErrors(cudaMalloc((void**)&d_packed_depth, depth_buffer_size));
    checkCudaErrors(cudaMalloc((void**)&d_visible_mask, mask_size));
    textureId = create_gl_texture(w, h);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_texture_resource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

// Initialize OpenGL texture
GLuint Renderer::create_gl_texture(int w, int h) {
    GLuint texId;
    // Generate OpenGL texture
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Allocate texture storage (RGBA format for CUDA interop compatibility)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    
    // Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);
    
    // Check for OpenGL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error in create_gl_texture: " << error << std::endl;
    }

    return texId;
}


// Clean up all allocated resources
void Renderer::cleanup_resources() {
    // Unregister CUDA-OpenGL interop
    if (cuda_texture_resource) {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_texture_resource));
        cuda_texture_resource = nullptr;
    }
    
    // Clean up CUDA resources
    if (d_frame_buffer) {
        checkCudaErrors(cudaFree(d_frame_buffer));
        d_frame_buffer = nullptr;
        buffer_size = 0;
    }
    
    if (d_packed_depth) {
        checkCudaErrors(cudaFree(d_packed_depth));
        d_packed_depth = nullptr;
        depth_buffer_size = 0;
    }

    if (d_visible_mask) {
        checkCudaErrors(cudaFree(d_visible_mask));
        d_visible_mask = nullptr;
        mask_size = 0;
    }
    
    // Clean up OpenGL resources
    if (textureId != 0) {
        glDeleteTextures(1, &textureId);
        textureId = 0;
    }
    
    // Reset dimensions
    width = 0;
    height = 0;
}

// Update OpenGL texture from CUDA framebuffer
void Renderer::update_texture() {
    if (textureId == 0 || !d_frame_buffer) {
        return;
    }
    
    // Map CUDA resource
    cudaArray_t cuda_array;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_texture_resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_texture_resource, 0, 0));
    
    // Copy from framebuffer to texture
    checkCudaErrors(cudaMemcpy2DToArray(
        cuda_array,                    // Destination array
        0, 0,                         // Offset
        d_frame_buffer,                 // Source
        width * sizeof(uchar4),       // Source pitch
        width * sizeof(uchar4),       // Width in bytes
        height,                       // Height
        cudaMemcpyDeviceToDevice      // Copy type
    ));
    
    // Unmap CUDA resource
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_texture_resource));
    
    // Synchronize
    checkCudaErrors(cudaDeviceSynchronize());
}

bool Renderer::save_to_image(const std::string& filename) {
    if (!d_frame_buffer || width <= 0 || height <= 0) {
        std::cerr << "No framebuffer data to save or invalid dimensions" << std::endl;
        return false;
    }
    
    try {
        // CUDA framebuffer is RGBA (4 components), but we want to save as RGB
        // Need to convert RGBA to RGB by dropping the alpha channel
        size_t pixel_count = width * height;
        size_t rgba_size = pixel_count * sizeof(uchar4);
        size_t rgb_size = pixel_count * 3;
        
        std::vector<uchar4> rgba_data(pixel_count);
        std::vector<unsigned char> rgb_image(rgb_size);
        
        std::cout << "DEBUG: CUDA framebuffer is RGBA, converting to RGB for save" << std::endl;
        std::cout << "DEBUG: sizeof(uchar4) = " << sizeof(uchar4) << ", copying " << rgba_size << " bytes" << std::endl;
        
        // Copy RGBA data from device
        checkCudaErrors(cudaMemcpy(rgba_data.data(), d_frame_buffer, 
                                   rgba_size, cudaMemcpyDeviceToHost));
        
        // Convert RGBA to RGB (drop alpha channel)
        for (size_t i = 0; i < pixel_count; i++) {
            rgb_image[i * 3 + 0] = rgba_data[i].x;  // R
            rgb_image[i * 3 + 1] = rgba_data[i].y;  // G
            rgb_image[i * 3 + 2] = rgba_data[i].z;  // B
            // Drop alpha (rgba_data[i].w)
        }
        
        // Determine format from file extension
        std::string extension = filename.substr(filename.find_last_of('.') + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        int result = 0;
        if (extension == "png") {
            result = stbi_write_png(filename.c_str(), width, height, 3, rgb_image.data(), width * 3);
        } else if (extension == "jpg" || extension == "jpeg") {
            result = stbi_write_jpg(filename.c_str(), width, height, 3, rgb_image.data(), 90); // 90% quality
        } else if (extension == "bmp") {
            result = stbi_write_bmp(filename.c_str(), width, height, 3, rgb_image.data());
        } else if (extension == "tga") {
            result = stbi_write_tga(filename.c_str(), width, height, 3, rgb_image.data());
        } else {
            std::cerr << "Unsupported image format: " << extension << std::endl;
            std::cerr << "Supported formats: PNG, JPG, BMP, TGA" << std::endl;
            return false;
        }
        
        if (result) {
            std::cout << "Successfully saved image: " << filename << " (" << width << "x" << height << ")" << std::endl;
            
            // Also dump raw CUDA framebuffer as binary file for debugging
            std::string binary_filename = filename.substr(0, filename.find_last_of('.')) + "_cuda.bin";
            std::ofstream binary_file(binary_filename, std::ios::binary);
            if (binary_file.is_open()) {
                binary_file.write(reinterpret_cast<const char*>(rgb_image.data()), rgb_image.size());
                binary_file.close();
                std::cout << "DEBUG: Saved CUDA framebuffer binary: " << binary_filename << " (" << rgb_image.size() << " bytes)" << std::endl;
            }
            
            return true;
        } else {
            std::cerr << "Failed to save image: " << filename << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving image: " << e.what() << std::endl;
        return false;
    }
}