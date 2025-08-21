#pragma once

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <memory>
#include <string>
#include "camera.h"
#include "point_cloud.h"

/**
 * @brief Renderer manages point cloud rendering with CUDA and OpenGL integration
 * 
 * The Renderer class handles:
 * - Point cloud data management
 * - Camera association for rendering
 * - CUDA framebuffer allocation and management
 * - OpenGL texture creation and updates
 * - Point cloud rasterization pipeline
 */
class Renderer {
public:

    /**
     * @brief Constructor with initial dimensions
     * 
     * @param w Initial width in pixels
     * @param h Initial height in pixels
     */
    Renderer(int w, int h);

    /**
     * @brief Destructor - cleans up CUDA and OpenGL resources
     */
    ~Renderer();

    /**
     * @brief Load and set the point cloud to render
     * 
     * @param filename Path to point cloud file
     * @return true if loaded successfully, false otherwise
     */
    bool set_point_cloud(const std::string& filename);

    /**
     * @brief Set the camera for rendering
     * 
     * @param cam Pointer to Camera object (does not take ownership)
     */
    void set_camera(Camera* cam);

    void set_visible_filter(bool visible_filter, float threshold);
    void get_visible_filter(bool& visible_filter, float& threshold);

    /**
     * @brief Render the point cloud to the framebuffer
     * 
     * Uses the current camera and point cloud to render an image
     * to the internal framebuffer, then updates the OpenGL texture.
     */
    void render();

    /**
     * @brief Resize the framebuffer and update camera intrinsics
     * 
     * @param w New width in pixels
     * @param h New height in pixels
     * @return OpenGL texture ID for the resized framebuffer
     */
    GLuint resize(int w, int h);

    /**
     * @brief Get OpenGL texture ID for display
     * @return OpenGL texture ID (0 if not initialized)
     */
    GLuint getTexture();
    
    /**
     * @brief Get current framebuffer width
     * @return Current width in pixels
     */
    int getWidth() const { return width; }
    
    /**
     * @brief Get current framebuffer height  
     * @return Current height in pixels
     */
    int getHeight() const { return height; }
    
    /**
     * @brief Save current framebuffer to image file
     * 
     * @param filename Output image file path
     * @return true if successful, false otherwise
     */
    bool save_to_image(const std::string& filename);
    
    /**
     * @brief Get the current point cloud (for accessing bounding box, etc.)
     * 
     * @return Pointer to current point cloud, or nullptr if none loaded
     */
    PointCloud* getPointCloud() const { return pcd.get(); }

private:
    std::unique_ptr<PointCloud> pcd;
    Camera* camera;
    
    // Framebuffer dimensions
    int width;
    int height;
    
    // CUDA-OpenGL interop
    cudaGraphicsResource_t cuda_texture_resource;
    GLuint textureId;
    
    bool visible_filter;
    float visible_threshold;
    
    // Rendering resources
    size_t buffer_size;
    size_t depth_buffer_size;
    size_t mask_size;
    uchar4* d_frame_buffer;
    uint64_t* d_packed_depth;
    bool* d_visible_mask;

    /**
     * @brief Initialize OpenGL texture
     * 
     * @param w Width in pixels
     * @param h Height in pixels
     */
    GLuint create_gl_texture(int w, int h);
    
    
    
    /**
     * @brief Helper method to allocate all buffers and texture
     * 
     * @param w Width in pixels
     * @param h Height in pixels
     */
    void initialize_resources(int w, int h);
    
    /**
     * @brief Clean up all allocated resources
     */
    void cleanup_resources();
    
    /**
     * @brief Update OpenGL texture from CUDA framebuffer
     */
    void update_texture();
}; 