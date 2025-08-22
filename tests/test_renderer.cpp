#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <fstream>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include "renderer.h"
#include "camera.h"

#include "stb_image_write.h"

// OpenGL error callback
void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// Function to download OpenGL texture and save to file
bool save_texture_to_file(GLuint textureId, int width, int height, const std::string& filename) {
    if (textureId == 0) {
        std::cerr << "Invalid texture ID" << std::endl;
        return false;
    }
    
    std::cout << "Downloading OpenGL texture (ID: " << textureId << ") to: " << filename << std::endl;
    
    // Bind the texture
    glBindTexture(GL_TEXTURE_2D, textureId);
    
    // Check for OpenGL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error binding texture: " << error << std::endl;
        return false;
    }
    
    // Allocate buffer for texture data (RGBA format)
    std::vector<unsigned char> textureData(width * height * 4);
    
    // Download texture data from GPU (RGBA format)
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData.data());
    
    // Check for OpenGL errors
    error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error reading texture: " << error << std::endl;
        return false;
    }
    
    // Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);
    
    // Convert RGBA to RGB for saving (drop alpha channel)
    std::vector<unsigned char> rgb_data(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb_data[i * 3 + 0] = textureData[i * 4 + 0];  // R
        rgb_data[i * 3 + 1] = textureData[i * 4 + 1];  // G
        rgb_data[i * 3 + 2] = textureData[i * 4 + 2];  // B
        // Drop alpha (textureData[i * 4 + 3])
    }
    
    // Determine format from file extension
    std::string extension = filename.substr(filename.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    int result = 0;
    if (extension == "png") {
        result = stbi_write_png(filename.c_str(), width, height, 3, rgb_data.data(), width * 3);
    } else if (extension == "jpg" || extension == "jpeg") {
        result = stbi_write_jpg(filename.c_str(), width, height, 3, rgb_data.data(), 90);
    } else if (extension == "bmp") {
        result = stbi_write_bmp(filename.c_str(), width, height, 3, rgb_data.data());
    } else if (extension == "tga") {
        result = stbi_write_tga(filename.c_str(), width, height, 3, rgb_data.data());
    } else {
        std::cerr << "Unsupported texture format: " << extension << std::endl;
        return false;
    }
    
    if (result) {
        std::cout << "Successfully saved texture to: " << filename << " (" << width << "x" << height << ")" << std::endl;
        
        // Also dump raw OpenGL texture data as binary file for debugging (RGB converted)
        std::string binary_filename = filename.substr(0, filename.find_last_of('.')) + "_opengl.bin";
        std::ofstream binary_file(binary_filename, std::ios::binary);
        if (binary_file.is_open()) {
            binary_file.write(reinterpret_cast<const char*>(rgb_data.data()), rgb_data.size());
            binary_file.close();
            std::cout << "DEBUG: Saved OpenGL texture binary: " << binary_filename << " (" << rgb_data.size() << " bytes, RGBA->RGB converted)" << std::endl;
        }
        
        return true;
    } else {
        std::cerr << "Failed to save texture to: " << filename << std::endl;
        return false;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_pointcloud> <output_image>" << std::endl;
    std::cout << "  input_pointcloud: Path to point cloud file (.ply, .pcd, .las, .laz)" << std::endl;
    std::cout << "  output_image:     Path to output image file (.png, .jpg, .bmp, .tga)" << std::endl;
    std::cout << std::endl;
    std::cout << "Output: Creates two files:" << std::endl;
    std::cout << "  1. <output_image>         - Direct CUDA framebuffer save" << std::endl;
    std::cout << "  2. <output_image>_texture - OpenGL texture download" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " data/sample.ply output.png" << std::endl;
    std::cout << "  Creates: output.png and output_texture.png" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Error: Expected 2 arguments, got " << (argc - 1) << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    std::cout << "Renderer Unit Test" << std::endl;
    std::cout << "Input:  " << input_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    std::cout << std::endl;
    
    // Initialize OpenGL context (required for Renderer)
    std::cout << "Initializing OpenGL context..." << std::endl;
    
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    
    // Configure GLFW for headless rendering
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hidden window for headless rendering
    
    // Create a hidden window for OpenGL context
    GLFWwindow* window = glfwCreateWindow(800, 600, "Renderer Test", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << std::endl;
    
    try {
        // Create renderer with 800x600 resolution
        const int width = 800;
        const int height = 600;
        std::cout << "Creating renderer (" << width << "x" << height << ")..." << std::endl;
        auto renderer = std::make_unique<Renderer>(width, height);
        
        // Load point cloud
        std::cout << "Loading point cloud: " << input_file << "..." << std::endl;
        if (!renderer->set_point_cloud(input_file)) {
            std::cerr << "Failed to load point cloud from: " << input_file << std::endl;
            return 1;
        }

        auto pcd = renderer->getPointCloud();
        while (!pcd->is_loaded()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Setup camera with reasonable defaults
        std::cout << "Setting up camera..." << std::endl;
        auto camera = std::make_unique<Camera>();
        
        // Set camera intrinsics
        camera->set_intrinsics(45.0f, width, height);  // 45 degree vertical FOV
        
        // Set camera position and orientation (looking down negative Z axis)
        camera->set_pos(Eigen::Vector3f(0.0f, 0.0f, -0.5f));  // 5 units back from origin
        camera->set_rot(Eigen::Matrix3f::Identity());  // No rotation (looking forward)
        
        // Set camera for renderer
        renderer->set_camera(camera.get());
        
        // Render one frame
        std::cout << "Rendering frame..." << std::endl;
        renderer->render();
        
        // Save framebuffer directly from CUDA
        std::cout << "Saving CUDA framebuffer to: " << output_file << "..." << std::endl;
        if (!renderer->save_to_image(output_file)) {
            std::cerr << "Failed to save CUDA framebuffer to: " << output_file << std::endl;
            return 1;
        }
        
        // Also save OpenGL texture for comparison
        std::string texture_filename = output_file.substr(0, output_file.find_last_of('.')) + "_texture" + 
                                     output_file.substr(output_file.find_last_of('.'));
        std::cout << "Saving OpenGL texture to: " << texture_filename << "..." << std::endl;
        
        GLuint textureId = renderer->getTexture();
        if (!save_texture_to_file(textureId, width, height, texture_filename)) {
            std::cerr << "Failed to save OpenGL texture to: " << texture_filename << std::endl;
            return 1;
        }
        
        std::cout << "Success! Rendered point cloud to both CUDA framebuffer and OpenGL texture." << std::endl;
        renderer.reset();
        // Cleanup OpenGL context
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
} 