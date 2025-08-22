#pragma once

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <string>
#include <memory>
#include "renderer.h"
#include "camera.h"
#include "camera_controller.h"

/**
 * @brief UI manages various ImGUI windows for the application
 * 
 * The UI class provides the graphical user interface using ImGUI.
 * It displays the renderer output, camera controls, point cloud information,
 * and performance metrics in separate windows.
 */
struct UI {
    std::shared_ptr<Renderer> renderer;
    std::shared_ptr<CameraController> cameraController;

    /**
     * @brief Constructor
     */
    UI();
    void render();
    
    // Check if viewport window is focused for camera controls
    bool isViewportHovered() const;
    void reset_camera();

private:
    // Window visibility flags
    bool showViewport;              ///< Show viewport window
    bool showCameraControls;        ///< Show camera controls window
    bool showPointCloudInfo;        ///< Show point cloud information window
    bool showPerformance;           ///< Show performance metrics window
    bool showSettings;              ///< Show application settings window
    bool showDebug;                 ///< Show debug information window

    // Check if viewport window is focused for camera controls
    bool viewport_hovered = false;
    
    /**
     * @brief Render the viewport window
     * 
     * @param renderer Renderer to display
     */
    void renderViewportWindow();
    
    /**
     * @brief Render camera controls window
     * 
     * @param camera Camera to control
     */
    void renderCameraControlsWindow();
    
    /**
     * @brief Render point cloud information window
     * 
     * @param renderer Renderer containing point cloud data
     */
    void renderPointCloudInfoWindow();
    
    /**
     * @brief Render performance metrics window
     * 
     * @param frameRate Current frame rate
     * @param memoryUsage Current memory usage
     * @param gpuUsage Current GPU usage
     */
    void renderPerformanceWindow();
    
    /**
     * @brief Render application settings window
     * 
     * @param renderer Renderer for settings that affect rendering
     */
    void renderSettingsWindow();
    
    /**
     * @brief Render debug information window
     * 
     * @param renderer Renderer for debug information
     * @param camera Camera for debug information
     */
    void renderDebugWindow();
}; 