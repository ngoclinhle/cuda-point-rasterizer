#pragma once

#include <GLFW/glfw3.h>
#include <string>
#include <memory>

// Application components
#include "renderer.h"
#include "camera.h"
#include "camera_controller.h"
#include "ui.h"

/**
 * @brief MainApp coordinates all application components and manages the main loop
 * 
 * The MainApp class is responsible for:
 * - Setting up GLFW, OpenGL, ImGUI, and CUDA
 * - Managing the main application loop
 * - Coordinating updates between UI, CameraController, and Renderer
 * - Handling window events and user input
 */
class MainApp {
    friend class CameraController;
public:
    /**
     * @brief Get the instance of MainApp
     * 
     * @return Reference to the singleton instance
     */
    static MainApp& getInstance();

    /**
     * @brief Destructor
     */
    ~MainApp();

    // Delete copy constructor and assignment operator
    MainApp(const MainApp&) = delete;
    MainApp& operator=(const MainApp&) = delete;

    /**
     * @brief Initialize the application
     * 
     * @param width Initial window width
     * @param height Initial window height
     * @param title Window title
     * @return true if initialization successful, false otherwise
     */
    bool initialize(int width = 1200, int height = 800, const std::string& title = "Point Cloud Rasterizer");

    /**
     * @brief Run the main application loop
     * 
     * This method runs until the user closes the window or an error occurs.
     */
    void run();

    /**
     * @brief Shutdown the application
     */
    void shutdown();

    /**
     * @brief Check if the application should close
     * 
     * @return true if application should exit
     */
    bool shouldClose() const;

private:
    /**
     * @brief Private constructor for singleton pattern
     */
    MainApp();

    // Window and context
    GLFWwindow* window;                 ///< GLFW window handle
    int windowWidth, windowHeight;      ///< Current window dimensions
    std::string windowTitle;            ///< Window title
    
    // Application components
    std::shared_ptr<Renderer> renderer;              ///< Point cloud renderer
    std::shared_ptr<Camera> camera;                  ///< Camera for viewing
    std::shared_ptr<CameraController> cameraController; ///< Camera control
    std::shared_ptr<UI> ui;                          ///< User interface
    
    // Performance metrics
    float frameRate;                    ///< Current frame rate
    size_t memoryUsage;                 ///< Current memory usage
    float gpuUsage;                     ///< Current GPU usage
    double lastFrameTime;               ///< Time of last frame
    
    // Application state
    bool isInitialized;                 ///< Initialization flag
    bool shouldExit;                    ///< Exit flag
    
    /**
     * @brief Initialize GLFW window system
     * 
     * @return true if successful
     */
    bool initGLFW();
    
    /**
     * @brief Initialize OpenGL context
     * 
     * @return true if successful
     */
    bool initOpenGL();
    
    /**
     * @brief Initialize ImGUI
     * 
     * @return true if successful
     */
    bool initImGUI();
    
    /**
     * @brief Initialize CUDA
     * 
     * @return true if successful
     */
    bool initCUDA();
    
    /**
     * @brief Initialize application components
     * 
     * @return true if successful
     */
    bool initComponents();
    
    /**
     * @brief Setup GLFW callbacks
     */
    void setupCallbacks();
    
    /**
     * @brief Process user input
     * 
     * @param deltaTime Time since last frame
     */
    void processInput(float deltaTime);
    
    /**
     * @brief Update application state
     * 
     */
    void update();
    
    /**
     * @brief Render the application
     */
    void render();
    
    /**
     * @brief Update performance metrics
     * 
     * @param deltaTime Time since last frame
     */
    void updatePerformanceMetrics(float deltaTime);
    
    /**
     * @brief Clean up all resources
     */
    void cleanup();
    
    // Static callback functions for GLFW
    static void errorCallback(int error, const char* description);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void charCallback(GLFWwindow* window, unsigned int codepoint);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void windowSizeCallback(GLFWwindow* window, int width, int height);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void dropCallback(GLFWwindow* window, int path_count, const char* paths[]);
    
    /**
     * @brief Handle window resize
     * 
     * @param width New window width
     * @param height New window height
     */
    void handleResize(int width, int height);
    
    /**
     * @brief Get MainApp instance from GLFW window user pointer
     * 
     * @param window GLFW window handle
     * @return Pointer to MainApp instance
     */
    static MainApp* getAppFromWindow(GLFWwindow* window);
}; 