#include "main_app.h"
#include <GL/gl.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <chrono>
#include <algorithm>

MainApp& MainApp::getInstance() {
    static MainApp instance;
    return instance;
}

MainApp::MainApp()
    : window(nullptr), windowWidth(0), windowHeight(0), windowTitle(""),
      frameRate(0.0f), memoryUsage(0), gpuUsage(0.0f), lastFrameTime(0.0),
      isInitialized(false), shouldExit(false) {
}


MainApp::~MainApp() {
    cleanup();
}


bool MainApp::initialize(int width, int height, const std::string& title) {
    if (isInitialized) {
        return true;
    }

    // Store window parameters
    windowWidth = width;
    windowHeight = height;
    windowTitle = title;

    if (!initGLFW()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    if (!initOpenGL()) {
        std::cerr << "Failed to initialize OpenGL" << std::endl;
        return false;
    }

    if (!initImGUI()) {
        std::cerr << "Failed to initialize ImGUI" << std::endl;
        return false;
    }

    if (!initComponents()) {
        std::cerr << "Failed to initialize components" << std::endl;
        return false;
    }

    setupCallbacks();
    
    isInitialized = true;
    std::cout << "MainApp initialized successfully" << std::endl;
    return true;
}


void MainApp::run() {
    if (!isInitialized) {
        std::cerr << "Application not initialized!" << std::endl;
        return;
    }

    auto lastTime = std::chrono::high_resolution_clock::now();
    
    while (!shouldClose() && !shouldExit) {

        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;


        glfwPollEvents();


        processInput(deltaTime);


        render();


        updatePerformanceMetrics(deltaTime);


        glfwSwapBuffers(window);
    }
}


void MainApp::shutdown() {
    shouldExit = true;
}


bool MainApp::shouldClose() const {
    return window ? glfwWindowShouldClose(window) : true;
}


bool MainApp::initGLFW() {

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }


    glfwSetErrorCallback(errorCallback);


    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif


    window = glfwCreateWindow(windowWidth, windowHeight, windowTitle.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }


    glfwMakeContextCurrent(window);


    glfwSwapInterval(1);


    glfwSetWindowUserPointer(window, this);

    return true;
}


bool MainApp::initOpenGL() {

    glViewport(0, 0, windowWidth, windowHeight);


    glEnable(GL_DEPTH_TEST);


    glClearColor(0.5f, 0.1f, 0.1f, 1.0f);

    return true;
}


bool MainApp::initImGUI() {

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;


    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    
    const char* glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);

    return true;
}


bool MainApp::initCUDA() {

    return true;
}


bool MainApp::initComponents() {
    renderer = std::make_shared<Renderer>(windowWidth, windowHeight);
    camera = std::make_shared<Camera>();
    cameraController = std::make_shared<CameraController>(camera.get());
    ui = std::make_shared<UI>();
    ui->cameraController = cameraController;
    ui->renderer = renderer;
    
    if (renderer && camera) {
        renderer->set_camera(camera.get());
    }
    
    return true;
}


void MainApp::setupCallbacks() {
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCharCallback(window, charCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetDropCallback(window, dropCallback);
}


void MainApp::processInput(float deltaTime) {

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}


void MainApp::update() {
    
}


void MainApp::render() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    renderer->render();
    ui->render();
    // update();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


void MainApp::updatePerformanceMetrics(float deltaTime) {
    if (deltaTime > 0.0f) {
        frameRate = 1.0f / deltaTime;
    }
}


void MainApp::cleanup() {
    if (isInitialized) {
        // Clean up components that depend on OpenGL context FIRST
        ui.reset();
        renderer.reset();
        
        // Then clean up ImGui
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        // Finally destroy GLFW context
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        glfwTerminate();

        isInitialized = false;
    }
}


void MainApp::handleResize(int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glViewport(0, 0, width, height);
}


MainApp* MainApp::getAppFromWindow(GLFWwindow* window) {
    return static_cast<MainApp*>(glfwGetWindowUserPointer(window));
}


void MainApp::errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

void MainApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Pass to ImGui first
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    

    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {

    }
}

void MainApp::charCallback(GLFWwindow* window, unsigned int codepoint) {
    ImGui_ImplGlfw_CharCallback(window, codepoint);
}

void MainApp::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    
    MainApp* app = getAppFromWindow(window);
    ImGuiIO& io = ImGui::GetIO();
    
    if (app && app->cameraController) {
        app->cameraController->on_mouse_button(button, action, mods);
    }
}

void MainApp::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    
    MainApp* app = getAppFromWindow(window);
    ImGuiIO& io = ImGui::GetIO();

    if (app && app->cameraController) {
        Eigen::Vector2f mousePos(static_cast<float>(xpos), static_cast<float>(ypos));
        app->cameraController->on_mouse_move(mousePos);
    }
}

void MainApp::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    
    MainApp* app = getAppFromWindow(window);
    ImGuiIO& io = ImGui::GetIO();

    
    if (app && app->cameraController) {
        app->cameraController->on_scroll(static_cast<float>(yoffset));
    }
}

void MainApp::windowSizeCallback(GLFWwindow* window, int width, int height) {
    MainApp* app = getAppFromWindow(window);
    if (app) {
        app->handleResize(width, height);
    }
}

void MainApp::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void MainApp::dropCallback(GLFWwindow* window, int path_count, const char* paths[]) {
    MainApp* app = getAppFromWindow(window);
    if (!app) {
        return;
    }
    
    if (path_count != 1) {
        std::cout << "Only single file drops are supported. " << path_count << " files dropped." << std::endl;
        return;
    }
    
    std::string filename = paths[0];
    std::cout << "File dropped: " << filename << std::endl;
    
    if (!app->renderer) {
        std::cerr << "Renderer not available" << std::endl;
        return;
    }
    
    // Load the point cloud
    if (app->renderer->set_point_cloud(filename)) {
        // Get the point cloud from renderer to access bounding box
        PointCloud* pointCloud = app->renderer->getPointCloud();
        if (pointCloud && app->cameraController) {
            BoundingBox bbox = pointCloud->get_bounding_box();
            app->cameraController->fitBoundingBox(bbox);
            std::cout << "Camera positioned based on point cloud bounding box" << std::endl;
        }
    }
} 