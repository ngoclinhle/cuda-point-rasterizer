#include "main_app.h"
#include <iostream>

int main() {
    std::cout << "Starting Point Cloud Rasterizer..." << std::endl;
    
    // Get singleton instance
    MainApp& app = MainApp::getInstance();
    
    // Initialize the application with parameters
    if (!app.initialize(1200, 800, "Point Cloud Rasterizer")) {
        std::cerr << "Failed to initialize application!" << std::endl;
        return -1;
    }
    
    // Run the main loop
    app.run();
    
    // Shutdown
    app.shutdown();
    
    std::cout << "Application finished." << std::endl;
    return 0;
} 