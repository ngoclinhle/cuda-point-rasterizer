#include "ui.h"
#include <iostream>
#include <GL/gl.h>
#include <vector>
#include <cmath>

UI::UI() 
    : showViewport(true)
    , showCameraControls(true)
    , showPointCloudInfo(false)
    , showPerformance(false)
    , showSettings(true)
    , showDebug(false)
    , viewport_hovered(false)
{
}

void UI::render() {
    // Main menu bar
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Viewport", nullptr, &showViewport);
            ImGui::MenuItem("Camera Controls", nullptr, &showCameraControls);
            ImGui::MenuItem("Point Cloud Info", nullptr, &showPointCloudInfo);
            ImGui::MenuItem("Performance", nullptr, &showPerformance);
            ImGui::MenuItem("Settings", nullptr, &showSettings);
            ImGui::MenuItem("Debug", nullptr, &showDebug);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    
    renderViewportWindow();
    renderCameraControlsWindow();
    renderPointCloudInfoWindow();
    renderPerformanceWindow();
    renderSettingsWindow();
    renderDebugWindow();
}

void UI::renderViewportWindow() {
    if (!showViewport || !renderer) {
        return;
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    
    // Window flags to make it fill the viewport without decorations
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration 
                                  | ImGuiWindowFlags_NoMove 
                                  | ImGuiWindowFlags_NoResize 
                                  | ImGuiWindowFlags_NoSavedSettings
                                  | ImGuiWindowFlags_NoBringToFrontOnFocus
                                  | ImGuiWindowFlags_NoNavFocus;
    
    // Create the viewport window
    if (ImGui::Begin("Main Image View", &showViewport, window_flags)) {
        viewport_hovered = ImGui::IsWindowHovered();
        
        ImVec2 content_region = ImGui::GetContentRegionAvail();
        
        // Get the texture from renderer
        GLuint texture_id = renderer->getTexture();
        auto width = renderer->getWidth();
        auto height = renderer->getHeight();
        
        
        if (texture_id != 0 && content_region.x > 0 && content_region.y > 0) {
            ImGui::Image((ImTextureID)(intptr_t)(texture_id), ImVec2(width, height));   
        } 
        
    }
    ImGui::End();
}

bool UI::isViewportHovered() const {
    return viewport_hovered;
}

void UI::renderCameraControlsWindow() {
    if (!showCameraControls || !cameraController) {
        return;
    }
    ImGui::SetNextWindowSize(ImVec2(0, 0));

    if (ImGui::Begin("Camera Controls", &showCameraControls, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (ImGui::BeginCombo("Mode", cameraController->get_mode() == CameraController::Mode::Model ? "Model" : "Orbit")) {
            if (ImGui::Selectable("Model", cameraController->get_mode() == CameraController::Mode::Model)) {
                cameraController->set_mode(CameraController::Mode::Model);
            }
            if (ImGui::Selectable("Orbit", cameraController->get_mode() == CameraController::Mode::Orbit)) {
                cameraController->set_mode(CameraController::Mode::Orbit);
            }
            ImGui::EndCombo();
        }
        ImGui::Text("Speed");
        float zoom_speed, rotate_speed;
        cameraController->get_sensitivity(rotate_speed, zoom_speed);
        bool changed = false;
        changed |= ImGui::DragFloat("Rotation", &rotate_speed, 0.0001f, 0.001f, 1.0f, "%.4f");
        changed |= ImGui::DragFloat("Zoom", &zoom_speed, 0.0001f, 0.001f, 1.0f, "%.4f");
        if (changed) {
            cameraController->set_sensitivity(rotate_speed, zoom_speed);
        }

        ImGui::Separator();
        ImGui::Text("Camera pose");
        auto camera = cameraController->get_camera();
        auto current_pos = camera->get_pos();
        auto current_euler_angles = camera->get_euler_angles();
        ImGui::Text("Position: %.2f, %.2f, %.2f", current_pos.x(), current_pos.y(), current_pos.z());
        ImGui::Text("Euler angles: %.2f, %.2f, %.2f", current_euler_angles.x(), current_euler_angles.y(), current_euler_angles.z());

        if (ImGui::Button("Fit Point Cloud")) {
            reset_camera();
        }

        ImGui::Separator();
        ImGui::Text("Orbit Controls");
        ImGui::Text("Radius: %.2f", cameraController->m_radius);
        ImGui::Text("Target: %.2f, %.2f, %.2f", cameraController->m_target.x(), cameraController->m_target.y(), cameraController->m_target.z());
        ImGui::Text("World Up: %.2f, %.2f, %.2f", cameraController->world_up.x(), cameraController->world_up.y(), cameraController->world_up.z());
        ImGui::End();
    }
}

void UI::reset_camera() {
    auto pcd = renderer->getPointCloud();
    auto bbox = pcd->get_bounding_box();
    cameraController->fitBoundingBox(bbox);
}

void UI::renderPointCloudInfoWindow() {
    // do nothing
}

void UI::renderPerformanceWindow() {
    // do nothing
}

void UI::renderSettingsWindow() {
    if (!showSettings) {
        return;
    }
    ImGui::Begin("Settings", &showSettings);
    bool visible_filter;
    float visible_threshold;
    renderer->get_visible_filter(visible_filter, visible_threshold);
    bool change = false;
    change |= ImGui::Checkbox("Visible Filter", &visible_filter);
    change |= ImGui::SliderFloat("Visible Threshold", &visible_threshold, 0.0f, 8.0f);
    if (change) {
        renderer->set_visible_filter(visible_filter, visible_threshold);
    }
    ImGui::End();
}

void UI::renderDebugWindow() {
    if (!showDebug) {
        return;
    }
    ImGui::ShowDemoWindow(&showDebug);
    ImGui::ShowMetricsWindow(&showDebug);
}