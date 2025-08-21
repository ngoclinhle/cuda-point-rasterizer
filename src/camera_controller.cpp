#include "camera_controller.h"
#include "camera.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <GLFW/glfw3.h>
#include "main_app.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CameraController::CameraController(Camera *camera, Mode mode) 
    : camera(camera), mode(mode),
      last_mouse_pos(Eigen::Vector2f(0.0f, 0.0f)),
      is_rotating(false),
      is_panning(false),
      world_up(Eigen::Vector3f(0.0f, 0.0f, 1.0f)),
      m_target(Eigen::Vector3f(0.0f, 0.0f, 0.0f)),
      m_radius(2.0f)
{
    rotation_sensitivity = M_PI/180.0f;
    zoom_sensitivity = 0.1f;
    m_rot = (Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitX()) 
        * Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitY())).toRotationMatrix();
}

// Set the control mode
void CameraController::set_mode(Mode mode) {
    this->mode = mode;
}

// Set the camera pointer
void CameraController::set_camera(Camera *camera) {
    this->camera = camera;
}

void CameraController::set_world_up(const Eigen::Vector3f& world_up) {
    this->world_up = world_up;
}

void CameraController::get_sensitivity(float& rotation_sensitivity, float& zoom_sensitivity) const {
    rotation_sensitivity = this->rotation_sensitivity;
    zoom_sensitivity = this->zoom_sensitivity;
}

void CameraController::set_sensitivity(float rotation_sensitivity, float zoom_sensitivity) {
    this->rotation_sensitivity = rotation_sensitivity;
    this->zoom_sensitivity = zoom_sensitivity;
}

void CameraController::look_at(Eigen::Vector3f target, Eigen::Vector3f up, Eigen::Vector3f pos) {
    m_target = target;
    m_radius = (pos - target).norm();
    Eigen::Vector3f target_to_cam = pos - target;
    float yaw = -atan2(target_to_cam.y(), target_to_cam.x());
    float pitch = -asin(target_to_cam.z() / m_radius);
    Eigen::AngleAxisf R_yaw(yaw, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf R_pitch(pitch, Eigen::Vector3f::UnitX());
    Eigen::Matrix3f R = m_rot * (R_yaw * R_pitch).toRotationMatrix();
    camera->set_rot(R);
    camera->set_pos(pos);
}

// Fit camera to bounding box
void CameraController::fitBoundingBox(const BoundingBox& bbox) {
    if (!camera) return;
    
    // Calculate bounding box center
    float3 min_pt = bbox.first;
    float3 max_pt = bbox.second;
    
    Eigen::Vector3f center(
        (min_pt.x + max_pt.x) * 0.5f,
        (min_pt.y + max_pt.y) * 0.5f,
        (min_pt.z + max_pt.z) * 0.5f
    );
    
    // Calculate bounding box diagonal for optimal viewing distance
    Eigen::Vector3f bbox_size(
        max_pt.x - min_pt.x,
        max_pt.y - min_pt.y,
        max_pt.z - min_pt.z
    );
    
    Eigen::Vector3f pos = center + bbox_size * 1.5;
    look_at(center, world_up, pos);
}


// Rotate camera around orbit target using spherical offset
void CameraController::rotate(float delta_x, float delta_y) {
    if (!camera) return;

    float yaw = delta_x * rotation_sensitivity;
    float pitch = -delta_y * rotation_sensitivity;
    Eigen::AngleAxisf R_yaw(yaw, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf R_pitch(pitch, Eigen::Vector3f::UnitX());
    Eigen::Matrix3f R = camera->get_rot() * (R_yaw * R_pitch).toRotationMatrix();
    camera->set_rot(R);

    if (mode == Mode::Orbit) {
        Eigen::Vector3f cam_to_target = R * Eigen::Vector3f(0, 0, 1);
        Eigen::Vector3f t = m_target - cam_to_target * m_radius;
        camera->set_pos(t);
    } else {
        Eigen::Vector3f cam_to_target = R * Eigen::Vector3f(0, 0, 1);
        Eigen::Vector3f t = camera->get_pos();
        m_target = t + cam_to_target * m_radius;
    }
}

// Zoom camera toward/away from target
void CameraController::zoom(float delta_y) {
    if (!camera) return;
    Eigen::Matrix3f R = camera->get_rot();
    float dz = m_radius * delta_y * zoom_sensitivity;
    Eigen::Vector3f dt = R * Eigen::Vector3f::UnitZ() * dz;
    Eigen::Vector3f t = camera->get_pos() + dt;
    camera->set_pos(t);
    if (mode == Mode::Orbit) {
        m_radius = std::max(m_radius - dz, 0.01f);
    } else {
        m_target += dt;
    }
}

// Pan the orbit target in camera's screen space
void CameraController::pan(float delta_x, float delta_y) {
    if (!camera) return;
    float f = camera->get_focal_length(); //should move the target 1 pixel
    float speed = m_radius / f;
    Eigen::Matrix3f R = camera->get_rot();
    Eigen::Vector3f dxy = Eigen::Vector3f(-delta_x, -delta_y, 0) *speed;
    Eigen::Vector3f dt = R * dxy;
    Eigen::Vector3f t = camera->get_pos() + dt;
    camera->set_pos(t);
    m_target += dt;
}

// Handle mouse movement for intrinsic orbit navigation
void CameraController::on_mouse_move(Eigen::Vector2f pos) {
    Eigen::Vector2f delta = pos - last_mouse_pos;
    last_mouse_pos = pos;
    
    MainApp& app = MainApp::getInstance();
    if (!app.ui->isViewportHovered()) {
        is_rotating = false;
        is_panning = false;
        return;
    }
    
    if (is_rotating) {
        rotate(delta.x(), delta.y());
    } else if (is_panning) {
        pan(delta.x(), delta.y());
    }

}

// Handle mouse button events
void CameraController::on_mouse_button(int button, int action, int mods) {
    MainApp& app = MainApp::getInstance();
    if (!app.ui->isViewportHovered()) {
        return;
    }

    is_rotating = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    is_panning = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void CameraController::on_scroll(float delta) {
    MainApp& app = MainApp::getInstance();
    if (!app.ui->isViewportHovered()) {
        return;
    }

    zoom(delta);
}
