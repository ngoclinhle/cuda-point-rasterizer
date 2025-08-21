#include "camera.h"
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>


Camera::Camera() 
    : v_fov(M_PI / 4.0f), width(800), height(600), 
      pos(Eigen::Vector3f(0.0f, 0.0f, -0.5f)), 
      rot(Eigen::Matrix3f::Identity()) {
}

// Set camera intrinsics
void Camera::set_intrinsics(float v_fov, int w, int h) {
    this->v_fov = v_fov;
    this->width = w;
    this->height = h;
    this->focal_length = height / (2.0f * tan(v_fov * 0.5f));
}

// Get camera intrinsics
void Camera::get_intrinsics(float& v_fov, int& w, int& h) {
    v_fov = this->v_fov;
    w = this->width;
    h = this->height;
}

Eigen::Vector3f Camera::get_pos() {
    return pos;
}

void Camera::set_pos(Eigen::Vector3f pos) {
    this->pos = pos;
}

Eigen::Matrix3f Camera::get_rot() {
    return rot;
}

void Camera::set_rot(Eigen::Matrix3f rot) {
    this->rot = rot;
}

Eigen::Vector3f Camera::get_euler_angles(bool degrees) {
    auto euler_angles = rot.eulerAngles(2, 1, 0);
    if (degrees) {
        euler_angles *= MathConstants::RAD_TO_DEG;
    }
    return euler_angles;
}

void Camera::set_euler_angles(Eigen::Vector3f euler_angles, bool degrees) {
    if (degrees) {
        euler_angles *= MathConstants::DEG_TO_RAD;
    }
    auto roll = Eigen::AngleAxisf(euler_angles(0), Eigen::Vector3f::UnitX());
    auto pitch = Eigen::AngleAxisf(euler_angles(1), Eigen::Vector3f::UnitY());
    auto yaw = Eigen::AngleAxisf(euler_angles(2), Eigen::Vector3f::UnitZ()); 
    this->rot = (yaw * pitch * roll).toRotationMatrix();
}

Eigen::Matrix3f Camera::get_K() {
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    float f = focal_length;
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    K(0, 0) = f;
    K(0, 2) = cx;
    K(1, 1) = f;
    K(1, 2) = cy;
    return K;
}


Eigen::Matrix4f Camera::get_F() {
    Eigen::Matrix3f R_T = rot.transpose();
    Eigen::Vector3f t_ = R_T * (-pos);
    
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    F.block<3,3>(0,0) = R_T;
    F.block<3,1>(0,3) = t_;
    
    return F;
}
 