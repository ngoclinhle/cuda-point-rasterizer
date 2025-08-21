#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

/**
 * @brief Camera-related definitions and utilities
 */

// Mathematical constants
namespace MathConstants {
    const float PI = 3.14159265359f;
    const float DEG_TO_RAD = PI / 180.0f;
    const float RAD_TO_DEG = 180.0f / PI;
}

// Camera intrinsic parameters
struct Camera {

    
    // Default constructor
    Camera();


    /**
     * @brief Get the intrinsics of the camera
     * 
     * @param v_fov: vertical field of view in radians
     * @param w: image width
     * @param h: image height
     */
    void get_intrinsics(float& v_fov, int& w, int& h);

    /**
     * @brief Set the intrinsics of the camera
     * 
     * @param v_fov: vertical field of view in radians
     * @param w: image width
     * @param h: image height
     */
    void set_intrinsics(float v_fov, int w, int h);

    /**
     * @brief Get the focal length of the camera
     * 
     * @return float: focal length
     */
    float get_focal_length() const {return focal_length;}

    /**
     * @brief Get the camera position
     * 
     * @return Eigen::Vector3f: camera position
     */
    Eigen::Vector3f get_pos();

    /**
     * @brief Get the camera rotation
     * 
     * @return Eigen::Matrix3f: camera rotation
     */
    Eigen::Matrix3f get_rot();

    /**
     * @brief Set the camera position
     * 
     * @param pos: camera position
     */
    void set_pos(Eigen::Vector3f pos);

    /**
     * @brief Set the camera rotation
     * 
     * @param rot: camera rotation matrix
     */
    void set_rot(Eigen::Matrix3f rot);

    /**
     * @brief Get the camera euler angles
     * 
     * @return Eigen::Vector3f: camera euler angles
     */
    Eigen::Vector3f get_euler_angles(bool degrees = true);

    /**
     * @brief Set the camera euler angles
     * 
     * @param euler_angles: camera euler angles
     */
    void set_euler_angles(Eigen::Vector3f euler_angles, bool degrees = true);

    /**
     * @brief Get the intrinsic matrix
     * 
     * @return Eigen::Matrix3f: intrinsic matrix
     */
    Eigen::Matrix3f get_K();

    /**
     * @brief Get the extrinsic matrix (world to camera)
     * 
     * @return Eigen::Matrix4f: extrinsic matrix
     */
    Eigen::Matrix4f get_F();

  private:
    float v_fov;
    int width;
    int height;
    float focal_length;
    Eigen::Vector3f pos; // camera position
    Eigen::Matrix3f rot; // camera rotation matrix
};