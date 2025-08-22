#pragma once

#include <Eigen/Dense>
#include <point_cloud_2.h>

struct Camera;
struct MainApp;
class UI;

class CameraController {
    friend class MainApp;
    friend class UI;

  public:
    enum class Mode {
        Model,
        Orbit
    };
    CameraController(Camera *camera, Mode mode = Mode::Orbit);
    void set_world_up(const Eigen::Vector3f& world_up);
  
    void set_mode(Mode mode);
    Mode get_mode() const { return mode; }
    void set_camera(Camera *camera);
    Camera* get_camera() const { return camera; }
    void get_sensitivity(float& rotation_sensitivity, float& zoom_sensitivity) const;
    void set_sensitivity(float rotation_sensitivity, float zoom_sensitivity);
    void fitBoundingBox(const BoundingBox& bbox);

    void look_at(Eigen::Vector3f target, Eigen::Vector3f up, Eigen::Vector3f pos);
    
    // move and rotate the camera around orbit; update orbit azimuth, elevation
    void rotate(float delta_x, float delta_y);
    // move the camera towards or away from the target; update orbit distance
    void zoom(float delta_y);
    // move the camera; update orbit target
    void pan(float delta_x, float delta_y);
    
  
  private:  
    Mode mode;
    Camera *camera;
    Eigen::Vector2f last_mouse_pos;
    bool is_rotating;
    bool is_panning;

    Eigen::Vector3f world_up;
    
    float rotation_sensitivity;
    float zoom_sensitivity;
    
    Eigen::Matrix3f m_rot;
    Eigen::Vector3f m_target;
    float m_radius;

    void on_mouse_move(Eigen::Vector2f pos);
    void on_mouse_button(int button, int action, int mods);
    void on_scroll(float delta);
};