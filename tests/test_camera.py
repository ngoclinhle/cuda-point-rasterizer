import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np


class CameraController:
    def __init__(self):
        self.rotate_sensitivity = np.pi/180 # 1pixel = 1 degree
        self.zoom_sensitivity = 0.1 # 1pixel = 10% distance to target
        self.pan_sensitivity = 0.1
        self.reset()
        
    def reset(self):
        self.target = np.array([0,0,0], dtype=np.float32)
        self.mode = "orbit"
        self.camera = np.eye(4)
        self.radius = 2.0
        self.rot = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2, -np.pi/2, 0])
        self.camera[:3, :3] = self.rot
        self.camera[:3, 3] = [self.radius, 0, 0]
        self.yaw = 0
        self.pitch = 0
    
    def set_mode(self, mode):
        self.mode = mode
    
    def rotate(self, delta_x, delta_y):
        yaw = delta_x * self.rotate_sensitivity
        pitch = -delta_y * self.rotate_sensitivity
        R_yaw = o3d.geometry.get_rotation_matrix_from_xyz([0, yaw, 0])
        R_pitch = o3d.geometry.get_rotation_matrix_from_xyz([pitch, 0, 0])
        R = self.camera[:3, :3] @ R_yaw @ R_pitch
        self.camera[:3, :3] = R
        if self.mode == "orbit":
            cam_to_target = R@np.array([0, 0, 1])
            t = self.target - cam_to_target * self.radius
            self.camera[:3, 3] = t
        else:
            cam_to_target = R@np.array([0, 0, 1])
            t = self.camera[:3, 3]
            self.target = t + cam_to_target * self.radius
    
    def rotate2(self, delta_x, delta_y):
        self.yaw += delta_x * self.rotate_sensitivity
        self.pitch -= delta_y * self.rotate_sensitivity
        R_yaw = o3d.geometry.get_rotation_matrix_from_xyz([0, self.yaw, 0])
        R_pitch = o3d.geometry.get_rotation_matrix_from_xyz([self.pitch, 0, 0])
        R = self.rot @ R_yaw @ R_pitch
        self.camera[:3, :3] = R
        
        if self.mode == "orbit":
            cam_to_target = R@np.array([0, 0, 1])
            t = self.target - cam_to_target * self.radius
            self.camera[:3, 3] = t
        else:
            cam_to_target = R@np.array([0, 0, 1])
            t = self.camera[:3, 3]
            self.target = t + cam_to_target * self.radius
        
    
    def zoom(self, delta):
        R = self.camera[:3, :3]
        dz = np.array([0, 0, 1]) * self.radius * delta * self.zoom_sensitivity
        dt = R@dz
        self.camera[:3, 3] += dt
        if self.mode == "orbit":
            self.radius -= dz
        else:
            self.target += dt
    
    def pan(self, delta_x, delta_y):
        R = self.camera[:3, :3]
        dxy = np.array([delta_x, delta_y, 0]) * self.pan_sensitivity * self.radius
        dt = R@dxy
        self.camera[:3, 3] += dt
        self.target += dt
    
    def get_target(self):
        return self.target

class MainApp:
    def __init__(self):
        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window("Mouse Example", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        self.create_scene()
        
        # Mouse callback
        self.scene.set_on_mouse(self.on_mouse_event)

    def on_mouse_event(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self.last_mouse_pos = np.array([event.x, event.y])
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.type == gui.MouseEvent.Type.DRAG:
            delta = np.array([event.x, event.y]) - self.last_mouse_pos
            self.last_mouse_pos = np.array([event.x, event.y])
            if event.buttons == gui.MouseButton.LEFT.value:
                self.controller.rotate(delta[0], delta[1])
            elif event.buttons == gui.MouseButton.RIGHT.value:
                self.controller.pan(delta[0], delta[1])
            self.update_scene_widget()
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.type == gui.MouseEvent.Type.WHEEL:
            self.controller.zoom(event.wheel_dy)
            self.update_scene_widget()
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED
    
    def create_scene(self):
        self.controller = CameraController()
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        target_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene.scene.add_geometry("world", world_axes, mat)
        self.scene.scene.add_geometry("camera", camera_axes, mat)
        self.scene.scene.add_geometry("target", target_axes, mat)
        self.scene.scene.set_geometry_transform("camera", self.controller.camera)
        self.scene.look_at([0,0,0], [5,5,5], [0,0,1])
        self.labels = []
        self.update_labels()
        
    def update_labels(self):
        if self.labels:
            for label in self.labels:
                self.scene.remove_3d_label(label)
            self.labels = []
            
        world = [0, 0, 0]
        camera_tf = self.scene.scene.get_geometry_transform("camera")
        camera = camera_tf[:3, 3]
        target = self.controller.get_target()
        self.labels.append(self.scene.add_3d_label(world, "world"))
        self.labels.append(self.scene.add_3d_label(camera, "camera"))
        self.labels.append(self.scene.add_3d_label(target, "target"))
        
    def update_scene_widget(self):
        self.scene.scene.set_geometry_transform("camera", self.controller.camera)
        target_tf = np.eye(4)
        target_tf[:3, 3] = self.controller.get_target()
        self.scene.scene.set_geometry_transform("target", target_tf)
        self.update_labels()

    def run(self):
        gui.Application.instance.run()

MainApp().run()
