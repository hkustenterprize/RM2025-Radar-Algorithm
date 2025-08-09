import numpy as np
import open3d as o3d
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2
import time


class PixelToWorld:
    def __init__(self, camera_matrix, R, T, mesh, dist_coeffs=None, max_octree_depth=8):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = (
            np.zeros(5)
            if dist_coeffs is None
            else np.array(dist_coeffs, dtype=np.float64)
        )
        self.R = np.array(R, dtype=np.float64)
        self.T = np.array(T, dtype=np.float64).reshape(3, 1)
        self.mesh = mesh
        self.mesh.paint_uniform_color([1, 1, 0])  # Yellow mesh surface
        # Create LineSet for mesh edges instead of point cloud
        self.edge_lineset = o3d.geometry.LineSet()
        vertices = np.asarray(self.mesh.vertices)
        # Create LineSet for mesh edges
        self.edge_lineset = o3d.geometry.LineSet()
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        # Manually extract edges from triangles
        # Manually extract edges from triangles
        edges = set()
        for tri in triangles:
            # Each triangle has three edges: (v0,v1), (v1,v2), (v2,v0)
            edges.add(tuple(sorted([tri[0], tri[1]])))  # v0-v1
            edges.add(tuple(sorted([tri[1], tri[2]])))  # v1-v2
            edges.add(tuple(sorted([tri[2], tri[0]])))  # v2-v0
        edges = np.array(list(edges))  # Convert to numpy array

        self.edge_lineset.points = o3d.utility.Vector3dVector(vertices)
        self.edge_lineset.lines = o3d.utility.Vector2iVector(edges)
        self.edge_lineset.paint_uniform_color([1, 0, 1])  # Purple edges

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))

    @classmethod
    def build_from_config(self, config):
        pixel_world_transform = PixelToWorld(
            camera_matrix=np.array(config["transform"]["K"]),
            R=np.array(config["transform"]["R"]),
            T=np.array(config["transform"]["t"]),
            dist_coeffs=np.array(config["transform"]["dist_coeffs"]),
            mesh=o3d.io.read_triangle_mesh(config["field"]["mesh_path"]),
        )
        return pixel_world_transform

    @classmethod
    def build_from_config_and_extrinsics(self, config, R, T):
        pixel_world_transform = PixelToWorld(
            camera_matrix=np.array(config["transform"]["K"]),
            R=np.array(R),  # Calibrated R
            T=np.array(T),  # Calibrated T
            dist_coeffs=np.array(config["transform"]["dist_coeffs"]),
            mesh=o3d.io.read_triangle_mesh(config["field"]["mesh_path"]),
        )
        return pixel_world_transform

    def pixel_to_world(self, pixel):
        u, v = pixel
        pixel_hom = np.array([u, v, 1.0], dtype=np.float64)
        if self.dist_coeffs is not None and not np.all(self.dist_coeffs == 0):
            points = np.array([[u, v]], dtype=np.float32).reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(
                points, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
            )
            u, v = undistorted[0, 0]
            pixel_hom = np.array([u, v, 1.0], dtype=np.float64)
        cam_dir = np.linalg.inv(self.camera_matrix) @ pixel_hom
        world_dir = self.R.T @ cam_dir
        origin = -self.R.T @ self.T.flatten()
        rays = o3d.core.Tensor([[*origin, *world_dir]], dtype=o3d.core.Dtype.Float32)
        result = self.scene.cast_rays(rays)
        t_hit = result["t_hit"].numpy()[0]
        # print(
        #     f"Pixel: ({u:.1f}, {v:.1f}), Ray Center: {origin}, Ray Direction: {world_dir}, Hit Distance: {t_hit}"
        # )
        if t_hit < float("inf"):
            hit_point = origin + t_hit * world_dir
            return hit_point
        return None

    def __call__(self, pixel):
        """
        Args: pixel(u, v) on image
        Returns: (x, y, z) in 3d
        """
        return self.pixel_to_world(pixel)

    def get_ray_geometry(self, pixel, ray_length=30.0):
        u, v = pixel
        pixel_hom = np.array([u, v, 1.0], dtype=np.float64)
        if self.dist_coeffs is not None and not np.all(self.dist_coeffs == 0):
            points = np.array([[u, v]], dtype=np.float32).reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(
                points, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
            )
            u, v = undistorted[0, 0]
            pixel_hom = np.array([u, v, 1.0], dtype=np.float64)
        cam_dir = np.linalg.inv(self.camera_matrix) @ pixel_hom
        world_dir = self.R.T @ cam_dir
        origin = -self.R.T @ self.T.flatten()
        origin_pcd = o3d.geometry.PointCloud()
        origin_pcd.points = o3d.utility.Vector3dVector([origin])
        origin_pcd.paint_uniform_color([0, 1, 0])  # 绿色
        ray_end = origin + ray_length * world_dir / np.linalg.norm(world_dir)
        ray_line = o3d.geometry.LineSet()
        ray_line.points = o3d.utility.Vector3dVector([origin, ray_end])
        ray_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        ray_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
        rays = o3d.core.Tensor([[*origin, *world_dir]], dtype=o3d.core.Dtype.Float32)
        result = self.scene.cast_rays(rays)
        t_hit = result["t_hit"].numpy()[0]
        geometries = [origin_pcd, ray_line]
        if t_hit < float("inf"):
            hit_point = origin + t_hit * world_dir
            hit_pcd = o3d.geometry.PointCloud()
            hit_pcd.points = o3d.utility.Vector3dVector([hit_point])
            hit_pcd.paint_uniform_color([0, 0, 1])  # 蓝色（改为与网格区分）
            geometries.append(hit_pcd)
        return geometries


class PixelToWorldGUI:

    def __init__(self, root, image_path, converter, scale_factor=1.0, depth_scale=0.5):
        self.root = root
        self.root.title("Pixel to World Coordinate Converter")
        self.converter = converter
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.is_drawing = False
        self.pixel_trajectory = []
        self.world_trajectory = []
        self.canvas_lines = []

        # 加载图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        new_size = (
            int(self.image.shape[1] * scale_factor),
            int(self.image.shape[0] * scale_factor),
        )
        self.image_scaled = cv2.resize(
            self.image, new_size, interpolation=cv2.INTER_LINEAR
        )
        self.image_display = self.image_scaled.copy()

        # 按钮框架
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        # Tkinter 画布
        self.image_pil = Image.fromarray(self.image_scaled)
        self.image_tk = ImageTk.PhotoImage(self.image_pil)
        self.canvas = tk.Canvas(root, width=new_size[0], height=new_size[1])
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        # 坐标标签
        self.coord_label = tk.Label(
            root,
            text="Select the point by left click on the image pixel (Right drag to draw, R to reset)",
            font=("Arial", 12),
        )
        self.coord_label.pack()

        # 重置按钮
        self.reset_button = tk.Button(
            root, text="Reset Trajectory", command=self.reset_trajectory
        )
        self.reset_button.pack()
        self.depth_button = tk.Button(
            self.button_frame,
            text="Generate Depth Map",
            command=self.generate_depth_map,
        )
        self.depth_button.pack(side=tk.LEFT, padx=5)
        self.depth_scale = depth_scale

        # OpenCV 窗口

        # Open3D 可视化
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Ray and Vertex Visualization", width=800, height=600
        )
        self.vis.add_geometry(self.converter.mesh)
        # self.vis.add_geometry(self.converter.vertex_pcd)
        self.vis.add_geometry(self.converter.edge_lineset)
        self.vis.get_render_option().point_size = 3.0
        self.vis.get_render_option().mesh_show_back_face = True
        self.vis.get_render_option().background_color = np.array([0.2, 0.2, 0.2])
        self.vis.get_render_option().line_width = 8.0  # 设置线条宽度

        # 绑定事件
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B3-Motion>", self.on_right_motion)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.root.bind("r", lambda event: self.reset_trajectory())
        self.root.bind("R", lambda event: self.reset_trajectory())

        # 打印网格范围
        vertices = np.asarray(self.converter.mesh.vertices)
        print(
            f"网格范围: min={np.min(vertices, axis=0)}, max={np.max(vertices, axis=0)}"
        )

        # 启动 Open3D 事件循环
        self.update_visualizer()

    def update_visualizer(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        self.root.after(10, self.update_visualizer)

    def reset_trajectory(self):
        self.is_drawing = False
        self.pixel_trajectory = []
        self.world_trajectory = []
        self.canvas_lines = []
        self.canvas.delete("trajectory")
        self.image_display = self.image_scaled.copy()
        self.vis.clear_geometries()
        self.vis.add_geometry(self.converter.mesh)
        self.vis.add_geometry(self.converter.edge_lineset)  # Changed from vertex_pcd
        self.coord_label.config(
            text="Trajectory reset. Left click to select pixel, right drag to draw"
        )

    def on_left_click(self, event):
        if self.is_drawing:
            self.is_drawing = False
            self.pixel_trajectory = []
            self.world_trajectory = []
            self.canvas_lines = []
            self.canvas.delete("trajectory")
            self.image_display = self.image_scaled.copy()
            self.coord_label.config(
                text="Exiting drawing mode. Left Click to select pixel."
            )

            return

        u_scaled, v_scaled = event.x, event.y
        u_orig = u_scaled / self.scale_factor
        v_orig = v_scaled / self.scale_factor
        pixel = (u_orig, v_orig)
        world_coord = self.converter.pixel_to_world(pixel)
        if world_coord is not None:
            coord_text = f"Pixel ({u_orig:.1f}, {v_orig:.1f}) -> 3D Coor: ({world_coord[0]:.3f}, {world_coord[1]:.3f}, {world_coord[2]:.3f})"
            cv2.circle(
                self.image_display, (int(u_scaled), int(v_scaled)), 5, (255, 0, 0), -1
            )
            cv2.putText(
                self.image_display,
                f"({world_coord[0]:.1f}, {world_coord[1]:.1f}, {world_coord[2]:.1f})",
                (int(u_scaled) + 10, int(v_scaled)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * self.scale_factor,
                (0, 255, 0),
                1,
            )
        else:
            coord_text = f"Pixel ({u_orig:.1f}, {v_orig:.1f}) No valid 3d points"
        self.coord_label.config(text=coord_text)
        self.vis.clear_geometries()
        self.vis.add_geometry(self.converter.mesh)
        self.vis.add_geometry(self.converter.edge_lineset)  # Changed from vertex_pcd
        ray_geometries = self.converter.get_ray_geometry(pixel)
        for geom in ray_geometries:
            self.vis.add_geometry(geom)

    def on_right_click(self, event):
        self.is_drawing = True
        self.pixel_trajectory = []
        self.world_trajectory = []
        self.canvas_lines = []
        self.canvas.delete("trajectory")
        self.image_display = self.image_scaled.copy()
        u_scaled, v_scaled = event.x, event.y
        u_orig = u_scaled / self.scale_factor
        v_orig = v_scaled / self.scale_factor
        pixel = (u_orig, v_orig)
        world_coord = self.converter.pixel_to_world(pixel)
        if world_coord is not None:
            self.pixel_trajectory.append((u_scaled, v_scaled))
            self.world_trajectory.append(world_coord)
            print(
                f"Start to draw line: Pixel=({u_scaled:.1f}, {v_scaled:.1f}), 3D=({world_coord})"
            )
        self.coord_label.config(
            text="Drawing: Right drag to draw, left click to exit, R to reset"
        )

    def on_right_motion(self, event):
        if not self.is_drawing:
            return
        u_scaled, v_scaled = event.x, event.y
        u_orig = u_scaled / self.scale_factor
        v_orig = v_scaled / self.scale_factor
        pixel = (u_orig, v_orig)
        world_coord = self.converter.pixel_to_world(pixel)
        if world_coord is not None and len(self.pixel_trajectory) > 0:
            prev_pixel = self.pixel_trajectory[-1]
            # Tkinter 画布绘制轨迹
            line_id = self.canvas.create_line(
                prev_pixel[0],
                prev_pixel[1],
                u_scaled,
                v_scaled,
                fill="red",
                width=2,
                tags="trajectory",
            )
            self.canvas_lines.append(line_id)
            # OpenCV 同步绘制
            cv2.line(
                self.image_display,
                (int(prev_pixel[0]), int(prev_pixel[1])),
                (int(u_scaled), int(v_scaled)),
                (255, 0, 0),
                2,
            )
            self.pixel_trajectory.append((u_scaled, v_scaled))
            self.world_trajectory.append(world_coord)
            self.update_3d_trajectory()
            print(
                f"Trajactory points: Pixel=({u_scaled:.1f}, {v_scaled:.1f}), 3D=({world_coord}), Total points={len(self.world_trajectory)}"
            )

    def on_right_release(self, event):
        if self.is_drawing:
            self.update_3d_trajectory()
            print(f"Trajectory ends: {len(self.world_trajectory)}")

    def update_3d_trajectory(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(self.converter.mesh)
        self.vis.add_geometry(self.converter.edge_lineset)  # Changed from vertex_pcd
        if len(self.world_trajectory) > 1:
            points = np.array(self.world_trajectory)
            lines = [[i, i + 1] for i in range(len(points) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(
                [[0, 0, 1]] * len(lines)
            )  # Blue
            self.vis.add_geometry(line_set)
            print(f"3D Trajactory updates: {len(lines)}")

    def generate_depth_map(self):
        start_time = time.time()
        self.coord_label.config(text="Generating depth map, please wait...")
        self.root.update()

        # 获取原图像尺寸
        h_orig, w_orig = self.image.shape[:2]
        # 下采样尺寸
        h_depth = int(h_orig * self.depth_scale)
        w_depth = int(w_orig * self.depth_scale)
        depth_map = np.zeros((h_depth, w_depth), dtype=np.float32)

        # 批量生成像素坐标
        v, u = np.indices((h_depth, w_depth), dtype=np.float32)
        u = u / self.depth_scale  # 还原到原图像坐标
        v = v / self.depth_scale
        pixels = np.stack([u.flatten(), v.flatten()], axis=1)

        # 批量射线投射
        pixel_hom = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
        cam_dirs = np.linalg.inv(self.converter.camera_matrix) @ pixel_hom.T
        world_dirs = self.converter.R.T @ cam_dirs
        origin = -self.converter.R.T @ self.converter.T.flatten()
        origins = np.tile(origin, (pixels.shape[0], 1))
        rays = np.hstack([origins, world_dirs.T])
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        result = self.converter.scene.cast_rays(rays_tensor)
        t_hit = result["t_hit"].numpy().reshape(h_depth, w_depth)

        # 填充深度图
        depth_map = t_hit
        depth_map[np.isinf(depth_map)] = 0  # 未命中设为 0

        # 归一化到 [0, 255]
        valid_depths = depth_map[depth_map > 0]
        if valid_depths.size > 0:
            min_depth = valid_depths.min()
            max_depth = valid_depths.max()
            depth_norm = np.zeros_like(depth_map)
            mask = depth_map > 0
            depth_norm[mask] = (
                255 * (depth_map[mask] - min_depth) / (max_depth - min_depth)
            )
            depth_norm = depth_norm.astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)

        # 上采样到原尺寸
        depth_norm = cv2.resize(
            depth_norm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
        )

        # 显示和保存
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite("depth_map.png", depth_colormap)

        elapsed_time = time.time() - start_time
        self.coord_label.config(
            text=f"Depth map generated in {elapsed_time:.2f}s. Min depth: {min_depth:.2f}, Max depth: {max_depth:.2f}"
        )

    def __del__(self):
        cv2.destroyAllWindows()
        self.vis.destroy_window()


if __name__ == "__main__":
    import yaml
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    converter = PixelToWorld.build_from_config(
        config
    )
    # 创建交互界面
    root = tk.Tk()
    app = PixelToWorldGUI(root, config["transform"]["demo_img_path"], converter, scale_factor=0.5)
    plt.ion()  # 开启Matplotlib交互模式
    root.mainloop()
