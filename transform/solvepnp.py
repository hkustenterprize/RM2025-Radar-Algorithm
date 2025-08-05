import json
import numpy as np
import cv2
from typing import Tuple, Optional
import itertools


class PnPSolver:
    def __init__(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, verbose=False
    ):
        """
        Initialize the PnPSolver with camera parameters.

        参数：
            camera_matrix (np.ndarray): 相机内参矩阵 (3x3)
            dist_coeffs (np.ndarray): 畸变系数
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.verbose = verbose

    @classmethod
    def from_config(cls, config: dict | str) -> "PnPSolver":
        """
        从配置字典中创建PnPSolver实例。

        参数：
            config (dict): 包含相机内参和畸变系数的配置字典
        """
        if isinstance(config, str):
            import yaml

            with open(config, "r") as f:
                config = yaml.safe_load(f)
            return cls.from_config(config)

        camera_matrix = np.array(config["transform"]["K"], dtype=np.float32)
        dist_coeffs = np.array(config["transform"]["dist_coeffs"], dtype=np.float32)
        return cls(camera_matrix, dist_coeffs, config["transform"].get("verbose", True))

    def solve(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        使用PnP算法求解相机姿态，支持 M <= N 的情况，通过暴力搜索匹配点对。

        参数：
            object_points (np.ndarray): 3D物体点，形状为 (N, 3)
            image_points (np.ndarray): 2D图像点，形状为 (M, 2)，M <= N

        返回：
            success (bool): 求解是否成功
            R (np.ndarray): 旋转矩阵 (3x3)
            tvec (np.ndarray): 平移向量 (3x1)
            residual (float): 投影残差
        """
        self.object_point = object_points
        self.image_point = image_points

        # 检查输入
        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            raise ValueError(
                "At least 4 points are required for PnP. "
                f"Got {object_points.shape[0]} object points and {image_points.shape[0]} image points."
            )

        N = object_points.shape[0]
        M = image_points.shape[0]
        if M > N:
            raise ValueError(
                "Number of image points cannot exceed number of object points. "
                f"Got {N} object points and {M} image points."
            )

        # M < N：使用暴力搜索匹配
        best_inliers = []
        best_rvec, best_tvec = None, None
        max_inliers = 0
        best_indices = None

        # 枚举所有 M 个 3D 点组合
        import tqdm

        for indices in tqdm.tqdm(itertools.permutations(range(N), M)):
            object_points_matched = object_points[list(indices)]

            # 使用 solvePnPRansac 验证
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points_matched,
                imagePoints=image_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                reprojectionError=6.0,
                confidence=0.99,
                iterationsCount=100,
                flags=cv2.SOLVEPNP_EPNP,
            )

            if success and len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_inliers = inliers
                best_rvec = rvec
                best_tvec = tvec
                best_indices = list(indices)

        # 检查是否找到有效解
        if max_inliers >= 4:
            self.rvec = best_rvec
            self.tvec = best_tvec
            object_points_matched = object_points[best_indices]
            # 使用内点重新优化
            object_points_inliers = object_points_matched[best_inliers.flatten()]
            image_points_inliers = image_points[best_inliers.flatten()]
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=object_points_inliers,
                imagePoints=image_points_inliers,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                rvec=best_rvec,
                tvec=best_tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            self.rvec = rvec
            self.tvec = tvec

            R, _ = cv2.Rodrigues(rvec)
            residual = self.calculate_residual(object_points_matched, image_points)
            if self.verbose:
                print("Best matched 3D point indices:", best_indices)
                print("Inliers:", best_inliers.flatten())
                print("Rotation matrix R:")
                print(R)
                print("\nOffset t:")
                print(tvec)
                print("\nProjection residual (mean):", residual)
            return True, R, tvec, residual
        else:
            if self.verbose:
                print("PnP Solve with brute-force search failed")
            return False, None, None, None

    def solve(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> Tuple[bool, np.ndarray | None, np.ndarray | None, float | None]:
        """
        使用PnP算法求解相机姿态。

        参数：
            object_points (np.ndarray): 3D物体点，形状为 (N, 3)
            image_points (np.ndarray): 2D图像点，形状为 (N, 2)

        返回：
            success (bool): 求解是否成功
            R (np.ndarray): 旋转矩阵 (3x3)
            tvec (np.ndarray): 平移向量 (3x1)
            residual (float): 投影残差
        """
        self.object_point = object_points
        self.image_point = image_points
        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            raise ValueError(
                "At least 4 points are required for PnP. "
                f"Got {object_points.shape[0]} object points and {image_points.shape[0]} image points."
            )
        success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=self.camera_matrix,
        distCoeffs=self.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        )
        self.rvec = rvec
        self.tvec = tvec
        if success:
            R, _ = cv2.Rodrigues(rvec)
            residual = self.calculate_residual(object_points, image_points)
            if self.verbose:
                print("Rotation matrix R:")
                print(R)
                print("\nOffset t:")
                print(tvec)
                print("\nProjection residual (mean):", residual)

            return success, R, tvec, residual
        else:
            if self.verbose:
                print("PnP Solve failed")
                return False, None, None, None
        if object_points.shape[0] != image_points.shape[0]:
            raise ValueError(
                "The number of object points must match the number of image points. "
                f"Got {object_points.shape[0]} object points and {image_points.shape[0]} image points."
            )
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True
        )
        self.rvec = rvec
        self.tvec = tvec
        if success:
            R, _ = cv2.Rodrigues(rvec)
            residual = self.calculate_residual(object_points, image_points)
            if self.verbose:
                print("Rotation matrix R:")
                print(R)
                print("\nOffset t:")
                print(tvec)
                print("\nProjection residual (mean):", residual)

            return success, R, tvec, residual
        else:
            if self.verbose:
                print("PnP Solve failed")
                return False, None, None, None

    def calculate_residual(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> float:
        """
        计算投影残差。

        参数：
            object_points (np.ndarray): 3D物体点，形状为 (N, 3)
            image_points (np.ndarray): 2D图像点，形状为 (N, 2)

        返回：
            residual (float): 投影残差
        """
        if not hasattr(self, "rvec") or not hasattr(self, "tvec"):
            raise ValueError("Please solve the PnP first using the solve method.")
        self.projected_points, _ = cv2.projectPoints(
            objectPoints=object_points,
            rvec=self.rvec,
            tvec=self.tvec,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
        )
        self.projected_points = self.projected_points.reshape(-1, 2)
        residual = np.linalg.norm(self.projected_points - image_points, axis=1)
        return np.mean(residual)

    def draw_visualize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Draw the projected points on the image for visualization.
        参数：
            img (np.ndarray): 输入图像
        返回：
            img (np.ndarray): 绘制了投影点的图像
            Red for projected points, green for original 2D points.
        """
        img = img.copy()
        if not hasattr(self, "projected_points"):
            raise ValueError("Please solve the PnP first using the solve method.")

        for point in self.projected_points:
            x, y = point.astype(int)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        for point in self.image_point:
            x, y = point.astype(int)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        return img


if __name__ == "__main__":
    selected_indices = np.array([1, 8, 11, 12, 16, 18])
    with open("test_assets/calib/0_20.json", "r") as f:
        image_points_dict = json.load(f)
        image_points = np.array(
            [image_points_dict[str(i)] for i in range(1, len(image_points_dict) + 1)], dtype=np.float32
        )

    object_points = np.loadtxt("transform/keypoint_6.txt", dtype=np.float32)
    # object_points = object_points[selected_indices]
    image_points = image_points[selected_indices]
    print("Object points shape:", object_points)
    print("Image points shape:", image_points)
    import yaml

    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    pnpsolver = PnPSolver.from_config(config)
    success, R, tvec, residual = pnpsolver.solve(object_points, image_points)
    vis_img = pnpsolver.draw_visualize_image(cv2.imread("test_assets/calib/0.jpg"))
    vis_img = cv2.resize(vis_img, (1536, 1024))
    cv2.imwrite("PnP.png", vis_img)

    # cv2.waitKey(0)
