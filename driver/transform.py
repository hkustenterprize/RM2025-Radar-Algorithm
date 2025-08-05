import numpy as np
import cv2
import time


def lidar2camera(points: np.ndarray, extrinsics: np.ndarray):
    """
    Project the point under the lidar frame to the camera frame, given the extrinsics matrix

    Args:
        points: (N, 3) Input 3D points, under the lidar reference frame
        extrinsics: (4, 4): Homogenous extrinsics matrix
    Returns:
        calibrated_points: (N, 3) Output 3D Points under the camera reference frame
    """
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    points_3d = points[:, :3].astype(np.float32, copy=False)
    # Convert the radar point to homogenous coordinates
    points_homo = np.hstack(
        (points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32))
    )
    # Convert to the point under the camera frame
    calibrted_points = (points_homo @ extrinsics.T.astype(np.float32))[:, :3]
    return calibrted_points


def lidar2uvd(
    points: np.ndarray,
    extrinsics: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the 3D point cloud to the camera plane and retrieve the depth

    Args:
        points: (N, 3), Input 3D points
        extrinsics: (4, 4): Homogenous extrinsics matrix
        camera_matrix: (3, 3) Camera matrix
        dist_coeffs: The distortion coefficients of the camera matrix (optional)

    Returns:
        image_points: (N, 2), Projected 2D points on the image
        depths: (N, 1), Depth values of the points
    """
    camera_points = lidar2camera(points=points, extrinsics=extrinsics)
    # depth > 0
    valid_mask = camera_points[:, 2] > 0
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.float32), camera_points
    # Filter out the valid points
    valid_camera_points = camera_points[valid_mask]
    depths = np.clip(camera_points[:, 2], 0, None)
    # Project the points to the camera plane
    image_points, _ = cv2.projectPoints(
        valid_camera_points,
        np.zeros(3, dtype=np.float32),  # rvec: No additional rotation
        np.zeros(3, dtype=np.float32),  # tvec: No additional displacement
        camera_matrix.astype(np.float32),
        (
            dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float32)
        ),  # Apply camera distortion
    )
    # Prepare the full set of image points
    image_points_full = np.zeros((points.shape[0], 2), dtype=np.float32)
    image_points_full[valid_mask] = image_points.reshape(-1, 2)
    uvds = np.concatenate([image_points_full, depths[:, np.newaxis]], axis=1)
    return uvds


def downsample_uvd(
    depths: np.ndarray,
    image_points: np.ndarray,
    image_height: int,
    image_width: int,
    down_height: int,
    down_width: int,
) -> np.ndarray:
    """
    Generate raw downsampled depth image from lidar points.

    Args:
        depths: (N,) Depth values of the points.
        image_points: (N, 2) Projected 2D points on the image.
        image_height: Original height of the image.
        image_width: Original width of the image.
        down_height: Height of the downsampled depth map.
        down_width: Width of the downsampled depth map.

    Returns:
        depth_map: (down_height, down_width) The raw downsampled depth map.
    """

    scale_x = down_width / image_width
    scale_y = down_height / image_height
    depth_map = np.zeros((down_height, down_width), dtype=np.float32)

    # Return an empty depth map if no valid data is available
    if depths.size == 0 or image_points.size == 0:
        return depth_map

    u_coords = (image_points[:, 0] * scale_x).astype(int)
    v_coords = (image_points[:, 1] * scale_y).astype(int)

    # Create a valid mask to filter points that fall within the downsampled image dimensions
    valid_mask = (
        (u_coords >= 0)
        & (u_coords < down_width)
        & (v_coords >= 0)
        & (v_coords < down_height)
    )

    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depths = depths[valid_mask]

    # Return the empty depth map if no valid points remain after filtering
    if u_coords.size == 0:
        return depth_map

    # Populate the depth map with depth values at valid coordinates
    depth_map[v_coords, u_coords] = depths

    return depth_map
