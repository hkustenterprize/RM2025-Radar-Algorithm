from typing import List
import numpy as np


def compute_iou(bbox1: List[float | int], bbox2: List[float | int]):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    """

    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    # Calculate the area of the intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of both bounding boxes
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    # Calculate the area of the union
    union_area = bbox1_area + bbox2_area - inter_area
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def xyxy2xywh(bbox: List[float | int]) -> List[float]:
    """
    Convert bounding box from [x1, y1, x2, y2] to [center_x, center_y, width, height].

    Args:
        bbox: Bounding box in format [x1, y1, x2, y2]

    Returns:
        Bounding box in format [center_x, center_y, width, height]
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [center_x, center_y, width, height]


def xywh2xyxy(bbox: List[float | int]) -> List[float]:
    """
    Convert bounding box from [center_x, center_y, width, height] to [x1, y1, x2, y2].

    Args:
        bbox: Bounding box in format [center_x, center_y, width, height]

    Returns:
        Bounding box in format [x1, y1, x2, y2]
    """
    center_x, center_y, width, height = bbox
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return [x1, y1, x2, y2]


def compute_normalized_center_distance(
    bbox1: List[float | int], bbox2: List[float | int], img: np.ndarray
) -> float:
    """
    Compute the Euclidean distance between the centers of two bounding boxes.

    Args:
        bbox1: Bounding box in format [x1, y1, x2, y2]
        bbox2: Bounding box in format [x1, y1, x2, y2]
        img: Image array to normalize the distance by its dimensions.

    Returns:
        Euclidean distance between the centers of the two bounding boxes.
    """
    height, width = img.shape[:2]
    # Normalize center distance boxes to [0, 1] range
    diagnal_length = (width**2 + height**2) ** 0.5
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
    return (
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    ) ** 0.5 / diagnal_length


def compute_center_distance_xy(bbox1: List[float | int], bbox2: List[float | int]):
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
    return center1[0] - center2[0], center1[1] - center2[1]
