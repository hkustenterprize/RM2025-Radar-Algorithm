import cv2
import numpy as np
from typing import List
from pydantic import BaseModel


class ArmorPlateParams(BaseModel):
    binary_threshold: int = 180
    light_min_ratio: float = 0.07
    light_min_fill_ratio: float = 0.20
    armor_min_small_center_distance: float = 0.8
    armor_max_small_center_distance: float = 3.2
    armor_min_large_center_distance: float = 3.2
    armor_max_large_center_distance: float = 5.5
    armor_min_light_ratio: float = 0.75


class Light:
    def __init__(self, rect, top, bottom, point_count, angle, color):
        self.rect = rect
        self.top = top
        self.bottom = bottom
        self.point_count = point_count
        self.angle = angle
        self.color = color  # RED or BLUE

    def area(self):
        return self.rect[2] * self.rect[3]


class ArmorLightClassifier:

    def __init__(self, params: ArmorPlateParams, debug=False):
        self.params = params
        self.debug = debug

    def grey_scale_and_binary(self, img: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale and then to binary.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Binary image after thresholding.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        av_grey = gray.mean()
        grey_threshold = np.clip(0, int(av_grey * 1.2), 220)
        grey_threshold = self.params.binary_threshold

        _, binary = cv2.threshold(gray, grey_threshold, 255, cv2.THRESH_BINARY)
        mask = np.ones_like(binary, dtype=np.uint8)
        h, w = binary.shape
        mask[int(h * 0.2) : int(h * 0.8), int(w * 0.3) : int(w * 0.7)] = 0
        # mask = ~mask
        binary = binary * mask

        return binary

    def find_lights(
        self,
        binary_img: np.ndarray,
        raw_img: np.ndarray,
    ) -> List[Light]:
        """
        Detect light bars from a binary image.

        Args:
            binary_img (np.ndarray): Binary image from preprocessing.
            raw_img (np.ndarray): Original image for color detection.
            params (ArmorPlateParams): Parameters for light detection.

        Returns:
            List[Light]: List of detected light objects.
        """
        # Find contours
        # Mask the middle area
        # binary_img = cv2.bitwise_not(binary_img)  # Invert binary image
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        lights = []
        for contour in contours:
            if len(contour) < 5:  # Filter small contours
                continue

            # Calculate bounding rect and rotated rect
            b_rect = cv2.boundingRect(contour)
            r_rect = cv2.minAreaRect(contour)

            # Create mask and get points
            mask = np.zeros((b_rect[3], b_rect[2]), dtype=np.uint8)
            mask_contour = contour - [
                b_rect[0],
                b_rect[1],
            ]  # Adjust contour to mask coordinates
            cv2.fillPoly(mask, [mask_contour], 255)
            points = cv2.findNonZero(mask)
            is_fill_rotated_rect = (
                len(points) / (r_rect[1][0] * r_rect[1][1])
                > self.params.light_min_fill_ratio
            )

            # Fit line to get top and bottom points
            if len(points) > 0:
                [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                k = vy / vx if vx != 0 else float("inf")
                b = y0 - k * x0

                top_x = (b_rect[1] - b) / k if k != 0 else b_rect[0] + b_rect[2] / 2
                bottom_x = (
                    (b_rect[1] + b_rect[3] - b) / k
                    if k != 0
                    else b_rect[0] + b_rect[2] / 2
                )
                top = (
                    (top_x + b_rect[0], b_rect[1])
                    if 0 <= top_x <= b_rect[2]
                    else (b_rect[0], b_rect[1])
                )
                bottom = (
                    (bottom_x + b_rect[0], b_rect[1] + b_rect[3])
                    if 0 <= bottom_x <= b_rect[2]
                    else (b_rect[0], b_rect[1] + b_rect[3])
                )
                angle = np.arctan2(vy, vx) * 180 / np.pi
                angle = angle if abs(angle) <= 90 else 180 - abs(angle)
                light = Light(b_rect, top, bottom, len(points), angle, None)

                # Validate light
                ratio = min(b_rect[2], b_rect[3]) / max(b_rect[2], b_rect[3])
                if (
                    ratio > self.params.light_min_ratio
                    # and abs(light.angle) > 45
                    and is_fill_rotated_rect
                    and light.area() > int(raw_img.shape[0] * raw_img.shape[1] * 0.0025)
                ):
                    roi = raw_img[
                        light.rect[1] : light.rect[1] + light.rect[3],
                        light.rect[0] : light.rect[0] + light.rect[2],
                    ]
                    if roi.size > 0:
                        sum_r = np.sum(roi[:, :, 2])  # Red channel
                        sum_b = np.sum(roi[:, :, 0])  # Blue channel
                        light.color = "RED" if sum_r > sum_b else "BLUE"

                    lights.append(light)

        return lights

    def classify_color(self, img) -> str:
        """
        Classify the color of the armor light based on the average color in the image.

        Args:
            img (np.ndarray): Input image containing the armor light.

        Returns:
            str: "RED" or "BLUE" or "GREY" based on the average color.
        """
        binary_image = self.grey_scale_and_binary(img)
        lights = self.find_lights(binary_image, img)
        if not lights:
            color = "GREY"
        else:
            # Majority vote weighted by area
            lights.sort(key=lambda x: x.area(), reverse=True)
            blue_count = sum(light.area() for light in lights if light.color == "BLUE")
            red_count = sum(light.area() for light in lights if light.color == "RED")
            if blue_count > red_count:
                color = "BLUE"
            elif red_count > blue_count:
                color = "RED"
            else:
                color = lights[0].color

        if self.debug:
            debug_img = img.copy()
            for light in lights:
                cv2.rectangle(
                    debug_img,
                    (light.rect[0], light.rect[1]),
                    (light.rect[0] + light.rect[2], light.rect[1] + light.rect[3]),
                    (0, 0, 255) if light.color == "RED" else (255, 0, 0),
                    1,
                )
                cv2.circle(
                    debug_img,
                    (int(light.top[0]), int(light.top[1])),
                    2,
                    (0, 255, 0),
                    -1,
                )
                cv2.circle(
                    debug_img,
                    (int(light.bottom[0]), int(light.bottom[1])),
                    2,
                    (0, 255, 0),
                    -1,
                )
            cv2.imshow("Binary Image", binary_image)
            cv2.imshow("Fit line and Lights", debug_img)
            cv2.waitKey(0)
        return color


if __name__ == "__main__":
    # Example usage
    # image_path = "data/armor/armor_2.jpg"
    image_path = "image.png"
    image = cv2.imread(image_path)
    params = ArmorPlateParams()

    classifier = ArmorLightClassifier(params, debug=True)
    import time

    start = time.time()
    color = classifier.classify_color(image)
    print(f"Detected armor light color: {color} in {time.time() - start:.5f} seconds")
