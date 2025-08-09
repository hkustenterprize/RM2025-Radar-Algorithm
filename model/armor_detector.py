from model.armor_light_classifier import ArmorLightClassifier
from model.digit_classifier.predictor import DigitClassifier
import PIL
import cv2


class_names = [
    "B1",
    "B2",
    "B3",
    "B4",
    "BS",
    "R1",
    "R2",
    "R3",
    "R4",
    "RS",
    "G1",
    "G2",
    "G3",
    "G4",
    "GS",  # dead
]


class ArmorClassifier:
    def __init__(self, armor_params, digit_model_type, digit_weights_path, debug=False):
        """
        Initialize the ArmorClassifier with armor parameters and digit classifier settings.

        Args:
            armor_params: Parameters for armor plate detection.
            digit_model_type: Type of the digit classifier model (e.g., 'resnet18', 'efficientnet', 'mobilenet').
            digit_weights_path: Path to the weights file for the digit classifier.
            debug: Whether to enable debug mode for additional logging and visualization.
        """
        self.armor_light_classifier = ArmorLightClassifier(armor_params, debug=debug)
        self.digit_classifier = DigitClassifier(
            model_type=digit_model_type,
            weights_path=digit_weights_path,
        )
        self.debug = debug

    def classify(self, img):
        """
        Classify the armor plate in the given image.

        Args:
            img: Input image containing the armor plate.

        Returns:
            A dictionary with detected lights and their classifications.
        """
        # Step 1: Preprocess the image to get binary representation
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        color_result = self.armor_light_classifier.classify_color(bgr_img)
        # Step 2: Classify the lights based on color
        pil_iamge = PIL.Image.fromarray(img).convert("RGB")
        armor_id, conf = self.digit_classifier.predict(pil_iamge, return_names=False)

        if color_result == "RED":
            armor_id += 5
        elif color_result == "GREY":
            armor_id += 10

        armor_name = class_names[armor_id]
        return armor_name, armor_id

    def classify_batch(self, imgs):
        """
        Classify a batch of images containing armor plates.

        Args:
            imgs: List of input images containing armor plates.

        Returns:
            A list of dictionaries with detected lights and their classifications.
        """
        import time

        start = time.time()
        pattern_indxes, confs = self.digit_classifier.predict_batch(
            [PIL.Image.fromarray(img).convert("RGB") for img in imgs],
            return_names=False,
        )
        print(f"[Digit Classifier]: {time.time() - start:.4f} seconds")
        start = time.time()
        max_confs = list(map(max, confs))

        colors = []
        for img in imgs:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            color_result = self.armor_light_classifier.classify_color(bgr_img)
            colors.append(color_result)

        print(f"[Find lights]: {time.time() - start:.4f} seconds")
        armor_names, armor_ids = [], []
        for idx, color in zip(pattern_indxes, colors):
            if color == "RED":
                idx += 5
            elif color == "GREY":
                idx += 10
            armor_names.append(class_names[idx])
            armor_ids.append(idx)

        return armor_names, armor_ids, max_confs


class TwoStepArmorDetectorClassifier:
    """A two-step armor detector and classifier that first detects armor plates
    1. The first step uses an armor detector to find the positions of armor plates in the image. Also returns the colors.
    2. The second step crops the detected armor plates and classifies them using a digit classifier.
    """

    def __init__(self, armor_detector_model, digit_classifier_model):
        """
        Initialize the TwoStepArmorDetectorClassifier with armor detector and digit classifier models.

        Args:
            armor_detector_model: The model used for detecting armor plates.
            digit_classifier_model: The model used for classifying the detected armor plates.
        """
        self.armor_detector_model = armor_detector_model
        self.digit_classifier = digit_classifier_model

    @classmethod
    def from_config(cls, config):
        from model.yolov12.predictor import Predictor

        original_armor_detector = Predictor(
            model_path=config["armor_detector"]["weights_path"],
            img_size=config["armor_detector"]["img_size"],
            max_det=config["armor_detector"]["max_det"],
            conf_thres=config["armor_detector"]["conf_thres"],
            iou_thres=config["armor_detector"]["iou_thres"],
        )
        digit_classifier = DigitClassifier(
            model_type=config["armor_detector"]["digit_model_type"],
            weights_path=config["armor_detector"]["digit_weights_path"],
        )

        armor_detector = TwoStepArmorDetectorClassifier(
            armor_detector_model=original_armor_detector,
            digit_classifier_model=digit_classifier,
        )
        return armor_detector

    def predict_batch(self, imgs):
        """
        保持与YOLOv5Detector相同的接口

        Args:
            imgs: List of numpy arrays (images)

        Returns:
            tuple: (detections_list, annotated_images_list)
                - detections_list: List[List[tuple(class_id, xyxy, confidence)]]
                - annotated_images_list: List of annotated images
        """
        import time

        start = time.time()
        raw_detections_list, annotated_images_list = (
            self.armor_detector_model.predict_batch(imgs)
        )
        print(f"Time for armor position detection: {time.time() - start:.4f} seconds")

        # Step 2: 收集所有装甲板区域
        all_armor_crops = []
        all_detection_info = []

        for img_idx, (img, raw_detections) in enumerate(zip(imgs, raw_detections_list)):
            if not raw_detections:  # 如果没有检测到装甲板
                continue

            # 提取装甲板区域
            for detection in raw_detections:
                color_id, xyxy, confidence = detection
                x1, y1, x2, y2 = map(int, xyxy)

                # 裁剪装甲板区域
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:  # 确保裁剪区域有效
                    all_armor_crops.append(crop)
                    all_detection_info.append((img_idx, xyxy, confidence, color_id))

        # Step 3: 一起分类所有装甲板
        enhanced_detections_list = [[] for _ in imgs]  # 初始化结果列表
        if all_armor_crops:
            # 批量分类所有装甲板
            pattern_indexes, digit_confs = self.digit_classifier.predict_batch(
                [PIL.Image.fromarray(img).convert("RGB") for img in all_armor_crops],
                return_names=False,
            )
            for (img_idx, xyxy, yolo_conf, color_id), pattern_id, digit_conf in zip(
                all_detection_info, pattern_indexes, digit_confs
            ):
                if pattern_id == 5: # 前哨站
                    continue  
                if color_id == 0:
                    armor_id = pattern_id + 10
                elif color_id == 1:
                    armor_id = pattern_id
                else:
                    armor_id = pattern_id + 5

                max_digit_conf = max(digit_conf)
                enhanced_detections_list[img_idx].append(
                    (armor_id, xyxy, max_digit_conf)
                )
        return enhanced_detections_list, annotated_images_list


class ArmorDetectorClassifier:
    def __init__(self, armor_detector_model, armor_classifier_model):
        self.armor_detector_model = armor_detector_model
        self.armor_classifier_model = armor_classifier_model

    def predict_batch(self, imgs):
        """
        保持与YOLOv5Detector相同的接口

        Args:
            imgs: List of numpy arrays (images)

        Returns:
            tuple: (detections_list, annotated_images_list)
                - detections_list: List[List[tuple(class_id, xyxy, confidence)]]
                - annotated_images_list: List of annotated images
        """
        # Step 1: 使用原始armor detector进行检测
        import time

        start = time.time()
        raw_detections_list, annotated_images_list = (
            self.armor_detector_model.predict_batch(imgs)
        )
        print(f"[Armor Detections]: {time.time() - start:.4f}seconds")

        # Step 2: 收集所有装甲板区域
        all_armor_crops = []
        all_detection_info = []  # 存储 (img_idx, xyxy, confidence)

        for img_idx, (img, raw_detections) in enumerate(zip(imgs, raw_detections_list)):
            if not raw_detections:  # 如果没有检测到装甲板
                continue

            # 提取装甲板区域
            for detection in raw_detections:
                original_class_id, xyxy, confidence = detection
                x1, y1, x2, y2 = map(int, xyxy)

                # 裁剪装甲板区域
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:  # 确保裁剪区域有效
                    all_armor_crops.append(crop)
                    all_detection_info.append((img_idx, xyxy, confidence))

        # Step 3: 一起分类所有装甲板
        enhanced_detections_list = [[] for _ in imgs]  # 初始化结果列表

        if all_armor_crops:
            # 批量分类所有装甲板
            armor_names, armor_ids, digit_confs = (
                self.armor_classifier_model.classify_batch(all_armor_crops)
            )

            # Step 4: 将分类结果分配回对应的图像
            for (img_idx, xyxy, original_conf), new_class_id, digit_conf in zip(
                all_detection_info, armor_ids, digit_confs
            ):
                enhanced_detections_list[img_idx].append(
                    (new_class_id, xyxy, digit_conf * original_conf) ## Multiply by original confidence
                )

        return enhanced_detections_list, annotated_images_list

    def predict(self, img):
        """
        单张图像预测，保持接口一致性
        """
        detections_list, annotated_images_list = self.predict_batch([img])
        return detections_list[0], annotated_images_list[0]

