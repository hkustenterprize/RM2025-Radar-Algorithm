from model.pfa_yolov5.predictor import YOLOv5Detector
from model.yolov12.predictor_with_tracker import PredictorWithTracker
from tracker.CascadeMatchTracker.type import TrackingState, SingleDetectionResult
from typing import List, Optional
import numpy as np
import cv2
import time
from model.armor_detector import TwoStepArmorDetectorClassifier


class BaseDetector:

    def __init__(self, config, pixel_world_transform, visualize=False):
        super().__init__()
        self.device = config["device"]
        self.class_names = config["armor_detector"]["class_names"]
        self.config = config

        self.car_detector = PredictorWithTracker(
            model_path=config["car_detector"]["weights_path"],
            img_size=config["car_detector"]["img_size"],
            max_det=config["car_detector"]["max_det"],
            conf_thres=config["car_detector"]["conf_thres"],
            iou_thres=config["car_detector"]["iou_thres"],
            tracker_config_path=config["car_detector"]["tracker_config_path"],
        )

        self.use_enhanced_detector = config["armor_detector"]["use_enhanced_detector"]
        if self.use_enhanced_detector:
            armor_detector_type = config["armor_detector"]["armor_detector_type"]
            if armor_detector_type == "three_step":
                from model.armor_detector import (
                    ArmorDetectorClassifier,
                    ArmorClassifier,
                )
                from model.armor_light_classifier import ArmorPlateParams

                original_armor_detector = YOLOv5Detector(
                    weights_path=config["armor_detector"]["weights_path"],
                    device=self.device,
                    img_size=config["armor_detector"]["img_size"],
                    augment=False,
                    visualize=False,
                    classes_name=config["armor_detector"]["class_names"],
                    conf_thres=config["armor_detector"]["conf_thres"],
                    iou_thres=config["armor_detector"]["iou_thres"],
                    max_det=config["armor_detector"]["max_det"],
                )

                armor_classifier = ArmorClassifier(
                    armor_params=ArmorPlateParams(),
                    digit_model_type=config["armor_detector"]["digit_model_type"],
                    digit_weights_path=config["armor_detector"]["digit_weights_path"],
                    debug=False,
                )

                self.armor_detector = ArmorDetectorClassifier(
                    armor_detector_model=original_armor_detector,
                    armor_classifier_model=armor_classifier,
                )
            else:
                self.armor_detector = TwoStepArmorDetectorClassifier.from_config(config)
        else:
            self.armor_detector = YOLOv5Detector(
                weights_path=config["car_detector"]["weight_path"],
                device=self.device,
                img_size=config["car_detector"]["img_size"],
                augment=False,
                visualize=False,
                classes_name=config["armor_detector"]["class"],
                conf_thres=config["car_detector"]["conf_thres"],
                iou_thres=config["car_detector"]["iou_thres"],
                max_det=config["car_detector"]["max_det"],
            )

        self.pixel_world_transform = pixel_world_transform
        self.visualize = visualize

        ## Initialize tracking state
        self.tracks = [
            TrackingState(class_id=class_id, name=class_name, max_frames=30)
            for class_id, class_name in enumerate(self.class_names[:10])
        ]
    
    def xyxy2pos3d(self, xyxy):
        x1, y1, x2, y2 = xyxy
        xo, yo = (x1 + x2) * 0.5, y2
        if (
            position_3d := self.pixel_world_transform.pixel_to_world([xo, yo])
        ) is None:
            position_3d = np.array([0.0, 0.0, 0.0])
        return position_3d.tolist()

    def detect(
        self, img
    ) -> tuple[List[SingleDetectionResult], Optional[np.ndarray | None]]:
        """
        Preform full detection pipeline on the input image.
        Args:
            img: Input image in BGR format.
        Returns:
            A tuple containing a list of detection results and an optional image with visualizations.
        """

        detections = []
        img_copy = img
        car_detections, _ = self.car_detector.predict(img_copy)

        num_car_detections = len(car_detections)
        if num_car_detections == 0:
            return [], img_copy

        crop_imgs = []
        for car_detection in car_detections:
            _, xyxy, conf, _ = car_detection
            x1, y1, x2, y2 = map(int, xyxy)
            crop_img = img_copy[y1:y2, x1:x2]
            crop_imgs.append(crop_img)

        armor_detections, _ = self.armor_detector.predict_batch(crop_imgs)

        for armor_detections, car_detection in zip(armor_detections, car_detections):
            _, car_box, car_conf, track_id = car_detection
            car_box = list(map(int, car_box))
            
            position_3d = self.xyxy2pos3d(car_box)

            if len(armor_detections) == 0:
                detection_result = SingleDetectionResult(
                    class_id=-1,
                    class_conf=0.0,
                    bot_id=track_id,
                    car_box=car_box,
                    armor_box=[0.0, 0.0, 0.0, 0.0],
                    car_conf=car_conf,
                    pos_3d=position_3d,
                    time_stamp=time.time(),
                )
                detections.append(detection_result)
                continue

            max_armor_confidence_detection = max(
                armor_detections, key=lambda x: x[2], default=None
            )

            class_id, armor_box, armor_conf = max_armor_confidence_detection
            detection_result = SingleDetectionResult(
                class_id=class_id,
                class_conf=armor_conf,
                bot_id=track_id,
                car_box=car_box,
                armor_box=list(map(int, armor_box)),
                car_conf=car_conf,
                pos_3d=position_3d,  # Placeholder for 3D position
                time_stamp=time.time(),
            )
            detections.append(detection_result)
        vis_img = None
        if self.visualize:
            vis_img = self._get_visualized_img(img_copy, detections)
        return detections, vis_img

    def _get_visualized_img(self, img, detections):
        vis_img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_LINEAR)
    
        # 计算缩放比例，用于调整边界框坐标
        original_height, original_width = img.shape[:2]
        scale_x = 1024 / original_width
        scale_y = 768 / original_height
        
        # 后续代码保持不变，但需要调整边界框坐标
        height, width = vis_img.shape[:2]  # 现在是768, 1024

        for detection in detections:
            ## Draw car bounding box
            x1_car, y1_car, x2_car, y2_car = map(int, detection.car_box)
            x1_car = int(x1_car * scale_x)
            y1_car = int(y1_car * scale_y)
            x2_car = int(x2_car * scale_x)
            y2_car = int(y2_car * scale_y)
            label = f"Car: {detection.car_conf:.2f} ID: {detection.bot_id}"
            cv2.rectangle(vis_img, (x1_car, y1_car), (x2_car, y2_car), (0, 255, 0), 2)
            cv2.putText(
                vis_img,
                label,
                (x1_car, y1_car - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            ## Draw armor detection if any
            class_id = detection.class_id
            if detection.class_id >= 0:
                x1_armor, y1_armor, x2_armor, y2_armor = map(int, detection.armor_box)
                x1_armor = int((x1_armor + detection.car_box[0]) * scale_x)
                y1_armor = int((y1_armor + detection.car_box[1]) * scale_y)
                x2_armor = int((x2_armor + detection.car_box[0]) * scale_x)
                y2_armor = int((y2_armor + detection.car_box[1]) * scale_y)

                label = f"{self.class_names[detection.class_id]}: {detection.class_conf:.2f}"
                if class_id < 5:  # blue
                    color = (0, 0, 255)
                elif class_id < 10:  # red
                    color = (255, 0, 0)
                else:  # grey
                    color = (255, 255, 255)

                cv2.rectangle(
                    vis_img, (x1_armor, y1_armor), (x2_armor, y2_armor), color, 2
                )
                cv2.putText(
                    vis_img,
                    label,
                    (x1_armor, y1_armor - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        return vis_img


if __name__ == "__main__":
    import yaml
    from transform.ray_renderer import PixelToWorld
    import numpy as np

    config = yaml.safe_load(open("config/params.yaml", "r"))
    pixel_world_transform = PixelToWorld.build_from_config(config)
    tracker = BaseDetector(config, pixel_world_transform, visualize=True)

    img = cv2.imread("test_assets/ustgz_0.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result, vis_img = tracker.detect(img)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    vis_img = cv2.resize(vis_img, (1536, 1024))
    cv2.imshow("Detection Result", vis_img)
    cv2.waitKey(0)
