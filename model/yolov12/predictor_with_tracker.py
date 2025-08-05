from ultralytics import YOLO
import cv2
import numpy as np
import time


class PredictorWithTracker:

    def __init__(
        self,
        model_path: str,
        img_size=1280,
        conf_thres=0.05,
        iou_thres=0.5,
        device="cuda",
        max_det=10,
        tracker_config_path: str = "config/bytetrack.yaml",
        visualize=False,
    ):
        self.model_path = model_path
        self.input_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.visualize = visualize
        self.max_det = max_det

        self.model = YOLO(model_path, task="detect")
        self.device = self.model.device
        print(f"Using device: {self.device}")
        # self.model = self.model.to(self.device)
        # self.model.eval()
        self.tracker_config_path = tracker_config_path

    def predict(self, image):
        detections = []
        last = time.time()
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            print("[Warning]: Invalid or empty image provided.")
            return detections, None
        
        resized_img = cv2.resize(
            image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        results = self.model.track(
            resized_img,
            stream=False,
            persist=True,
            tracker=self.tracker_config_path,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det = self.max_det
        )[0]
        now = time.time()
        print(f"[Detect car time]: {now - last:.4f} seconds")
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections, None


        for result in results:
            for box in result.boxes:
                if box is None or box.conf is None or box.cls is None or box.xyxyn is None or box.id is None:
                    print("[Warning]: Invalid detection box detected.")
                    continue
                for conf, cls, xyxyn, track_id in zip(
                    box.conf, box.cls, box.xyxyn, box.id
                ):
                    conf = float(conf.detach().cpu().numpy())
                    cls = int(cls.detach().cpu().numpy())
                    xyxyn = xyxyn.detach().cpu().numpy()
                    track_id = int(track_id.detach().cpu().numpy())
                    x1, y1, x2, y2 = xyxyn
                    ## Normalzied coordinates to pixel coordinates
                    x1, y1, x2, y2 = (
                        int(x1 * image.shape[1]),
                        int(y1 * image.shape[0]),
                        int(x2 * image.shape[1]),
                        int(y2 * image.shape[0]),
                    )
                    line = (int(cls), [x1, y1, x2, y2], float(conf), track_id)
                    detections.append(line)
        return detections, None

    def visualize_results(self, image, detections, class_names=None):
        """可视化检测结果"""
        # vis_image = image.copy()
        vis_image = cv2.resize(
            image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        h, w = vis_image.shape[:2]

        # 默认类别名称
        if class_names is None:
            class_names = {0: "car"}  # 根据你的模型调整

        # 颜色列表
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for detection in detections:
            cls_id, bbox, conf = detection
            x1, y1, x2, y2 = bbox

            # 转换归一化坐标到像素坐标
            # x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # 选择颜色
            color = colors[cls_id % len(colors)]

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            label = f"{class_names.get(cls_id, f'Class_{cls_id}')}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return vis_image


if __name__ == "__main__":
    import cv2
    import os

    predictor = PredictorWithTracker(model_path="weights/car_v1.2.pt", visualize=True)
    image_path = "test_assets/zk_5.jpg"
    image = cv2.imread(image_path)

    # 预测
    results = predictor.predict(image)
    print("Detection result:", results)

    # 可视化
    if predictor.visualize:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = predictor.predict(image_rgb)

        # 可视化结果
        vis_image = predictor.visualize_results(image, results[0])
        save_path = "visualize/yolov12/zk_5.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_image)
