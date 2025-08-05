from ultralytics import YOLO
import cv2
import numpy as np


class Predictor:

    def __init__(
        self,
        model_path: str,
        img_size=1280,
        conf_thres=0.15,
        iou_thres=0.45,
        max_det=10,
        device="cuda",
        visualize=False,
    ):
        self.model_path = model_path
        self.input_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.visualize = visualize
        self.max_det = max_det

        self.model = YOLO(model_path)
        if model_path.split(".")[-1] != "engine":
            self.model = self.model.to(self.device)
            self.model.eval()

    def predict(self, image):
        detections = []
        results = self.model.predict(
            image,
            imgsz=self.input_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
        )
        for result in results:
            for box in result.boxes:
                for conf, cls, xyxyn in zip(box.conf, box.cls, box.xyxyn):
                    conf = float(conf.detach().cpu().numpy())
                    cls = int(cls.detach().cpu().numpy())
                    xyxyn = xyxyn.detach().cpu().numpy()
                    x1, y1, x2, y2 = xyxyn
                    ## Normalzied coordinates to pixel coordinates
                    x1, y1, x2, y2 = (
                        int(x1 * image.shape[1]),
                        int(y1 * image.shape[0]),
                        int(x2 * image.shape[1]),
                        int(y2 * image.shape[0]),
                    )
                    line = (int(cls), [x1, y1, x2, y2], float(conf))
                    detections.append(line)
        return detections, None

    def predict_batch(self, images):
        all_detections = []
        results = self.model.predict(
            images,
            imgsz=self.input_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
        )
        for result, image in zip(results, images):
            detections = []
            for box in result.boxes:
                for conf, cls, xyxyn in zip(box.conf, box.cls, box.xyxyn):
                    conf = float(conf.detach().cpu().numpy())
                    cls = int(cls.detach().cpu().numpy())
                    xyxyn = xyxyn.detach().cpu().numpy()
                    x1, y1, x2, y2 = xyxyn
                    ## Normalzied coordinates to pixel coordinates
                    x1, y1, x2, y2 = (
                        int(x1 * image.shape[1]),
                        int(y1 * image.shape[0]),
                        int(x2 * image.shape[1]),
                        int(y2 * image.shape[0]),
                    )
                    line = (int(cls), [x1, y1, x2, y2], float(conf))
                    detections.append(line)
            all_detections.append(detections)
        return all_detections, None

    def visualize_results(self, image, detections, class_names=None):
        """可视化检测结果"""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # 默认类别名称
        if class_names is None:
            class_names = {0: "car", 1: "person", 2: "bike"}  # 根据你的模型调整

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

    predictor = Predictor(model_path="weights/car_v1.2.pt", visualize=True)
    image_path = "test_assets/zk_5.jpg"
    image = cv2.imread(image_path)

    # 预测
    results = predictor.predict(image)
    print("检测结果:", results)

    # 可视化
    if predictor.visualize:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = predictor.predict(image_rgb)

        # 可视化结果
        vis_image = predictor.visualize_results(image, results)
        save_path = "visualize/yolov12/zk_5.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_image)
