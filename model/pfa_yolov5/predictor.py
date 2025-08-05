from models.common import DetectMultiBackend
import torch
from model.pfa_yolov5.utils.general import (
    scale_boxes,
    xyxy2xywh,
    check_img_size,
    non_max_suppression,
)
from model.pfa_yolov5.utils.augmentations import letterbox
import numpy as np
from model.pfa_yolov5.utils.plots import Annotator
import random
from typing import List
import cv2


class YOLOv5Detector:

    def __init__(
        self,
        weights_path,
        img_size=(640, 640),
        conf_thres=0.15,
        iou_thres=0.30,
        max_det=10,
        device="cuda",
        classes_name: List[str] =  ['B1','B2','B3','B4','B5','B7',
        'R1','R2','R3','R4','R5','R7'],
        classes=None,
        agnostic_nms=False,
        augment=False,
        half=True,
        visualize=True,
    ):
        self.device = torch.device(device)
        self.model = DetectMultiBackend(
            weights=weights_path, device=self.device, dnn=False, fp16=True, fuse=True
        )

        stride, names, pt, jit, onnx, engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        self.names = classes_name
        self.img_size = check_img_size(img_size, s=stride)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.half = half and (pt or jit or onnx or engine) and self.device.type != "cpu"

        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.augment = augment
        self.visualize = visualize
        self.agnostic_nms = agnostic_nms

    def preprocess(self, img):
        im = letterbox(img, self.img_size, self.model.stride, auto=self.model.pt)[0]
        # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW
        im = im.transpose((2, 0, 1))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def preprocess_batch(self, img: List[np.ndarray]):
        im = [
            letterbox(image, self.img_size, self.model.stride, auto=self.model.pt)[0]
            for image in img
        ]
        # im = [image.transpose((2, 0, 1))[::-1] for image in im]
        im = [image.transpose((2, 0, 1)) for image in im]
        im = [np.ascontiguousarray(image) for image in im]
        im = [torch.from_numpy(image).to(self.device) for image in im]
        im = [image.half() if self.half else image.float() for image in im]
        im = [image / 255 for image in im]
        ims = (
            torch.stack(im) if len(im) > 1 else im[0].unsqueeze(0)
        )  # stack if batch size > 1
        return ims

    def inference(self, img):
        if isinstance(img, list):  # List of images
            im = self.preprocess_batch(img)
        else:
            im = self.preprocess(img)
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        return pred

    def postprocess(self, pred, im, im0):
        pred = non_max_suppression(
            prediction=pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
        )

        detections = []
        annot_img = None

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [
                        xywh[0] - xywh[2] // 2,
                        xywh[1] - xywh[3] // 2,
                        xywh[2],
                        xywh[3],
                    ]
                    if self.visualize:
                        annotator = Annotator(
                            np.ascontiguousarray(im0),
                            line_width=3,
                            example=str(self.names),
                        )
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])

                    line = (int(cls), list(map(lambda x: float(x), xyxy)), float(conf))
                    detections.append(line)

                if self.visualize:
                    annot_img = annotator.result()

        return detections, annot_img

    def predict(self, img):
        # im0 = img.copy()
        im0 = img
        im = self.preprocess(img)
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        detections = self.postprocess(pred, im, im0)
        return detections

    def predict_batch(self, img: List[np.ndarray]) -> List[List[tuple]]:
        """
        Predict detections for a batch of images.

        Args:
            img: List of input images (NumPy arrays).

        Returns:
            List of detections for each image, where each detection is a tuple
            (class_name, [x, y, w, h], score).
        """
        im0 = [image.copy() for image in img]
        im = self.preprocess_batch(img)
        pred = self.model(im, augment=self.augment, visualize=self.visualize)

        results = []
        annot_imgs = []
        for i in range(len(img)):
            detections, annot_img = self.postprocess(
                pred[i].unsqueeze(0), im[i].unsqueeze(0), im0[i]
            )
            results.append(detections)
            if self.visualize:
                annot_imgs.append(annot_img)

        return results, annot_imgs


if __name__ == "__main__":
    predictor = YOLOv5Detector(
        weights_path="weights/car_pfa.pt",
        visualize=True,
        img_size=(640, 640),
    )