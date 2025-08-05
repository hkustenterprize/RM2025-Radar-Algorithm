import time
import torch
import torchvision
from general import LOGGER, xywh2xyxy, box_iou


def non_max_suppression_merged(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
    tf=[[0, 0, 1.0, 1.0]],
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    # bs = prediction.shape[0]  # batch size
    bs = 1
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = False  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = True  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, (6 + nm) * bs), device=prediction.device)]
    x = torch.tensor(data=[], device=device)
    for xi, x_origin in enumerate(prediction):  # image index, image inference
        x_origin = x_origin[xc[xi]]  # confidence
        if x_origin.shape[0]:
            multiplier = torch.ones(x_origin.shape[-1], device=device)
            multiplier[0:4] = torch.tensor(
                [tf[xi][2], tf[xi][3], tf[xi][2], tf[xi][3]], device=device
            )
            adder = torch.zeros(x_origin.shape[-1], device=device)
            adder[0:2] = torch.tensor([tf[xi][0], tf[xi][1]], device=device)
            x = torch.cat((x, x_origin * multiplier + adder), 0)

    # If none remain process next image
    if not x.shape[0]:
        return None

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box/Mask
    box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
    mask = x[:, mi:]  # zero columns if no masks

    # best class only
    conf, j = x[:, 5:mi].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

    # Filter by class
    if classes is not None:
        x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return None
    elif n > max_nms:  # excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
    else:
        x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes

    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det * prediction.shape[0]:  # limit detections
        i = i[: max_det * prediction.shape[0]]
    if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
        # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        weights = iou * scores[None]  # box weights
        x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
            1, keepdim=True
        )  # merged boxes
        if redundant:
            i = i[iou.sum(1) > 1]  # require redundancy

    output[0] = x[i]
    if mps:
        output[0] = output[0].to(device)
    if (time.time() - t) > time_limit:
        LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
    #     return None

    return output
