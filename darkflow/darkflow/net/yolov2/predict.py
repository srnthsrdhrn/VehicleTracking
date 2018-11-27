import numpy as np
import math
import cv2
import os
import json
# from scipy.special import expit
# from utils.box import BoundBox, box_iou, prob_compare
# from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
# from turtle import *

# from .sort import black
from ..help import for_fps


def expit(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()

    return out


def findboxes(self, net_out):
    # file = self.FLAGS.demo
    # a = camera(self)
    # print(a)
    # meta
    meta = self.meta
    boxes = list()
    boxes = box_constructor(meta, net_out)
    # print(boxes)
    return boxes


def extract_boxes(self, new_im):
    cont = []
    new_im = new_im.astype(np.uint8)
    ret, thresh = cv2.threshold(new_im, 127, 255, 0)
    p, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 30 ** 2 and (
                (w < new_im.shape[0] and h <= new_im.shape[1]) or (w <= new_im.shape[0] and h < new_im.shape[1])):
            if self.FLAGS.tracker == "sort":
                cont.append([x, y, x + w, y + h])
            else:
                cont.append([x, y, w, h])
    return cont


def postprocess(self, net_out, im, resultQueue, video_id, counter):
    """
    Takes net output, draw net_out, save to disk
    """
    boxes = self.findboxes(net_out)
    meta = self.meta
    nms_max_overlap = 0.1
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else:
        imgcv = im
    h, w, _ = imgcv.shape
    thick = int((h + w) // 300)
    resultsForJSON = []

    if not self.FLAGS.track:
        for b in boxes:
            boxResults = self.process_box(b, h, w, threshold)
            if boxResults is None:
                continue
            left, right, top, bot, mess, max_indx, confidence = boxResults

            if self.FLAGS.json:
                resultsForJSON.append(
                    {"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top},
                     "bottomright": {"x": right, "y": bot}})
                continue
            if self.FLAGS.display or self.FLAGS.saveVideo:
                cv2.rectangle(imgcv,
                              (left, top), (right, bot),
                              colors[max_indx], thick)
                cv2.putText(imgcv, mess, (left, top - 12),
                            0, 1e-3 * h, colors[max_indx], thick // 3)
    else:
        detections = []
        boxes_final = []
        for b in boxes:
            boxResults = self.process_box(b, h, w, threshold)
            if boxResults is None:
                continue
            left, right, top, bot, mess, max_indx, confidence = boxResults
            boxes_final.append(boxResults)

            if mess not in self.FLAGS.trackObj:
                continue
            elif self.FLAGS.tracker == "sort":
                detections.append(np.array([left, top, right, bot]).astype(np.float64))

        if len(detections) < 3 and self.FLAGS.BK_MOG:
            detections = detections + extract_boxes(self, None)
        detections = np.array(detections)
        resultQueue.put([counter, detections, boxes_final, imgcv, video_id])
    # return imgcv  # ,avg_flow_car, avg_flow_bus, avg_flow_motorbike
