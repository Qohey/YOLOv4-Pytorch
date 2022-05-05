# -*- coding: utf-8 -*-

import os
import logging
import time

import cv2
import torch
from numpy import random

from src.models.models import Darknet, load_darknet_weights
from src.utils.utils import load_class_names

from src.utils.torch_utils import select_device, time_synchronized
from src.utils.general import non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer
from src.utils.plots import plot_one_box


logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, opt):
        self.cfg         = opt.cfg
        self.names       = load_class_names(opt.names)
        self.weight      = opt.weight
        self.device      = opt.device
        self.img_size    = opt.img_size
        self.conf_thresh = opt.conf_thresh
        self.iou_thresh  = opt.iou_thresh
        self.save_txt    = opt.save_txt
        self.augment     = opt.augment
        self.colors      = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]


    def setup(self):
        self.model  = Darknet(self.cfg, self.img_size).cuda()
        self.device = select_device(self.device)
        self.half   = (self.device.type != "cpu")
        try:
            self.model.load_state_dict(torch.load(self.weight, map_location=self.device)["model"])
        except:
            load_darknet_weights(self.model, self.weight)
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()


    def detect(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        predictions = self.model(img, augment=self.augment)[0]
        # Apply NMS
        predictions = non_max_suppression(predictions, self.conf_thresh, self.iou_thresh)
        return predictions, img


    def visualize(self, detect, img, im0s):
        s = ""
        s += "%gx%g " % img.shape[2:]  # print string
        if detect is not None and len(detect):
            # Rescale boxes from img_size to im0 size
            detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], im0s.shape).round()
            # Print results
            for c in detect[:, -1].unique():
                n = (detect[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
            # Write results
            for *xyxy, conf, cls in detect:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)], line_thickness=3)
        return im0s