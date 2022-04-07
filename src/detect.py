# -*- coding: utf-8 -*-

import logging
import time

import torch
from numpy import random

from src.models.models import Darknet, load_darknet_weights
from src.utils.utils import load_class_names
from src.utils.datasets import LoadImages
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
        self.dont_show   = opt.dont_show
        self.save_img    = opt.save_img
        self.save_txt    = opt.save_txt
        self.augment     = opt.augment


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


    def detect(self, input=""):
        dataset = LoadImages(input, img_size=self.img_size, auto_size=64)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        t0 = time.time()
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            predictions = self.model(img, augment=self.augment)[0]

            # Apply NMS
            predictions = non_max_suppression(predictions, self.conf_thresh, self.iou_thresh)
            t2 = time_synchronized()
            from icecream import ic
            for i, detect in enumerate(predictions):
                p, s, im0 = path, "", im0s
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if detect is not None and len(detect):
                    # Rescale boxes from img_size to im0 size
                    detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], im0.shape).round()

                    # Print results
                    for c in detect[:, -1].unique():
                        n = (detect[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in detect:
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                print("%sDone. (%3.fs)" % (s, t2 - t1))
