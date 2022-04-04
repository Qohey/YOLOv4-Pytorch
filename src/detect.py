# -*- coding: utf-8 -*-

import logging

import torch
from numpy import random

from src.models.models import Darknet, load_darknet_weights
from src.utils.utils import load_class_names
from src.utils.datasets import LoadImages
from src.utils.torch_utils import select_device


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
        return
