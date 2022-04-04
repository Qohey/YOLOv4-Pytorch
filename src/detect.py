# -*- coding: utf-8 -*-

import logging

from src.models.models import Darknet
from src.utils.utils import load_class_names
from src.utils.torch_utils import select_device


logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, opt):
        self.cfg        = opt.cfg
        self.names      = load_class_names(opt.names)
        self.weight     = opt.weight
        self.device     = opt.device
        self.img_size   = opt.img_size
        self.dont_show  = opt.dont_show
        self.save_img   = opt.save_img
        self.save_txt   = opt.save_txt


    def setup(self):
        self.model  = Darknet(self.cfg, self.img_size).cuda()
        self.device = select_device(self.device)
        self.half   = (self.device.type != "cpu")


    def detect(self, input="", conf_thresh=0.001, iou_thresh=0.6):
        return
