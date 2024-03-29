# -*- coding: UTF-8 -*-

import os
import sys
import logging

import cv2
import torch

from src import train
from src.train import Trainer
from src.detect import Detector
from src.test import Tester
from src.utils.option import Options
from src.utils.datasets import LoadImages
from src.utils.general import xyxy2xywh


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    opt = Options().parse()
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    if opt.action == "train":
        trainner = Trainer(opt)
        trainner.make_dirs()
        trainner.save_setting(opt)
        trainner.create_dataset(opt)
        trainner.setup()
        try:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {os.path.join("backup", opt.project, opt.exp)}", view at http://localhost:6006/\n')
            trainner.train(opt)
        except KeyboardInterrupt:
            torch.save(train.ckpt, os.path.join(os.getcwd(), "INTERRUPTED.pth"))
            sys.exit("Saved INTERRUPTED.pth")

    elif opt.action == "detect":
        detector = Detector(opt)
        detector.setup()
        input_data = os.path.abspath(opt.input)
        datasets = LoadImages(input_data, img_size=opt.img_size, auto_size=64)

        for path, img, im0s, vid_cap in datasets:
            predictions, img = detector.detect(img, opt.conf_thresh, opt.iou_thresh)
            for detect in predictions:
                visualized = detector.visualize(detect, img, im0s)
                if not opt.dont_show:
                    cv2.imshow(path, visualized)
                    cv2.waitKey(0)
                if opt.save_img:
                    out_path = os.path.join(os.getcwd(), "out", "image")
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    save_path = os.path.join(out_path, os.path.basename(path))
                    cv2.imwrite(save_path, im0s)
                if opt.save_json:
                    out_path = os.path.join(os.getcwd(), "out", "json")
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    save_path = os.path.join(out_path, os.path.basename(path))

    elif opt.action == "test":
        tester = Tester(opt)
        tester.setup()
        tester.test(opt.conf_thresh, opt.iou_thresh)