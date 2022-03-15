# -*- coding: UTF-8 -*-

import os
import sys
import logging

import torch

from src import train
from src.train import Trainer
from src.tester import Tester
from src.utils.option import Options

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
        print("NotImplemented")

    elif opt.action == "test":
        if opt.task == "test" or opt.task == "valid":
            tester = Tester(opt=opt)
            tester.setup()
            tester.test(conf_thresh=0.001, iou_thresh=0.6)
        elif opt.task == "study":
            raise NotImplemented