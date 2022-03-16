# -*- coding: utf-8 -*-

import sys
import argparse
from pprint import pprint


class Options:
    """
    プログラム引数の設定、パースを行うクラス
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        argv = sys.argv
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument("action", type=str, help="Choose type of execution", choices=["train", "detect", "test"])

        # ===============================================================
        #                     Training options
        # ===============================================================
        if "train" in argv:
            self.parser.add_argument("--cfg",           type=str,   default="config/yolov4.cfg",    help="Path to CFG file")
            self.parser.add_argument("--names",         type=str,   default="config/food.names",    help="Path to names file")
            self.parser.add_argument("--param",         type=str,   default="config/param.yaml",    help="Path to param file")
            self.parser.add_argument("--input",         type=str,   default="data",                 help="Select input data")
            self.parser.add_argument("--label",         type=str,   default=["all"],                help="Select training labels",   nargs="+")
            self.parser.add_argument("--optimizer",     type=str,   default="sgd",                  help="Choose optimize function", choices=["adam", "sgd"])
            self.parser.add_argument("--device",        type=str,   default="0",                    help="GPU Index num or cpu")
            self.parser.add_argument("--project",       type=str,   default="train",                help="Save to backup/[--project]")
            self.parser.add_argument("--exp",           type=str,   default="exp",                  help="Save to backup/[--project]/[--name]")

            self.parser.add_argument("--seed",          type=int,   default=-1,         help="Initial seed")
            self.parser.add_argument("--batch_size",    type=int,   default=16,         help="Total batch size for all GPUs")
            self.parser.add_argument("--workers",       type=int,   default=8,          help="Maximum number of dataloader workers")
            self.parser.add_argument("--epochs",        type=int,   default=100,        help="Train epochs")
            self.parser.add_argument("--img_size",      type=int,   default=[448, 448], help="[train, test] image sizes", nargs="+",)

            self.parser.add_argument("--cache",         action="store_true",    help="Cache images for faster training")
            self.parser.add_argument("--rect",          action="store_true",    help="Rectangular training")
            self.parser.add_argument("--multi_scale",   action="store_true",    help="Vary img-size +/- 50%%")

        # ===============================================================
        #                     Detecting options
        # ===============================================================
        if "detect" in argv:
            self.parser.add_argument("--cfg",       type=str,   default="config/Yolov4.cfg",    help="Path to CFG file")
            self.parser.add_argument("--names",     type=str,   default="config/Food.names",    help="Path to names file")
            self.parser.add_argument("--input",     type=str,   default="data",                 help="Select training labels")
            self.parser.add_argument("--weights",   type=str,   default="",                     help="Select *.pth file")
            self.parser.add_argument("--device",    type=str,   default="0",                    help="GPU Index num or cpu")

            self.parser.add_argument("--thresh",    type=float, default=0.5,    help="Detection threshold specified between 0.0~1.0")

            self.parser.add_argument("--dont_show",     action="store_true",    help="Dont show results")
            self.parser.add_argument("--save_img",      action="store_true",    help="Save results to [--input/*.jpg]")
            self.parser.add_argument("--save_labels",   action="store_true",    help="Save results to *.txt")

        # ===============================================================
        #                     Testing options
        # ===============================================================
        if "test" in argv:
            self.parser.add_argument("task",            type=str,   default="valid",                help="Choose type of test", choices=["test", "valid", "study"])
            self.parser.add_argument("--cfg",           type=str,   default="config/yolov4.cfg",    help="Path to CFG file")
            self.parser.add_argument("--names",         type=str,   default="config/food.names",    help="Path to names file")
            self.parser.add_argument("--weights",       type=str,   default="",                     help="Select *.pth files",  nargs="+", required=True)
            self.parser.add_argument("--device",        type=str,   default="0",                    help="GPU Index num or cpu")
            self.parser.add_argument("--project",       type=str,   default="result",               help="Save to backup/[--project]")
            self.parser.add_argument("--exp",           type=str,   default="exp",                  help="Save to backup/[--project]/[--name]")
            self.parser.add_argument("--test_path",     type=str,   default="test.txt",             help="Path to test.txt")

            self.parser.add_argument("--img_size",      type=int,   default=640,    help="Inference size (pixels)")
            self.parser.add_argument('--batch_size',    type=int,   default=32,     help="Size of each image batch")
            self.parser.add_argument("--conf_thresh",   type=float, default=0.001,  help="Object confidence threshold")
            self.parser.add_argument("--iou_thresh",    type=float, default=0.65,   help="IOU threshold for NMS")

            self.parser.add_argument("--augment",   action="store_true",    help="Augmented inference")
            self.parser.add_argument("--verbose",   action="store_true",    help="Report mAP by class")
            self.parser.add_argument("--save_txt",  action="store_true",    help="Save results to *.txt")
            self.parser.add_argument("--save_conf", action="store_true",    help="Save confidences in [--save_txt] labels")
            self.parser.add_argument("--save_json", action="store_true",    help="Save a cocoapi-compatible JSON results file")


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt