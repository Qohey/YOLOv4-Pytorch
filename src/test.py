# -*- coding: utf-8 -*-

import os
import glob
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.utils import load_class_names
from src.utils.torch_utils import select_device, time_synchronized
from src.models.models import Darknet, load_darknet_weights
from src.utils.general import box_iou, increment_path, check_img_size, coco80_to_coco91_class, xywh2xyxy
from src.utils.general import non_max_suppression, scale_coords, xyxy2xywh, clip_coords
from src.utils.datasets import create_dataloader
from src.utils.loss import compute_loss
from src.utils.plots import output_to_target, plot_images
from src.utils.metrics import ap_per_class


logger = logging.getLogger(__name__)

class Tester:
    def __init__(self, model=None, data_loader=None, opt=None):
        self.is_train   = model is not None # 学習か直実行か
        self.cfg        = opt.cfg
        self.names      = opt.names
        self.num_class  = len(self.names)
        self.img_size   = opt.img_size
        self.batch_size = opt.batch_size
        self.test_path  = opt.test_path
        self.weights    = None
        self.device     = None
        self.augment    = False
        self.verbose    = False
        self.save_txt   = False
        self.save_conf  = False
        self.save_json  = False
        self.plots      = True

        if self.is_train:
            self.model          = model
            self.data_loader    = data_loader
            self.save_dir       = opt.save_dir
            self.plots          = opt.plots
        else:
            self.weights    = opt.weights
            self.device     = opt.device
            self.save_dir   = os.path.abspath(os.path.join("test", opt.project, opt.exp))
            self.augment    = opt.augment
            self.verbose    = opt.verbose
            self.save_txt   = opt.save_txt
            self.save_conf  = opt.save_conf
            self.save_json  = opt.save_json


    def setup(self):
        if self.is_train:
            self.device = next(self.model.parameters()).device
        else:
            self.device = select_device(self.device, batch_size=self.batch_size)
            self.save_dir = Path(increment_path(self.save_dir))
            self.label_path = Path(self.save_dir)
            if self.save_txt:
                self.label_path = Path(self.save_dir / "labels")
            Path(self.label_path).mkdir(parents=True, exist_ok=True)
            self.model = Darknet(self.cfg).to(self.device)
            try:
                checkpoint = torch.load(self.weights[0], map_location=self.device)
                checkpoint["model"] = {k: v for k, v in checkpoint["model"].items() if self.model.state_dict()[k].numel() == v.numel()}
                self.model.load_state_dict(checkpoint["model"], strict=False)
            except:
                load_darknet_weights(self.model, self.weights[0])
            self.img_size = check_img_size(self.img_size, s=64)

        self.half = (self.device.type != "cpu")
        if self.half:
            self.model.half()

        if not self.is_train:
            img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
            _ = self.model(img.half() if self.half else img) if self.device.type != "cpu" else None
            self.data_loader = create_dataloader(self.test_path, self.img_size, self.batch_size, 64, pad=0.5, rect=True)[0]
        try:
            self.names = self.model.names if hasattr(self.model, "names") else self.model.module.names
        except:
            self.names = load_class_names(self.names)
        finally:
            self.num_class = len(self.names)


    def test(self, conf_thresh=0.001, iou_thresh=0.6):
        self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        loss = torch.zeros(3, device=self.device)

        p, r, f1, mp, mr, map50, map, t0, t1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        jdict, stats, ap, ap_class = [], [], [], []
        seen = 0
        info = (("%12s" * 8) % ("Class", "Images", "Targets", "P", "R", "mAP@.5", "mAP@.5:.95", ""))
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(self.data_loader, desc=info, unit="img")):
            img = img.to(self.device, non_blocking=True)
            img = img.half() if self.half else img.float()
            img /= 255.0
            targets = targets.to(self.device)
            batch_size, _, height, width = img.shape
            whwh = torch.Tensor([width, height, width, height]).to(self.device)

            # Disable gradients
            with torch.no_grad():
                t = time_synchronized()
                inf_out, train_out = self.model(img, augment=self.augment)
                t0 += time_synchronized() - t

                if self.is_train:
                    loss += compute_loss([x.float() for x in train_out], targets, self.model)[1][:3]

                t = time_synchronized()
                output = non_max_suppression(inf_out, conf_thres=conf_thresh, iou_thres=iou_thresh)
                t1 += time_synchronized() - t

            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                num_labels = len(labels)
                target_class = labels[:, 0].tolist() if num_labels else []
                seen += 1

                if len(pred) == 0:
                    if num_labels:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), target_class))
                    continue

                # Append to text file
                path = Path(paths[si])
                if self.save_txt:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]] # normalization gain whwh
                    x = pred.clone()
                    x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                    for *xyxy, conf, cls, in x:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(self.label_path / (path.stem + ".txt"), mode="a", encoding="UTF-8") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Append to pycocotools JSON dictionary
                if self.save_json:
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = pred[:, :4].clone()
                    scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])
                    box = xyxy2xywh(box)
                    box[:, :2] -= box[:, 2:] / 2
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({"image_id": image_id,
                                      "category_id": int(p[5]),
                                      "bbox": [round(x, 3) for x in b],
                                      "score": round(p[4], 5)})
                                      
                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=self.device)
                if num_labels:
                    detected = []
                    target_class_tensor = labels[:, 0]
                    target_boxes = xywh2xyxy(labels[:, 1:5]) * whwh

                    for cls in torch.unique(target_class_tensor):
                        target_indeces = (cls == target_class_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        predict_indeces = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)           # target indices

                        if predict_indeces.shape[0]:
                            ious, i = box_iou(pred[predict_indeces, :4], target_boxes[target_indeces]).max(1)
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = target_indeces[i[j]]
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[predict_indeces[j]] = ious[j] > iouv
                                    if len(detected) == num_labels:
                                        break

                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

            if self.plots and batch_i < 3:
                f = os.path.join(self.save_dir, f"test_batch{batch_i}_labels.jpg")
                plot_images(images=img, targets=targets, paths=paths, fname=f, names=self.names)
                f = os.path.join(self.save_dir, f"test_batch{batch_i}_pred.jpg")
                plot_images(images=img, targets=output_to_target(output, width, height), paths=paths, fname=f, names=self.names)

        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plots, fname=os.path.join(self.save_dir, "precision-recall_curve.png"))
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.num_class)
        else:
            nt = torch.zeros(1)

        print_format = "%12s" + "%12.3g" * 6 + "%12s"
        print(print_format % ("all", seen, nt.sum(), mp, mr, map50, map, ""))

        if self.verbose and self.num_class > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(print_format % (self.names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (self.img_size, self.img_size, self.batch_size)
        if not self.is_train:
            print("Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g" % t)

        if self.save_json and len(jdict):
            weights = Path(self.weights[0] if isinstance(self.weights, list) else self.weights).stem if self.weights is not None else ""
            annotation_json = glob.glob(os.path.join(self.save_dir, "annotations", "instances_val.json"))[0]
            pred_json = os.path.join(self.save_dir, f"{weights}_predinctions.json")
            print("\nEvaluting pycocotools mAP... saving %s..." % pred_json)
            with open(pred_json, mode="w", encoding="UTF-8") as f:
                json.dump(jdict, f)

            try:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                annotation = COCO(annotation_json)
                pred = annotation.loadRes(pred_json)
                eval = COCOeval(annotation, pred, "bbox")
                eval.params.imgIds = [int(Path(x).stem) for x in self.data_loader.dataset.img_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]
            except Exception as e:
                print("ERROR: pycocotools unable to run: %s" % e)

        if not self.is_train:
            print("Results saved to %s" % self.save_dir)
        self.model.float()
        maps = np.zeros(self.num_class) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        return (mp, mr, map50, map, *(loss.cpu() / len(self.data_loader)).tolist()), maps, t
