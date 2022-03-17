# -*- coding: UTF-8 -*-

import os
import glob
import random
import math
import time
import logging
from copy import deepcopy

import numpy as np
import yaml
import torch
from tqdm import tqdm
from torch import optim
from torch.cuda import amp
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils.utils import load_class_names, replace_extension
from src.utils.torch_utils import ModelEMA, select_device
from src.utils.datasets import create_dataloader
from src.utils.general import check_img_size, labels_to_class_weights, strip_optimizer, increment_path
from src.utils.general import fitness, fitness_p, fitness_r, fitness_ap50, fitness_ap, fitness_f
from src.utils.plots import plot_images, plot_labels, plot_results
from src.utils.loss import compute_loss
from src.models.models import Darknet
from src.test import Tester


logger = logging.getLogger(__name__)

class Trainer:
    global ckpt # Checkpoint
    def __init__(self, opt):
        self.cfg            = opt.cfg
        self.names          = load_class_names(opt.names)
        self.seed           = opt.seed
        self.batch_size     = opt.batch_size
        self.labels         = "all" if "all" in opt.label else [label for label in opt.label]
        self.save_dir       = os.path.abspath(os.path.join("backup", opt.project, opt.exp))
        self.optimizer_type = opt.optimizer
        self.device         = opt.device

        with open(opt.param, mode="r", encoding="UTF-8") as f:
            self.param = yaml.safe_load(f)


    def make_dirs(self):
        self.save_dir = increment_path(self.save_dir) # Update save_dir path
        os.makedirs(os.path.join(self.save_dir, "weights"))
        self.weight_dir = os.path.join(self.save_dir, "weights")


    def save_setting(self, opt):
        with open(os.path.join(self.save_dir, "param.yaml"), mode="w", encoding="UTF-8") as f:
            yaml.dump(self.param, f, sort_keys=False)
        with open(os.path.join(self.save_dir, "opt.yaml"), mode="w", encoding="UTF-8") as f:
            yaml.dump(vars(opt), f, sort_keys=False)


    def create_dataset(self, opt):
        truth = []
        train_dataset = []
        valid_dataset = []

        if self.labels == "all":
            img_paths = glob.iglob(os.path.join(opt.input, "**", "*.jpg"), recursive=True)
        else:
            img_paths = []
            for label in self.labels:
                img_paths += glob.iglob(os.path.join(opt.input, label.lower(),"*.jpg"), recursive=False)

        for img_path in img_paths:
            img_path = os.path.abspath(img_path)
            txt_path = replace_extension(img_path, ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, mode="r", encoding="UTF-8") as file:
                    format_check = True
                    for line in file:
                        if len(line.split(" ")[:]) != 5:
                            format_check = False
                    if format_check:
                        truth.append(img_path)

        TRUTH_LEN = len(truth)
        while len(train_dataset) != int(TRUTH_LEN*0.7):
            random_index = int(len(truth)*random.random())
            if not truth[random_index] in train_dataset:
                train_dataset.append(truth[random_index])
                truth.remove(truth[random_index])
        train_dataset.sort()
        valid_dataset = truth.copy()

        self.train_path = os.path.join(self.save_dir, "train.txt")
        self.valid_path = os.path.join(self.save_dir, "valid.txt")
        self.result_path = os.path.join(self.save_dir, "results.txt")

        with open(self.train_path, mode="w", encoding="UTF-8") as f:
            f.writelines("\n".join(train_dataset))

        with open(self.valid_path, mode="w", encoding="UTF-8") as f:
            f.writelines("\n".join(valid_dataset))


    def fix_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if seed == 0:  # slower, more reproducible
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:  # faster, less reproducible
            cudnn.deterministic = False
            cudnn.benchmark = True


    def setup(self):
        self.device = select_device(self.device, batch_size=self.batch_size)
        self.use_cuda = (self.device.type != "cpu")
        self.fix_seed(2 + self.seed)
        self.model = Darknet(self.cfg).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.names = self.names
        self.model.nc = len(self.names)
        self.model.hyp = self.param
        self.model.gr = 1.0

        self.nominam_batch_size = 64
        self.accumulate = max(round(self.nominam_batch_size / self.batch_size), 1)  # accumulate loss before optimizing
        self.param["weight_decay"] *= self.batch_size * self.accumulate / self.nominam_batch_size  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for key, value in dict(self.model.named_parameters()).items():
            if ".bias" in key:
                pg2.append(value)  # biases
            elif "Conv2d.weight" in key:
                pg1.append(value)  # apply weight_decay
            elif "m.weight" in key:
                pg1.append(value)  # apply weight_decay
            elif "w.weight" in key:
                pg1.append(value)  # apply weight_decay
            else:
                pg0.append(value)  # all else

        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(pg0, lr=self.param["lr0"], betas=(self.param["momentum"], 0.999))  # adjust beta1 to momentum
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(pg0, lr=self.param["lr0"], momentum=self.param["momentum"], nesterov=True)

        self.optimizer.add_param_group({"params": pg1, "weight_decay": self.param["weight_decay"]})  # add pg1 with weight_decay
        self.optimizer.add_param_group({"params": pg2})  # add pg2 (biases)

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.EMA = ModelEMA(self.model)

        logger.info("Optimizer groups: %g .bias, %g conv.weight, %g other" % (len(pg2), len(pg1), len(pg0)))


    def lf(self, x):
        return ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - self.param["lrf"]) + self.param["lrf"]


    def train(self, opt):
        grid_size = 64
        img_size, img_size_test = [check_img_size(x, grid_size) for x in opt.img_size]  # verify imgsz are gs-multiples
        train_loader, train_dataset = create_dataloader(self.train_path, img_size, self.batch_size, grid_size,
                                                        hyp=self.param, augment=True, cache=opt.cache, rect=opt.rect,
                                                        rank=-1, world_size=1, workers=opt.workers)
        max_label_class = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
        num_batches = len(train_loader)
        assert max_label_class < len(self.names), "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (max_label_class, len(self.names), opt.data, len(self.names) - 1)

        start_epoch, best_fitness = 0, 0.0
        self.EMA.updates = start_epoch * num_batches // self.accumulate
        test_loader  = create_dataloader(self.valid_path, img_size_test, self.batch_size*2, grid_size,
                                        hyp=self.param, cache=opt.cache, rect=True,
                                        rank=-1, world_size=1, workers=opt.workers)[0]  # testloader

        labels = np.concatenate(train_dataset.labels, 0)
        classes = torch.tensor(labels[:, 0])
        plot_labels(labels, save_dir=self.save_dir)
        tensorboard_writer = SummaryWriter(self.save_dir)
        tensorboard_writer.add_histogram("classes", classes, 0)

        self.param["cls"] *= len(self.names) / 80.
        self.model.class_weights = labels_to_class_weights(train_dataset.labels, len(self.names)).to(self.device)
        num_warmup = max(round(self.param["warmup_epochs"] * num_batches), 1000)
        results = (0, 0, 0, 0, 0, 0, 0)
        self.scheduler.last_epoch = start_epoch - 1
        scaler = amp.GradScaler(enabled=self.use_cuda)
        logger.info("\nImage sizes %g train, %g test Using %g dataloader workers\n"
                    "Logging results to %s\n"
                    "Starting training for %g epochs..." % (img_size, img_size_test, train_loader.num_workers, self.save_dir, opt.epochs))

        torch.save(self.model, os.path.join(self.weight_dir, "init.pth"))

        t0 = time.time()
        best_fitness_p, best_fitness_r, best_fitness_ap50, best_fitness_ap, best_fitness_f = 0.0, 0.0, 0.0, 0.0, 0.0
        for epoch in range(start_epoch, opt.epochs):
            self.model.train()
            mloss = torch.zeros(4, device=self.device)
            pbar = enumerate(train_loader)
            logger.info(("\n" + "%12s" * 8) % ("Epoch", "GPU_Mem", "Box", "Obj", "Cls", "Total", "Targets", "Img_size"))
            pbar = tqdm(pbar, total=num_batches, unit="img")
            self.optimizer.zero_grad()

            for i, (imgs, targets, paths, _) in pbar:
                num_integrated = i + num_batches * epoch
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0

                if num_integrated <= num_warmup:
                    x_interp = [0, num_warmup]
                    self.accumulate = max(1, np.interp(num_integrated, x_interp, [1, self.nominam_batch_size / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(num_integrated, x_interp, [self.param["warmup_bias_lr"] if j==2 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(num_integrated, x_interp, [self.param["warmup_momentum"], self.param["momentum"]])

                if opt.multi_scale:
                    sz = random.randrange(img_size * 0.5, img_size * 1.5 + grid_size) // grid_size * grid_size  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / grid_size) * grid_size for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

                # Forward
                with amp.autocast(enabled=self.use_cuda):
                    pred = self.model(imgs)
                    loss, loss_times = compute_loss(pred, targets.to(self.device), self.model)

                # Backward
                scaler.scale(loss).backward()

                if num_integrated % self.accumulate == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.EMA.update(self.model)

                if num_integrated < 3:
                    file_name = os.path.join(self.save_dir, f"train_batch{num_integrated}.jpg")
                    plot_images(images=imgs, targets=targets, paths=paths, fname=file_name, names=self.names)

                # Print
                mloss = (mloss * i + loss_times) / (i + 1)
                memory = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                info = ("%12s" * 2 + "%12.4g" * 6) % ("%g/%g" % (epoch + 1, opt.epochs), memory, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(info)
            # End batch

            lr = [x["lr"] for x in self.optimizer.param_groups]
            self.scheduler.step()
            self.EMA.update_attr(self.model)

            if 3 <= epoch:
                test_model = self.EMA.ema.module if hasattr(self.EMA.ema, "module") else self.EMA.ema
                test_opt = deepcopy(opt)
                members = [member for member in test_opt.__dict__]
                # Initialize opt member
                for member in members:
                    delattr(test_opt, member)
                test_opt.cfg        = opt.cfg
                test_opt.names      = opt.names
                test_opt.test_path  = self.valid_path
                test_opt.img_size   = img_size_test
                test_opt.batch_size = self.batch_size*2
                test_opt.save_dir   = self.save_dir
                test_opt.plots      = True

                tester = Tester(model=test_model, data_loader=test_loader, opt=test_opt)
                tester.setup()
                results, maps, times = tester.test(conf_thresh=0.001, iou_thresh=0.6)
                del tester, test_opt

            with open(self.result_path, mode="a", encoding="UTF-8") as f:
                f.write(info + "%10.4g" * 7 % results + "\n")

            tags = ["train/box_loss",       "train/obj_loss",   "train/cls_loss",
                    "metrics/precision",    "metrics/recall",   "metrics/mAP_0.5", "metrics/mAP_0.5:0.95",
                    "val/box_loss",         "val/obj_loss",     "val/cls_loss",
                    "x/lr0",                "x/lr1",            "x/lr2"]

            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                tensorboard_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))              # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_p = fitness_p(np.array(results).reshape(1, -1))          # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_r = fitness_r(np.array(results).reshape(1, -1))          # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap50 = fitness_ap50(np.array(results).reshape(1, -1))    # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap = fitness_ap(np.array(results).reshape(1, -1))        # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if (fi_p > 0.0) or (fi_r > 0.0):
                fi_f = fitness_f(np.array(results).reshape(1, -1))      # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            else:
                fi_f = 0.0
            if fi > best_fitness:
                best_fitness = fi
            if fi_p > best_fitness_p:
                best_fitness_p = fi_p
            if fi_r > best_fitness_r:
                best_fitness_r = fi_r
            if fi_ap50 > best_fitness_ap50:
                best_fitness_ap50 = fi_ap50
            if fi_ap > best_fitness_ap:
                best_fitness_ap = fi_ap
            if fi_f > best_fitness_f:
                best_fitness_f = fi_f
            # End epochs
        # End training

        # Save model
        with open(self.result_path, mode="r", encoding="UTF-8") as f:
            ckpt = {"epoch":                epoch,
                    "best_fitness":         best_fitness,
                    "best_fitness_p":       best_fitness_p,
                    "best_fitness_r":       best_fitness_r,
                    "best_fitness_ap50":    best_fitness_ap50,
                    "best_fitness_ap":      best_fitness_ap,
                    "best_fitness_f":       best_fitness_f,
                    "training_results":     f.read(),
                    "model":                self.EMA.ema.module.state_dict() if hasattr(self.EMA, "module") else self.EMA.ema.state_dict(),
                    "optimizer":            self.optimizer.state_dict()}

        torch.save(ckpt, os.path.join(self.weight_dir, "last.pth"))
        if best_fitness == fi:
            torch.save(ckpt, os.path.join(self.weight_dir, "best.pth"))
            torch.save(ckpt, os.path.join(self.weight_dir, "best_overall.pth"))
            if 200 <= epoch:
                torch.save(ckpt, os.path.join(self.weight_dir, f"best_{epoch:03d}.pth"))
        if best_fitness_p == fi_p:
            torch.save(ckpt, os.path.join(self.weight_dir, "best_p.pth"))
        if best_fitness_r == fi_r:
            torch.save(ckpt, os.path.join(self.weight_dir, "best_r.pth"))
        if best_fitness_ap50 == fi_ap50:
            torch.save(ckpt, os.path.join(self.weight_dir, "best_ap50.pth"))
        if best_fitness_ap == fi_ap:
            torch.save(ckpt, os.path.join(self.weight_dir, "best_ap.pth"))
        if best_fitness_f == fi_f:
            torch.save(ckpt, os.path.join(self.weight_dir, "best_f.pth"))

        fresults = os.path.join(self.save_dir, "results.txt")
        flast = os.path.join(self.weight_dir, "last.pth")
        fbest = os.path.join(self.weight_dir, "best.pth")
        for f1, f2 in zip([os.path.join(self.weight_dir, "last.pth"), os.path.join(self.weight_dir, "best.pth"), self.result_path], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)
                if f2.endswith(".pth"):
                    strip_optimizer(f2)

        plot_results(save_dir=self.save_dir)
        logger.info("%g epochs completed in %.3f hours.\n" % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        torch.cuda.empty_cache()
        return results