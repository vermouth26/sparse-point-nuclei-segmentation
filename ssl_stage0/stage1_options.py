from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .common import ensure_detection_imports


ensure_detection_imports()

from options import Options  # noqa: E402


def build_stage1_opt(config: Dict[str, Any]) -> Options:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})
    post_cfg = config.get("postprocess", {})

    opt = Options(isTrain=True)
    opt.dataset = str(dataset_cfg["name"])
    opt.ratio = float(dataset_cfg["ratio"])
    opt.description = str(dataset_cfg["description"])
    opt.raw_data_dir = str(dataset_cfg["raw_data_dir"])
    opt.processed_data_dir = str(dataset_cfg["processed_data_dir"])
    opt.save_dir = str(dataset_cfg["save_dir"])
    opt.gpus = list(train_cfg.get("gpus", [0]))

    opt.model["name"] = str(model_cfg.get("name", "ResUNet34"))
    opt.model["pretrained"] = bool(model_cfg.get("pretrained", True))
    opt.model["out_c"] = int(model_cfg.get("out_c", 1))

    opt.train["random_seed"] = int(train_cfg.get("seed", 0))
    opt.train["input_size"] = int(train_cfg.get("input_size", 224))
    opt.train["train_epochs"] = int(train_cfg.get("epochs", 100))
    opt.train["batch_size"] = int(train_cfg.get("batch_size", 64))
    opt.train["checkpoint_freq"] = int(train_cfg.get("checkpoint_freq", 999999))
    opt.train["lr"] = float(train_cfg.get("lr", 1e-4))
    opt.train["weight_decay"] = float(train_cfg.get("weight_decay", 1e-4))
    opt.train["log_interval"] = int(train_cfg.get("log_interval", 30))
    opt.train["workers"] = int(train_cfg.get("workers", 8))
    opt.train["thresh"] = float(train_cfg.get("threshold", 0.65))
    opt.train["update_freq"] = int(train_cfg.get("update_freq", 30))
    opt.train["max_growth_rate"] = float(train_cfg.get("max_growth_rate", 0.5))
    opt.train["bg_threshold"] = float(train_cfg.get("bg_threshold", 0.1))
    opt.train["k"] = int(train_cfg.get("k_neighbors", 3))
    opt.train["curriculum_use_gt_points"] = bool(train_cfg.get("curriculum_use_gt_points", True))
    opt.train["best_precision_guard"] = float(train_cfg.get("best_precision_guard", -1.0))
    opt.train["best_precision_guard_name"] = str(train_cfg.get("best_precision_guard_name", "checkpoint_best_precision_guard_f1.pth.tar"))

    opt.test["epoch"] = "best"
    opt.test["threshold"] = float(eval_cfg.get("threshold", train_cfg.get("threshold", 0.65)))
    opt.test["patch_size"] = int(eval_cfg.get("patch_size", 224))
    opt.test["overlap"] = int(eval_cfg.get("overlap", 80))

    opt.post["max_area"] = int(post_cfg.get("max_area", 150))
    opt.post["min_area"] = int(post_cfg.get("min_area", 12))

    opt.r1 = int(eval_cfg.get("matching_radius", 11))
    opt.r2 = int(train_cfg.get("r2", 22))
    opt.radius = int(eval_cfg.get("pseudo_center_radius", 15))
    opt.gaussian_sigma = float(train_cfg.get("gaussian_sigma", opt.r1 / 4))

    opt.train["data_dir"] = f"{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/"
    opt.train["train_image_dir"] = f"{opt.processed_data_dir}/{opt.dataset}/images/train/"
    opt.train["val_image_dir"] = f"{opt.processed_data_dir}/{opt.dataset}/images/val/"
    opt.train["label_detect_dir"] = f"{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_detect/"
    opt.train["label_bg_dir"] = f"{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_bg/"
    opt.train["save_dir"] = f"{opt.save_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/"
    opt.test["image_dir"] = f"{opt.processed_data_dir}/{opt.dataset}/images/test/"
    opt.test["label_dir"] = f"{opt.raw_data_dir}/{opt.dataset}/labels_point/"
    opt.test["model_path"] = f"{opt.train['save_dir']}/checkpoints/checkpoint_best.pth.tar"
    opt.test["save_dir"] = f"{opt.train['save_dir']}/{opt.test['epoch']}"

    mean_std_path = Path(opt.processed_data_dir) / opt.dataset / "images" / "mean_std.npy"
    if mean_std_path.exists():
        opt.define_transforms()
    else:
        opt.transform["train"] = {}
        opt.transform["val"] = {}
        opt.transform["test"] = {}
    return opt


def apply_stage1_runtime_environment(opt: Options) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt.gpus)
