from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import ensure_detection_imports, save_json
from .stage1_init import load_ssl_encoder_into_stage1


ensure_detection_imports()

from dataset import DataFolder  # noqa: E402
from model import create_model  # noqa: E402
from my_transforms import get_transforms  # noqa: E402
from prepare_data import main as prepare_data_main  # noqa: E402
from test import main as detection_test  # noqa: E402
from train import save_checkpoint, train, validate  # noqa: E402
import utils  # noqa: E402


def create_stage1_model(opt, init_cfg: Dict[str, Any]) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    mode = str(init_cfg.get("mode", "imagenet"))
    model = create_model(opt.model["name"], opt.model["out_c"], opt.model["pretrained"])
    report = None
    if mode == "ssl_stage0":
        report = load_ssl_encoder_into_stage1(
            model,
            init_cfg["ssl_encoder_path"],
            strict=bool(init_cfg.get("strict", False)),
            load_conv1=bool(init_cfg.get("load_conv1", True)),
            load_bn1=bool(init_cfg.get("load_bn1", True)),
            load_layers=init_cfg.get("load_layers", ["layer1", "layer2", "layer3", "layer4"]),
            conv1_strategy=str(init_cfg.get("conv1_strategy", "repeat_divide")),
        )
    elif mode != "imagenet":
        raise ValueError(f"Unsupported initialization mode: {mode}")
    return model, report


def run_stage1_training(opt, init_cfg: Dict[str, Any]) -> Dict[str, Any]:
    utils.create_folder(opt.train["save_dir"])
    opt.isTrain = True
    mean_std_path = Path(opt.processed_data_dir) / opt.dataset / "images" / "mean_std.npy"
    if mean_std_path.exists():
        opt.define_transforms()
    opt.save_options()

    model, init_report = create_stage1_model(opt, init_cfg)
    model = nn.DataParallel(model).cuda()

    if init_report is not None:
        report_path = Path(opt.train["save_dir"]) / "ssl_init_report.json"
        save_json(report_path, init_report)

    optimizer = torch.optim.Adam(
        model.parameters(),
        opt.train["lr"],
        betas=(0.9, 0.99),
        weight_decay=opt.train["weight_decay"],
    )
    criterion = torch.nn.MSELoss(reduction="none").cuda()

    dir_list = [opt.train["train_image_dir"], opt.train["label_detect_dir"]]
    post_fix = ["label_detect.png"]
    num_channels = [3, 1]
    train_transform = get_transforms(opt.transform["train"])
    train_set = DataFolder(dir_list, post_fix, num_channels, train_transform, r=opt.r2)
    train_loader = DataLoader(
        train_set,
        batch_size=opt.train["batch_size"],
        shuffle=True,
        num_workers=opt.train["workers"],
    )
    val_transform = get_transforms(opt.transform["val"])

    bg_flag = False
    best_score = -1.0
    best_guarded_score = -1.0
    best_guarded_checkpoint = None
    update_time = 0
    for epoch in range(opt.train["train_epochs"]):
        print(f"Epoch: [{epoch + 1}/{opt.train['train_epochs']}]")
        train_loss = train(opt, train_loader, model, optimizer, criterion, with_bg=bg_flag)
        with torch.no_grad():
            val_recall, val_prec, val_f1 = validate(opt, model, val_transform)

        is_best = val_f1 > best_score
        best_score = max(val_f1, best_score)
        cp_flag = (epoch + 1) % opt.train["checkpoint_freq"] == 0
        save_checkpoint(
            {"epoch": epoch + 1, "state_dict": model.state_dict()},
            epoch,
            is_best,
            opt.train["save_dir"],
            cp_flag,
        )
        precision_guard = float(opt.train.get("best_precision_guard", -1.0))
        guarded_name = str(opt.train.get("best_precision_guard_name", "checkpoint_best_precision_guard_f1.pth.tar"))
        is_guarded_best = precision_guard >= 0 and val_prec > precision_guard and val_f1 > best_guarded_score
        if is_guarded_best:
            best_guarded_score = val_f1
            checkpoint_dir = Path(opt.train["save_dir"]) / "checkpoints"
            guarded_path = checkpoint_dir / guarded_name
            shutil.copyfile(checkpoint_dir / "checkpoint.pth.tar", guarded_path)
            best_guarded_checkpoint = str(guarded_path)
            print(
                f"=> Guarded best updated: precision>{precision_guard:.4f}, "
                f"F1 {val_f1:.4f}, checkpoint {guarded_path}"
            )

        if (epoch + 1) % opt.train["update_freq"] == 0:
            update_time += 1
            opt.test["image_dir"] = f"{opt.raw_data_dir}/{opt.dataset}/images"
            detection_test(opt)
            from prepare_data import main as update_data  # noqa: E402

            update_data(opt, time=update_time)
            if (epoch + 1) == opt.train["update_freq"]:
                bg_flag = True
                dir_list = [
                    opt.train["train_image_dir"],
                    opt.train["label_detect_dir"],
                    opt.train["label_bg_dir"],
                ]
                post_fix = ["label_detect.png", "label_bg.png"]
                num_channels = [3, 1, 1]
                train_set = DataFolder(dir_list, post_fix, num_channels, train_transform, r=opt.r2)
                train_loader = DataLoader(
                    train_set,
                    batch_size=opt.train["batch_size"],
                    shuffle=True,
                    num_workers=opt.train["workers"],
                )

    return {
        "save_dir": opt.train["save_dir"],
        "best_checkpoint": str(Path(opt.train["save_dir"]) / "checkpoints" / "checkpoint_best.pth.tar"),
        "best_guarded_checkpoint": best_guarded_checkpoint,
        "best_precision_guard": float(opt.train.get("best_precision_guard", -1.0)),
        "ssl_init_report_path": str(Path(opt.train["save_dir"]) / "ssl_init_report.json") if init_report is not None else None,
    }


def run_stage1_pipeline(config: Dict[str, Any], opt) -> Dict[str, Any]:
    pipeline_cfg = config.get("pipeline", {})
    if bool(pipeline_cfg.get("prepare_data", True)):
        prepare_data_main(opt)

    summary = run_stage1_training(opt, config.get("initialization", {}))

    if bool(pipeline_cfg.get("run_test_after_train", True)):
        opt.test["image_dir"] = f"{opt.processed_data_dir}/{opt.dataset}/images/test/"
        detection_test(opt, eval_flag=True)
    if bool(pipeline_cfg.get("run_inference_after_train", True)):
        opt.test["image_dir"] = f"{opt.raw_data_dir}/{opt.dataset}/images/"
        detection_test(opt, eval_flag=False)
    if bool(pipeline_cfg.get("final_prepare_voronoi", True)):
        prepare_data_main(opt, time=-1)
    return summary
