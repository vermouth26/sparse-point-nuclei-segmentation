from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
DETECTION = ROOT / "detection"
for path in (ROOT, DETECTION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import DataFolder  # noqa: E402
from model import create_model  # noqa: E402
from my_transforms import get_transforms  # noqa: E402
from prepare_data import main as prepare_data_main  # noqa: E402
from train import save_checkpoint, train, validate  # noqa: E402
from ssl_stage0.common import set_random_seed  # noqa: E402
from ssl_stage0.config import load_yaml_config, resolve_config_paths  # noqa: E402
from ssl_stage0.stage1_options import apply_stage1_runtime_environment, build_stage1_opt  # noqa: E402


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_forward_batched(
    model: nn.Module,
    input_tensor: torch.Tensor,
    size: int,
    overlap: int,
    out_channels: int,
    tile_batch_size: int,
) -> torch.Tensor:
    """Match the legacy overlap stitching while batching independent tiles."""
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be positive")
    batch, channels, original_h, original_w = input_tensor.shape
    if batch != 1:
        raise ValueError("Curriculum full-image inference expects batch size 1")
    stride = size - overlap
    if stride <= 0:
        raise ValueError("patch size must be larger than overlap")

    pad_h = 0
    if original_h - size > 0 and (original_h - size) % stride > 0:
        pad_h = stride - (original_h - size) % stride
        input_tensor = torch.cat(
            (input_tensor, torch.zeros((batch, channels, pad_h, original_w), dtype=input_tensor.dtype)),
            dim=2,
        )
    pad_w = 0
    if original_w - size > 0 and (original_w - size) % stride > 0:
        pad_w = stride - (original_w - size) % stride
        input_tensor = torch.cat(
            (
                input_tensor,
                torch.zeros((batch, channels, original_h + pad_h, pad_w), dtype=input_tensor.dtype),
            ),
            dim=3,
        )

    _, _, height, width = input_tensor.shape
    output = torch.zeros((1, out_channels, height, width), dtype=torch.float32)
    tiles: list[torch.Tensor] = []
    metadata: list[tuple[int, int, int, int, int, int]] = []
    device = next(model.parameters()).device

    def flush() -> None:
        if not tiles:
            return
        tile_batch = torch.cat(tiles, dim=0).to(device, non_blocking=True)
        predictions = model(tile_batch).detach().cpu()
        for prediction, (row, col, row_start, row_end, col_start, col_end) in zip(predictions, metadata):
            output[:, :, row_start:row_end, col_start:col_end] = prediction[
                None, :, row_start - row : row_end - row, col_start - col : col_end - col
            ]
        tiles.clear()
        metadata.clear()

    for row in range(0, height - overlap, stride):
        row_end = min(row + size, height)
        row_start_keep = row + overlap // 2 if row > 0 else 0
        row_end_keep = row + size - overlap // 2 if row + size < height else height
        for col in range(0, width - overlap, stride):
            col_end = min(col + size, width)
            col_start_keep = col + overlap // 2 if col > 0 else 0
            col_end_keep = col + size - overlap // 2 if col + size < width else width
            tiles.append(input_tensor[:, :, row:row_end, col:col_end])
            metadata.append((row, col, row_start_keep, row_end_keep, col_start_keep, col_end_keep))
            if len(tiles) >= tile_batch_size:
                flush()
    flush()
    return output[:, :, :original_h, :original_w]


@torch.inference_mode()
def get_probmap_batched(input_tensor: torch.Tensor, model: nn.Module, opt, tile_batch_size: int) -> np.ndarray:
    patch_size = int(opt.test["patch_size"])
    if patch_size == 0:
        output = model(input_tensor.to(next(model.parameters()).device))
        output = output.detach().cpu()
    else:
        output = split_forward_batched(
            model,
            input_tensor,
            patch_size,
            int(opt.test["overlap"]),
            int(opt.model["out_c"]),
            tile_batch_size,
        )
    return torch.sigmoid(output[0, 0]).numpy()


@torch.inference_mode()
def infer_train24_probmaps(opt, model: nn.Module, tile_batch_size: int) -> Path:
    split = json.loads((Path(opt.raw_data_dir) / opt.dataset / "train_val_test.json").read_text(encoding="utf-8"))
    train_ids = [Path(item).stem for item in split["train"]]
    output_dir = Path(opt.test["save_dir"]) / "images_prob_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_model = model.module if isinstance(model, nn.DataParallel) else model
    was_training = inference_model.training
    inference_model.eval()
    transform = get_transforms(opt.transform["test"])
    image_dir = Path(opt.raw_data_dir) / opt.dataset / "images"
    import imageio.v2 as imageio

    total_start = time.perf_counter()
    for index, image_id in enumerate(train_ids, start=1):
        image_start = time.perf_counter()
        tensor = transform((Image.open(image_dir / f"{image_id}.png").convert("RGB"),))[0].unsqueeze(0)
        prob = np.asarray(get_probmap_batched(tensor, inference_model, opt, tile_batch_size), dtype=np.float32)
        imageio.imwrite(output_dir / f"{image_id}_prob.png", (prob * 255).astype(np.uint8))
        print(f"[curriculum inference {index:02d}/24] {image_id} {time.perf_counter() - image_start:.2f}s")
    if was_training:
        inference_model.train()
    print(
        f"Saved batched curriculum probability maps: {output_dir} "
        f"(tile_batch_size={tile_batch_size}, total={time.perf_counter() - total_start:.2f}s)"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pure ImageNet-initialized leakage-safe BoNuS-derived R0.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--curriculum-tile-batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = resolve_config_paths(load_yaml_config(args.config))
    if config["initialization"]["mode"] != "imagenet" or not bool(config["model"]["pretrained"]):
        raise RuntimeError("This runner only permits ImageNet R0 initialization.")
    if bool(config["training"].get("curriculum_use_gt_points", True)):
        raise RuntimeError("Hidden full-center directory access must be disabled.")
    opt = build_stage1_opt(config)
    apply_stage1_runtime_environment(opt)
    opt.train["curriculum_use_gt_points"] = False
    if args.workers is not None:
        opt.train["workers"] = int(args.workers)
    if args.dry_run:
        print(json.dumps({
            "pass": True,
            "initialization": "imagenet",
            "epochs": int(opt.train["train_epochs"]),
            "batch_size": int(opt.train["batch_size"]),
            "workers": int(opt.train["workers"]),
            "curriculum_tile_batch_size": int(args.curriculum_tile_batch_size),
            "curriculum_use_gt_points": False,
            "hibou_imported": False,
            "alphamix_checkpoint_loaded": False,
            "test_run": False,
        }, indent=2))
        return

    set_random_seed(int(opt.train["random_seed"]))
    Path(opt.train["save_dir"]).mkdir(parents=True, exist_ok=True)
    opt.isTrain = True
    opt.define_transforms()
    opt.save_options()
    model = nn.DataParallel(create_model(opt.model["name"], opt.model["out_c"], opt.model["pretrained"])).cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.train["lr"], betas=(0.9, 0.99), weight_decay=opt.train["weight_decay"])
    criterion = torch.nn.MSELoss(reduction="none").cuda()
    train_transform = get_transforms(opt.transform["train"])
    val_transform = get_transforms(opt.transform["val"])

    def make_loader(with_bg: bool) -> DataLoader:
        dirs = [opt.train["train_image_dir"], opt.train["label_detect_dir"]]
        postfix = ["label_detect.png"]
        channels = [3, 1]
        if with_bg:
            dirs.append(opt.train["label_bg_dir"])
            postfix.append("label_bg.png")
            channels.append(1)
        dataset = DataFolder(dirs, postfix, channels, train_transform, r=opt.r2)
        return DataLoader(dataset, batch_size=opt.train["batch_size"], shuffle=True, num_workers=opt.train["workers"])

    loader = make_loader(False)
    best_score = -1.0
    update_index = 0
    history = []
    for epoch in range(int(opt.train["train_epochs"])):
        print(f"Epoch: [{epoch + 1}/{opt.train['train_epochs']}]")
        loss = train(opt, loader, model, optimizer, criterion, with_bg=update_index > 0)
        with torch.no_grad():
            recall, precision, f1 = validate(opt, model, val_transform)
        is_best = f1 > best_score
        best_score = max(best_score, f1)
        save_checkpoint(
            {"epoch": epoch + 1, "state_dict": model.state_dict()},
            epoch,
            is_best,
            opt.train["save_dir"],
            (epoch + 1) % int(opt.train["checkpoint_freq"]) == 0,
        )
        history.append({"epoch": epoch + 1, "loss": float(loss), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
        if (epoch + 1) % int(opt.train["update_freq"]) == 0:
            update_index += 1
            infer_train24_probmaps(opt, model, args.curriculum_tile_batch_size)
            prepare_data_main(opt, time=update_index)
            if update_index == 1:
                loader = make_loader(True)

    summary = {
        "variant": "R0",
        "initialization": "imagenet",
        "bn_recalibration": False,
        "operating_point_sweep": False,
        "curriculum_updates": update_index,
        "curriculum_inference": "batched tiled overlap stitching",
        "curriculum_tile_batch_size": int(args.curriculum_tile_batch_size),
        "history": history,
        "train_hidden_gt_read": False,
        "test_run": False,
        "downstream_run": False,
    }
    run_dir = Path(opt.train["save_dir"])
    save_json(run_dir / "r0_training_summary.json", summary)
    save_json(run_dir / "stage1_ssl_summary.json", {"config": config, "summary": {"variant": "R0", "complete": True}})
    print("[DONE] Pure R0 Stage-I training complete; no BN, test, or downstream was run.")


if __name__ == "__main__":
    main()
