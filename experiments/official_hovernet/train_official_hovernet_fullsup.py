from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common_official_hovernet import (
    DEFAULT_SPLIT,
    DEFAULT_TRAIN_IMAGES,
    DEFAULT_TRAIN_LABELS,
    add_hovernet_root,
    augment_pair,
    ensure_dir,
    image_label_paths,
    mode_shapes,
    random_crop_pair,
    read_instance,
    read_rgb,
    read_split,
    write_csv,
)


class CPM17OfficialHoverDataset(Dataset):
    def __init__(
        self,
        image_ids: list[str],
        image_dir: Path,
        label_dir: Path,
        hovernet_root: Path,
        mode: str,
        samples_per_epoch: int,
        seed: int,
        augment: bool,
        blur_augment: bool,
        min_fg_ratio: float,
    ) -> None:
        add_hovernet_root(hovernet_root)
        from models.hovernet.targets import gen_targets  # noqa: WPS433

        self.image_ids = image_ids
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_shape, self.output_shape = mode_shapes(mode)
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.augment = augment
        self.blur_augment = blur_augment
        self.min_fg_ratio = min_fg_ratio
        self.gen_targets = gen_targets
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, image_id: str) -> tuple[np.ndarray, np.ndarray]:
        if image_id not in self.cache:
            image_path, label_path = image_label_paths(image_id, self.image_dir, self.label_dir)
            self.cache[image_id] = (read_rgb(image_path), read_instance(label_path))
        return self.cache[image_id]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + index * 104729)
        image_id = self.image_ids[index % len(self.image_ids)]
        image, inst = self._load(image_id)
        image_crop, inst_crop = random_crop_pair(
            image=image,
            inst=inst,
            crop_size=self.input_shape,
            rng=rng,
            min_fg_ratio=self.min_fg_ratio,
        )
        if self.augment:
            image_crop, inst_crop = augment_pair(image_crop, inst_crop, rng, blur=self.blur_augment)
        targets = self.gen_targets(inst_crop.astype(np.int32), [self.output_shape, self.output_shape])
        return {
            "image_id": image_id,
            "img": torch.from_numpy(image_crop.transpose(2, 0, 1).copy()).float(),
            "np_map": torch.from_numpy(targets["np_map"].astype(np.int64)),
            "hv_map": torch.from_numpy(targets["hv_map"].astype(np.float32)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train official HoVerNet on CPM17 full-supervised GT targets.")
    parser.add_argument("--hovernet-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/official_hovernet/fullsup"))
    parser.add_argument("--mode", choices=["original", "fast"], default="original")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-split-mode", choices=["train28", "train32"], default="train28")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--blur-augment", action="store_true")
    parser.add_argument("--min-fg-ratio", type=float, default=0.01)
    parser.add_argument("--lambda-hv-mse", type=float, default=1.0)
    parser.add_argument("--lambda-hv-msge", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--imagenet-pretrained", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_hovernet_root(args.hovernet_root)
    from models.hovernet.net_desc import create_model  # noqa: WPS433

    set_seed(args.seed)
    split = read_split(args.split)
    if args.train_split_mode == "train28":
        train_ids = list(split["train"])
        protocol = "CPM17 official train/test split with fixed validation split; train28 only for fullsup training."
    else:
        train_ids = sorted(list(split["train"]) + list(split["val"]))
        protocol = "CPM17 paper-style full-supervised reproduction; official train32 used for training, no val checkpoint selection."
    out_dir = ensure_dir(args.out_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    log_path = out_dir / "train_log.csv"

    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = create_model(input_ch=3, nr_types=None, freeze=bool(args.freeze_encoder), mode=args.mode).to(device)
    if args.imagenet_pretrained:
        load_pretrained(model, args.imagenet_pretrained)
    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    samples_per_epoch = args.iters_per_epoch * args.batch_size
    dataset = CPM17OfficialHoverDataset(
        image_ids=train_ids,
        image_dir=args.train_images,
        label_dir=args.train_labels,
        hovernet_root=args.hovernet_root,
        mode=args.mode,
        samples_per_epoch=samples_per_epoch,
        seed=args.seed,
        augment=args.augment,
        blur_augment=args.blur_augment,
        min_fg_ratio=args.min_fg_ratio,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    config = vars_for_json(args)
    config.update({"train_ids": train_ids, "protocol": protocol})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        for iteration, batch in enumerate(loader, start=1):
            image = batch["img"].to(device, non_blocking=True)
            true_np = batch["np_map"].to(device, non_blocking=True)
            true_hv = batch["hv_map"].to(device, non_blocking=True)
            pred = model(image)
            np_logits = pred["np"].permute(0, 2, 3, 1).contiguous()
            hv_pred = pred["hv"].permute(0, 2, 3, 1).contiguous()
            if np_logits.shape[1:3] != true_np.shape[1:3]:
                raise RuntimeError(f"NP output/target shape mismatch: {tuple(np_logits.shape)} vs {tuple(true_np.shape)}")
            loss_dict = hovernet_loss(np_logits, hv_pred, true_np, true_hv, args.lambda_hv_mse, args.lambda_hv_msge)
            loss = loss_dict["total_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            global_step += 1
            rec = {
                "epoch": epoch,
                "iter": iteration,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                **{k: float(v.detach().cpu()) for k, v in loss_dict.items()},
                "np_positive_ratio": float((true_np > 0).float().mean().detach().cpu()),
                "hv_min": float(true_hv.min().detach().cpu()),
                "hv_max": float(true_hv.max().detach().cpu()),
                "hv_has_nan": bool(torch.isnan(true_hv).any().detach().cpu()),
            }
            rows.append(rec)
            epoch_losses.append(rec["total_loss"])
            print(f"[train] epoch={epoch:03d} iter={iteration:03d} loss={rec['total_loss']:.6f}")
        scheduler.step()
        save_checkpoint(ckpt_dir / "hovernet_latest.pt", model, args, epoch, global_step)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(ckpt_dir / f"hovernet_epoch_{epoch:03d}.pt", model, args, epoch, global_step)
        write_csv(log_path, rows)
        print(f"[epoch] {epoch:03d}/{args.epochs} mean_loss={float(np.mean(epoch_losses)):.6f}")

    print(f"Saved official HoVerNet fullsup training outputs to: {out_dir.resolve()}")


def hovernet_loss(
    np_logits: torch.Tensor,
    hv_pred: torch.Tensor,
    true_np: torch.Tensor,
    true_hv: torch.Tensor,
    lambda_hv_mse: float,
    lambda_hv_msge: float,
) -> dict[str, torch.Tensor]:
    true_np_onehot = F.one_hot(true_np.long(), num_classes=2).float()
    np_prob = F.softmax(np_logits, dim=-1)
    ce = F.cross_entropy(np_logits.permute(0, 3, 1, 2).contiguous(), true_np.long())
    dice = dice_loss(true_np_onehot, np_prob)
    hv_mse = F.mse_loss(hv_pred, true_hv)
    hv_msge = msge_loss(true_hv, hv_pred, (true_np > 0).float())
    total = ce + dice + lambda_hv_mse * hv_mse + lambda_hv_msge * hv_msge
    return {
        "total_loss": total,
        "np_ce_loss": ce,
        "np_dice_loss": dice,
        "hv_mse_loss": hv_mse,
        "hv_msge_loss": hv_msge,
    }


def dice_loss(true: torch.Tensor, pred: torch.Tensor, smooth: float = 1e-3) -> torch.Tensor:
    inse = torch.sum(pred * true, dim=(0, 1, 2))
    left = torch.sum(pred, dim=(0, 1, 2))
    right = torch.sum(true, dim=(0, 1, 2))
    return torch.sum(1.0 - (2.0 * inse + smooth) / (left + right + smooth))


def msge_loss(true: torch.Tensor, pred: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
    kernel_h, kernel_v = sobel_kernels(size=5, device=pred.device)
    true_grad = hv_gradient(true, kernel_h, kernel_v)
    pred_grad = hv_gradient(pred, kernel_h, kernel_v)
    focus2 = focus[..., None].repeat(1, 1, 1, 2)
    loss = focus2 * (pred_grad - true_grad).pow(2)
    return loss.sum() / (focus2.sum() + 1.0e-8)


def sobel_kernels(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vals = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=device)
    h, v = torch.meshgrid(vals, vals, indexing="ij")
    kernel_h = h / (h * h + v * v + 1.0e-15)
    kernel_v = v / (h * h + v * v + 1.0e-15)
    return kernel_h.view(1, 1, size, size), kernel_v.view(1, 1, size, size)


def hv_gradient(hv: torch.Tensor, kernel_h: torch.Tensor, kernel_v: torch.Tensor) -> torch.Tensor:
    h_ch = hv[..., 0].unsqueeze(1)
    v_ch = hv[..., 1].unsqueeze(1)
    h_grad = F.conv2d(h_ch, kernel_h, padding=kernel_h.shape[-1] // 2)
    v_grad = F.conv2d(v_ch, kernel_v, padding=kernel_v.shape[-1] // 2)
    return torch.cat([h_grad, v_grad], dim=1).permute(0, 2, 3, 1).contiguous()


def load_pretrained(model: torch.nn.Module, path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Pretrained checkpoint not found: {path}")
    loaded = torch.load(path, map_location="cpu")
    state = loaded.get("desc", loaded.get("state_dict", loaded.get("model_state", loaded)))
    if all(k.split(".")[0] == "module" for k in state):
        state = {".".join(k.split(".")[1:]): v for k, v in state.items()}
    feedback = model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained checkpoint: {path}")
    print("Missing keys:", feedback.missing_keys)
    print("Unexpected keys:", feedback.unexpected_keys)


def save_checkpoint(path: Path, model: torch.nn.Module, args: argparse.Namespace, epoch: int, global_step: int) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": vars_for_json(args),
        },
        path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vars_for_json(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


if __name__ == "__main__":
    main()
