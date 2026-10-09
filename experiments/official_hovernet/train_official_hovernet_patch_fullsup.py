from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common_official_hovernet import add_hovernet_root, ensure_dir, mode_shapes, write_csv
from train_official_hovernet_fullsup import (
    hovernet_loss,
    load_pretrained,
    save_checkpoint,
    set_seed,
    vars_for_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train official HoVerNet from 540x540 patch files.")
    parser.add_argument("--hovernet-root", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["original", "fast"], default="original")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lambda-hv-mse", type=float, default=1.0)
    parser.add_argument("--lambda-hv-msge", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--imagenet-pretrained", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    worker_seed = int(torch.randint(0, 2**32, (1,))[0].cpu().item()) + worker_id
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)


def main() -> None:
    args = parse_args()
    add_hovernet_root(args.hovernet_root)
    patch_numpy_for_imgaug()
    from dataloader.train_loader import FileLoader  # noqa: WPS433
    from models.hovernet.net_desc import create_model  # noqa: WPS433
    from models.hovernet.targets import gen_targets  # noqa: WPS433

    set_seed(args.seed)
    patch_files = sorted(args.patch_dir.glob("*.npy"))
    if not patch_files:
        raise SystemExit(f"No official HoVerNet patch files found in {args.patch_dir}")
    input_shape, mask_shape = mode_shapes(args.mode)
    dataset = FileLoader(
        [str(path) for path in patch_files],
        with_type=False,
        input_shape=[input_shape, input_shape],
        mask_shape=[mask_shape, mask_shape],
        mode="train",
        setup_augmentor=args.workers == 0,
        target_gen=(gen_targets, {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn if args.workers > 0 else None,
        pin_memory=args.device == "cuda" and torch.cuda.is_available(),
    )

    out_dir = ensure_dir(args.out_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
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
    rows: list[dict] = []
    config = vars_for_json(args)
    config.update(
        {
            "patch_count": len(patch_files),
            "patch_dir": str(args.patch_dir.resolve()),
            "protocol": "Official HoVerNet FileLoader-style training from compatible 540x540_164x164 patches.",
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for iteration, batch in enumerate(loader, start=1):
            image = batch["img"].permute(0, 3, 1, 2).contiguous().float().to(device, non_blocking=True)
            true_np = batch["np_map"].long().to(device, non_blocking=True)
            true_hv = batch["hv_map"].float().to(device, non_blocking=True)
            pred = model(image)
            np_logits = pred["np"].permute(0, 2, 3, 1).contiguous()
            hv_pred = pred["hv"].permute(0, 2, 3, 1).contiguous()
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
                **{key: float(val.detach().cpu()) for key, val in loss_dict.items()},
                "np_positive_ratio": float((true_np > 0).float().mean().detach().cpu()),
                "hv_min": float(true_hv.min().detach().cpu()),
                "hv_max": float(true_hv.max().detach().cpu()),
                "hv_has_nan": bool(torch.isnan(true_hv).any().detach().cpu()),
            }
            rows.append(rec)
            losses.append(rec["total_loss"])
            print(f"[patch_train] epoch={epoch:03d} iter={iteration:03d}/{len(loader):03d} loss={rec['total_loss']:.6f}")
        scheduler.step()
        save_checkpoint(ckpt_dir / "hovernet_latest.pt", model, args, epoch, global_step)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(ckpt_dir / f"hovernet_epoch_{epoch:03d}.pt", model, args, epoch, global_step)
        write_csv(out_dir / "train_log.csv", rows)
        print(f"[patch_epoch] {epoch:03d}/{args.epochs} mean_loss={float(np.mean(losses)):.6f}")
    print(f"Saved official FileLoader-style HoVerNet training outputs to: {out_dir.resolve()}")


def patch_numpy_for_imgaug() -> None:
    # imgaug 0.4.0 and the upstream HoVerNet code use pre-2.0 NumPy aliases.
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]
    if not hasattr(np, "sctypes"):
        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_],
        }


if __name__ == "__main__":
    main()
