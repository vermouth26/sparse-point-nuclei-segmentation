from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch

from .common import ensure_dir


def save_stage0_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model,
    optimizer,
    config: Dict[str, Any],
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    is_best: bool,
    scheduler=None,
) -> Dict[str, Path]:
    ensure_dir(checkpoint_dir)
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "encoder_state": model.export_encoder_state(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "config": config,
    }
    latest_path = checkpoint_dir / "checkpoint_latest.pth"
    torch.save(payload, latest_path)

    encoder_latest = checkpoint_dir / "encoder_latest.pth"
    torch.save({"encoder_state": payload["encoder_state"], "epoch": int(epoch), "config": config}, encoder_latest)

    result = {"latest": latest_path, "encoder_latest": encoder_latest}
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pth"
        encoder_best = checkpoint_dir / "encoder_best.pth"
        torch.save(payload, best_path)
        torch.save({"encoder_state": payload["encoder_state"], "epoch": int(epoch), "config": config}, encoder_best)
        result["best"] = best_path
        result["encoder_best"] = encoder_best
    return result


def load_checkpoint_payload(path: Union[str, Path]) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    return payload
