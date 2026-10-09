from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch

from .checkpointing import load_checkpoint_payload
from .common import strip_known_prefixes


ENCODER_PARTS = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _extract_encoder_source(payload: Dict) -> Dict[str, torch.Tensor]:
    if "encoder_state" in payload and isinstance(payload["encoder_state"], dict):
        source = payload["encoder_state"]
    elif "model_state" in payload and isinstance(payload["model_state"], dict):
        source = payload["model_state"]
    elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
        source = payload["state_dict"]
    else:
        source = payload

    if not isinstance(source, dict):
        raise ValueError("Unsupported encoder checkpoint format.")

    extracted: Dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if not torch.is_tensor(value):
            continue
        normalized = strip_known_prefixes(str(key))
        if normalized.startswith(ENCODER_PARTS):
            extracted[normalized] = value
    if not extracted:
        raise ValueError("No encoder weights found in checkpoint.")
    return extracted


def _should_load_part(part_name: str, load_bn1: bool, load_layers: Iterable[str]) -> bool:
    if part_name == "bn1":
        return load_bn1
    if part_name.startswith("layer"):
        return part_name in set(load_layers)
    return True


def _adapt_conv1(source: torch.Tensor, target: torch.Tensor, strategy: str) -> Optional[torch.Tensor]:
    if source.shape == target.shape:
        return source
    if strategy == "skip":
        return None
    if source.ndim != 4 or target.ndim != 4:
        return None
    src_in = source.shape[1]
    tgt_in = target.shape[1]
    if src_in == 1 and tgt_in == 3:
        if strategy == "repeat_divide":
            return source.repeat(1, 3, 1, 1) / 3.0
        if strategy == "repeat":
            return source.repeat(1, 3, 1, 1)
    if src_in == 3 and tgt_in == 1:
        return source.mean(dim=1, keepdim=True)
    return None


def load_ssl_encoder_into_stage1(
    model,
    checkpoint_path: Union[str, Path],
    *,
    strict: bool = False,
    load_conv1: bool = True,
    load_bn1: bool = True,
    load_layers: Iterable[str] = ("layer1", "layer2", "layer3", "layer4"),
    conv1_strategy: str = "repeat_divide",
) -> Dict[str, object]:
    payload = load_checkpoint_payload(checkpoint_path)
    source_state = _extract_encoder_source(payload)
    target_model = _unwrap_model(model)
    target_state = target_model.state_dict()

    mapped_state: Dict[str, torch.Tensor] = {}
    skipped = []
    missing_candidates = []

    for source_key, tensor in source_state.items():
        part_name = source_key.split(".", 1)[0]
        if part_name == "conv1" and not load_conv1:
            skipped.append(source_key)
            continue
        if not _should_load_part(part_name, load_bn1, load_layers):
            skipped.append(source_key)
            continue

        target_key = f"resnet.{source_key}"
        if target_key not in target_state:
            missing_candidates.append(target_key)
            continue

        if part_name == "conv1":
            adapted = _adapt_conv1(tensor, target_state[target_key], conv1_strategy)
            if adapted is None:
                skipped.append(source_key)
                continue
            mapped_state[target_key] = adapted
            continue

        if tensor.shape != target_state[target_key].shape:
            skipped.append(source_key)
            continue
        mapped_state[target_key] = tensor

    load_result = target_model.load_state_dict(mapped_state, strict=False)
    report = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "loaded_keys": sorted(mapped_state.keys()),
        "loaded_key_count": len(mapped_state),
        "missing_after_load": list(load_result.missing_keys),
        "unexpected_after_load": list(load_result.unexpected_keys),
        "skipped_source_keys": sorted(skipped),
        "missing_target_candidates": sorted(missing_candidates),
        "conv1_strategy": conv1_strategy,
        "strict": bool(strict),
    }

    if strict and (missing_candidates or load_result.unexpected_keys):
        raise RuntimeError(f"Strict SSL init failed: {report}")
    return report


def materialize_stage1_init_checkpoint(
    model,
    report: Dict[str, object],
    output_path: Union[str, Path],
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    payload = {
        "epoch": 0,
        "state_dict": _unwrap_model(model).state_dict(),
        "ssl_init_report": report,
    }
    if extra:
        payload.update(extra)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output
