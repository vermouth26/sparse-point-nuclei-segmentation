from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2hed
from torch.utils.data import Dataset


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES])


def rgb_to_hematoxylin(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    hed = rgb2hed(np.clip(rgb, 0.0, 1.0))
    hematoxylin = -hed[..., 0]
    hematoxylin -= float(hematoxylin.min())
    denom = float(hematoxylin.max())
    if denom > 1e-8:
        hematoxylin /= denom
    return hematoxylin.astype(np.float32)


class SSLTrainTransform:
    def __init__(self, crop_size: int = 224, hflip: bool = True, vflip: bool = True):
        self.crop_size = int(crop_size)
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        crop = self.crop_size
        if height >= crop and width >= crop:
            top = 0 if height == crop else random.randint(0, height - crop)
            left = 0 if width == crop else random.randint(0, width - crop)
            rgb = rgb[top : top + crop, left : left + crop]
        if self.hflip and random.random() < 0.5:
            rgb = np.flip(rgb, axis=1)
        if self.vflip and random.random() < 0.5:
            rgb = np.flip(rgb, axis=0)
        return np.ascontiguousarray(rgb)


class SSLValTransform:
    def __init__(self, crop_size: int = 224):
        self.crop_size = int(crop_size)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        crop = self.crop_size
        if height >= crop and width >= crop:
            top = max((height - crop) // 2, 0)
            left = max((width - crop) // 2, 0)
            rgb = rgb[top : top + crop, left : left + crop]
        return np.ascontiguousarray(rgb)


class HematoxylinReconstructionDataset(Dataset):
    def __init__(self, image_dir: Union[str, Path], transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.image_dir = Path(image_dir)
        self.image_paths = list_images(self.image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.image_paths[index]
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if self.transform is not None:
            rgb = self.transform(rgb)
        target_rgb = torch.from_numpy((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1))
        hematoxylin = rgb_to_hematoxylin(rgb)
        source_h = torch.from_numpy(hematoxylin[None, ...])
        return source_h.float(), target_rgb.float(), path.stem
