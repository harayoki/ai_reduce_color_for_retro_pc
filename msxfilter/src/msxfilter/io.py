"""I/O helpers for MSX filter toolkit."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image_01(path: str | Path, width: int, height: int) -> torch.Tensor:
    """Load an RGB image, resize, and return float tensor in [0, 1] with shape (H, W, 3)."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr)


def save_image_01(path: str | Path, img01_hwc: torch.Tensor) -> None:
    """Save a float image (H, W, 3) in [0, 1] to disk."""
    arr = (img01_hwc.clamp(0, 1).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr, mode="RGB").save(path)


def load_palette_txt(path: str | Path) -> torch.Tensor:
    """Load palette text file with lines like 'R G B' or 'R,G,B', returning (N, 3) float tensor."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError(f"bad palette line: {line}")
        r, g, b = [float(x) for x in parts]
        rows.append([r, g, b])
    if not rows:
        raise ValueError("palette empty")
    return torch.tensor(rows, dtype=torch.float32) / 255.0
