"""Overlay the official surface prediction on a CT slice of Scroll 3 (PHerc0332).

The prediction volume is stored at 1/4 scale of the CT volume, so CT pyramid
level N+2 matches prediction level N. Streams only the requested slice.

Usage:
    python scripts/overlay_surface.py [pred_level] [z_frac]
      pred_level: prediction pyramid level (default 2 -> CT level 4, ~986px)
      z_frac:     relative z position 0..1 (default 0.5)
"""

import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332"
CT = f"{BUCKET}/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr"
PRED = (
    f"{BUCKET}/representations/predictions/surfaces/"
    "20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr"
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    pred_level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    z_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    ct_level = pred_level + 2

    ct = zarr.open_array(f"{CT}/{ct_level}", mode="r")
    pred = zarr.open_array(f"{PRED}/{pred_level}", mode="r")
    print(f"CT L{ct_level}: {ct.shape}  PRED L{pred_level}: {pred.shape}")

    z = int(ct.shape[0] * z_frac)
    ct_sl = np.asarray(ct[z, :, :]).astype(np.float32)
    pz = min(int(pred.shape[0] * z_frac), pred.shape[0] - 1)
    pred_sl = np.asarray(pred[pz, :, :])

    # prediction grid can differ by a couple of px from the CT grid; crop to common area
    h = min(ct_sl.shape[0], pred_sl.shape[0])
    w = min(ct_sl.shape[1], pred_sl.shape[1])
    ct_sl, pred_sl = ct_sl[:h, :w], pred_sl[:h, :w]

    # grayscale CT as background, prediction in red on top
    bg = np.clip(ct_sl * (255.0 / max(ct_sl.max(), 1)), 0, 255).astype(np.uint8)
    rgb = np.stack([bg, bg, bg], axis=-1)
    mask = pred_sl > 0
    rgb[mask, 0] = 255
    rgb[mask, 1] = (rgb[mask, 1] * 0.25).astype(np.uint8)
    rgb[mask, 2] = (rgb[mask, 2] * 0.25).astype(np.uint8)

    frac = 100.0 * mask.sum() / mask.size
    print(f"z={z} (pred z={pz}): surface prediction covers {frac:.2f}% of the slice")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"pherc0332_overlay_L{pred_level}_z{z}.png"
    Image.fromarray(rgb).save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
