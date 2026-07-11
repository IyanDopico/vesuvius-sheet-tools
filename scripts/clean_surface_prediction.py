"""Prototype label cleaner for official surface predictions (wishlist #192/#193).

Streams a crop of the official surface prediction plus the matching CT crop,
then cleans the prediction:
  1. CT-mask gating: drop predicted voxels in air (CT == 0), which removes the
     chunk-aligned false positives observed outside the scroll.
  2. Small-component removal: drop 3D connected components below a voxel count.

Saves before/after mid-slice PNGs and the cleaned crop (.npy) to output/.

Usage:
    python scripts/clean_surface_prediction.py [scroll] [pred_level] [z0] [z1] [min_cc]
      scroll:     sample key: scroll3 | pherc1218 (default scroll3)
      pred_level: prediction pyramid level (CT level = pred_level + ct_offset)
      z0, z1:     z range of the crop (default centered 64 slices)
      min_cc:     minimum connected-component size in voxels (default 50)
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from scipy import ndimage

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

# ct_offset: the prediction was run at CT pyramid level N, so prediction
# level L matches CT level L + ct_offset.
SCROLLS = {
    "scroll3": {
        "ct": f"{S3}/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr",
        "pred": (
            f"{S3}/PHerc0332/representations/predictions/surfaces/"
            "20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr"
        ),
        "ct_offset": 2,
        "default_pred_level": 2,
    },
    "pherc1218": {
        "ct": f"{S3}/PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr",
        "pred": (
            f"{S3}/PHerc1218/representations/predictions/surfaces/"
            "20250521120456-surface-20260413222639-surface-m7-L0-th0.2.zarr"
        ),
        "ct_offset": 0,
        "default_pred_level": 3,
    },
}

OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def save_slice(arr: np.ndarray, ct_bg: np.ndarray, path: Path) -> None:
    """Render a label slice in red over the CT background."""
    bg = np.clip(ct_bg * (255.0 / max(ct_bg.max(), 1)), 0, 255).astype(np.uint8)
    rgb = np.stack([bg, bg, bg], axis=-1)
    mask = arr > 0
    rgb[mask] = [255, 40, 40]
    Image.fromarray(rgb).save(path)


def main() -> None:
    scroll = sys.argv[1] if len(sys.argv) > 1 else "scroll3"
    cfg = SCROLLS[scroll]
    pred_level = int(sys.argv[2]) if len(sys.argv) > 2 else cfg["default_pred_level"]
    ct_level = pred_level + cfg["ct_offset"]

    pred_arr = zarr.open_array(f"{cfg['pred']}/{pred_level}", mode="r")
    ct_arr = zarr.open_array(f"{cfg['ct']}/{ct_level}", mode="r")
    zdim = pred_arr.shape[0]
    z0 = int(sys.argv[3]) if len(sys.argv) > 3 else zdim // 2 - 32
    z1 = int(sys.argv[4]) if len(sys.argv) > 4 else zdim // 2 + 32
    min_cc = int(sys.argv[5]) if len(sys.argv) > 5 else 50

    print(f"Streaming crop z[{z0}:{z1}] of PRED L{pred_level} {pred_arr.shape} ...")
    t0 = time.time()
    pred = np.asarray(pred_arr[z0:z1])
    ct = np.asarray(ct_arr[z0:z1, : pred.shape[1], : pred.shape[2]])
    pred = pred[:, : ct.shape[1], : ct.shape[2]]
    print(f"  downloaded {pred.nbytes / 1e6 + ct.nbytes / 1e6:.0f} MB in {time.time() - t0:.1f}s")

    n_raw = int((pred > 0).sum())

    # 1. gate by CT mask (air is 0 in the masked volume)
    gated = np.where(ct > 0, pred, 0)
    n_gated = int((gated > 0).sum())

    # 2. remove small 3D connected components
    labels, n_comp = ndimage.label(gated > 0)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_cc
    keep[0] = False
    cleaned = np.where(keep[labels], gated, 0)
    n_clean = int((cleaned > 0).sum())
    n_kept_comp = int(keep.sum())

    print(f"  raw voxels:            {n_raw:>12,}")
    print(f"  after CT gating:       {n_gated:>12,}  (-{n_raw - n_gated:,}, {100 * (n_raw - n_gated) / max(n_raw, 1):.1f}%)")
    print(f"  after CC filter >={min_cc:>4}: {n_clean:>12,}  (-{n_gated - n_clean:,}; kept {n_kept_comp}/{n_comp} components)")

    OUT_DIR.mkdir(exist_ok=True)
    mid = pred.shape[0] // 2
    tag = f"{scroll}_L{pred_level}_z{z0 + mid}"
    save_slice(pred[mid], ct[mid].astype(np.float32), OUT_DIR / f"clean_before_{tag}.png")
    save_slice(cleaned[mid], ct[mid].astype(np.float32), OUT_DIR / f"clean_after_{tag}.png")
    np.save(OUT_DIR / f"cleaned_crop_{tag}.npy", cleaned)
    print(f"Saved before/after PNGs and cleaned crop to {OUT_DIR}")


if __name__ == "__main__":
    main()
