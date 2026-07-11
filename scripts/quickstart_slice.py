"""Quickstart: stream a slice of Scroll 3 (PHerc0332) from the public S3 bucket.

No credentials needed - the bucket is public (CC-BY-NC 4.0).
Reads a downsampled resolution level so only a few MB are transferred.

Usage:
    python scripts/quickstart_slice.py [level] [axis]
      level: zarr pyramid level (0=full res ... 5=smallest, default 4)
      axis:  z|y|x slice orientation (default z)
"""

import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

BUCKET = "vesuvius-challenge-open-data"
VOLUME = "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr"

OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    level = sys.argv[1] if len(sys.argv) > 1 else "4"
    axis = sys.argv[2] if len(sys.argv) > 2 else "z"

    store_url = f"https://{BUCKET}.s3.amazonaws.com/{VOLUME}"
    print(f"Opening {store_url} (level {level}) ...")
    arr = zarr.open_array(f"{store_url}/{level}", mode="r")
    print(f"  shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks}")

    z, y, x = arr.shape
    if axis == "z":
        sl = arr[z // 2, :, :]
    elif axis == "y":
        sl = arr[:, y // 2, :]
    else:
        sl = arr[:, :, x // 2]
    sl = np.asarray(sl)
    print(f"  slice {axis}={sl.shape}, min={sl.min()} max={sl.max()} mean={sl.mean():.1f}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"pherc0332_L{level}_{axis}mid.png"
    Image.fromarray(sl).save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
