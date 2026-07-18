"""Layer-count QA over stitched slab outputs (Diego-dcv's Obs. 1, our ray form).

For the mid-slice of every processed slab: reconstruct the stitched global
label slice, cast rays from the section centroid, and count distinct instances
crossed per ray. Any valid segmentation must recover ~the same layer count N
at every interior cross-section, so the (z, theta) profile localizes seam
errors without any ground truth. Per Diego's note: the report is the per-axis
N profile (not pass/fail), and the anchor should be chosen conservatively.

Usage: python scripts/layer_count_qa.py output/scroll_run [theta_step_deg]
Outputs: layer_count.json + layer_count_profile.png in the run dir.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TILE_RE = re.compile(r"tile_y(\d+)_x(\d+)\.npz")
OVERLAP = 64
STRIDE = 448
GRID = 3797  # L1 y/x extent of PHerc1218
ZMID_LOCAL = 128


def slice_canvas(run: Path, z0: int) -> np.ndarray | None:
    tdir = run / "blocks" / f"z{z0}"
    table_path = run / f"stitch_table_z{z0}.json"
    if not tdir.is_dir() or not table_path.exists():
        return None
    with open(table_path) as fh:
        table = json.load(fh)
    canvas = np.zeros((GRID, GRID), dtype=np.int32)
    tiles = {}
    for p in tdir.glob("tile_*.npz"):
        m = TILE_RE.fullmatch(p.name)
        tiles[(int(m.group(1)), int(m.group(2)))] = p
    for (y0, x0), p in sorted(tiles.items()):
        t = table.get(f"y{y0}_x{x0}")
        if not t:
            continue
        with np.load(p) as d:
            zdim = d["labels"].shape[0]
            sl2d = d["labels"][min(ZMID_LOCAL, zdim - 1)]
        lut = np.zeros(sl2d.max() + 1, dtype=np.int32)
        for k, v in t.items():
            if int(k) <= sl2d.max():
                lut[int(k)] = v
        own_y = OVERLAP // 2 if (y0 - STRIDE, x0) in tiles else 0
        own_x = OVERLAP // 2 if (y0, x0 - STRIDE) in tiles else 0
        piece = lut[sl2d[own_y:, own_x:]]
        canvas[y0 + own_y : y0 + own_y + piece.shape[0],
               x0 + own_x : x0 + own_x + piece.shape[1]] = piece
    return canvas


def ray_counts(canvas: np.ndarray, theta_step: float) -> np.ndarray:
    ys, xs = np.nonzero(canvas)
    if len(ys) == 0:
        return np.zeros(0)
    cy, cx = ys.mean(), xs.mean()
    r_max = int(np.hypot(max(cy, GRID - cy), max(cx, GRID - cx)))
    rr = np.arange(0, r_max, 1.0)
    thetas = np.deg2rad(np.arange(0, 360, theta_step))
    counts = np.zeros(len(thetas))
    for i, th in enumerate(thetas):
        py = np.clip((cy + rr * np.sin(th)).astype(int), 0, GRID - 1)
        px = np.clip((cx + rr * np.cos(th)).astype(int), 0, GRID - 1)
        ids = canvas[py, px]
        counts[i] = len(np.unique(ids[ids > 0]))
    return counts


def main() -> None:
    run = Path(sys.argv[1])
    theta_step = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    z0s = sorted(int(p.name[1:]) for p in (run / "blocks").iterdir()
                 if p.name.startswith("z"))
    profile = {}
    for z0 in z0s:
        canvas = slice_canvas(run, z0)
        if canvas is None:
            continue
        counts = ray_counts(canvas, theta_step)
        if len(counts) == 0:
            continue
        profile[z0] = counts.tolist()
        print(f"z{z0 + ZMID_LOCAL}: N median {np.median(counts):.0f}, "
              f"p10 {np.percentile(counts, 10):.0f}, "
              f"p90 {np.percentile(counts, 90):.0f}", flush=True)

    with open(run / "layer_count.json", "w") as fh:
        json.dump({"theta_step_deg": theta_step, "profile": profile}, fh)

    # heatmap (z x theta) + median curve
    z_keys = sorted(profile)
    mat = np.array([profile[z] for z in z_keys])
    norm = np.clip(mat / max(mat.max(), 1), 0, 1)
    img = (norm * 255).astype(np.uint8)
    img = np.kron(img, np.ones((12, 3), dtype=np.uint8))  # upscale for viewing
    Image.fromarray(img).save(run / "layer_count_profile.png")
    med = [float(np.median(profile[z])) for z in z_keys]
    print("\nper-axis N profile (median per slab mid-slice):")
    for z, m in zip(z_keys, med):
        bar = "#" * int(m / 2)
        print(f"  z{z + ZMID_LOCAL:5d}: {m:5.0f} {bar}")


if __name__ == "__main__":
    main()
