"""Umbilicus constraint for official spiral fitting, from instance-label blocks.

Composites the nonzero-label mask of sampled slices per slab (local per-tile
labels suffice for a centroid mask, so no stitch tables are needed), takes the
papyrus centroid of each slice as the umbilicus position, smooths the
z->(y,x) series with a running median, and converts to full-resolution
voxels. Output schema matches the official loader (umbilicus.py in
ScrollPrize/villa), which reads {"control_points": [{"z","y","x"}, ...]} in
full-res voxels and sorts by z itself.

Usage:
    python scripts/constraints/make_umbilicus.py <run_dir>
        [--slabs 4928,5152]   comma-separated slab z0 list (default: all)
        [--out DIR]           output directory (default: run_dir)

Outputs in --out: umbilicus.json, umbilicus_check.png
Exits nonzero with a message if any self-check fails.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TILE_RE = re.compile(r"tile_y(\d+)_x(\d+)\.npz")
OVERLAP = 64
STRIDE = 448
GRID = 3797  # L1 y/x extent of PHerc1218
Z_LOCALS = [16, 48, 80, 112, 144, 176, 208, 240]  # samples per 256-slice slab
MIN_PIXELS = 10_000  # skip slices with fewer nonzero pixels
MEDIAN_WINDOW = 5  # running-median smoothing window (slices)
SCALE = 2  # L1 -> full-resolution voxels
FULL_Z = 23247  # full-res grid dims
FULL_YX = 7593
MAX_JUMP = 400  # max allowed consecutive (y,x) jump, full-res voxels

INK = "#3d4451"
MUTED = "#7a8494"


def slab_mask_canvases(run: Path, z0: int) -> dict[int, np.ndarray] | None:
    """All sampled slices of one slab as composited nonzero-label masks.

    Ownership rule (same as the stitched-canvas QA scripts): a tile owns its
    region except the first OVERLAP//2 rows/cols of an overlap when a
    lower-origin neighbor tile exists; later tiles overwrite the remainder.
    """
    tdir = run / "blocks" / f"z{z0}"
    if not tdir.is_dir():
        return None
    tiles = {}
    for p in tdir.glob("tile_*.npz"):
        m = TILE_RE.fullmatch(p.name)
        if m:
            tiles[(int(m.group(1)), int(m.group(2)))] = p
    if not tiles:
        return None
    canvases = {zl: np.zeros((GRID, GRID), dtype=bool) for zl in Z_LOCALS}
    for (y0, x0), p in sorted(tiles.items()):
        with np.load(p) as d:
            mask = d["labels"] > 0
        own_y = OVERLAP // 2 if (y0 - STRIDE, x0) in tiles else 0
        own_x = OVERLAP // 2 if (y0, x0 - STRIDE) in tiles else 0
        for zl in Z_LOCALS:
            sl2d = mask[min(zl, mask.shape[0] - 1), own_y:, own_x:]
            canvases[zl][y0 + own_y : y0 + own_y + sl2d.shape[0],
                         x0 + own_x : x0 + own_x + sl2d.shape[1]] = sl2d
    return canvases


def running_median(values: np.ndarray, window: int) -> np.ndarray:
    """Running median with window clipped at the series edges."""
    half = window // 2
    out = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        out[i] = np.median(values[max(0, i - half) : i + half + 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build umbilicus.json control points from label blocks.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--slabs", default=None,
                    help="comma-separated slab z0 values (default: all)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: run_dir)")
    args = ap.parse_args()
    run = args.run_dir
    out_dir = args.out if args.out is not None else run
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = run / "blocks"
    if not blocks.is_dir():
        sys.exit(f"error: {blocks} not found")
    all_z0 = sorted(int(p.name[1:]) for p in blocks.iterdir()
                    if p.is_dir() and p.name.startswith("z"))
    if args.slabs:
        wanted = {int(s) for s in args.slabs.split(",")}
        missing = wanted - set(all_z0)
        if missing:
            sys.exit(f"error: requested slabs not in run: {sorted(missing)}")
        z0s = [z for z in all_z0 if z in wanted]
    else:
        z0s = all_z0
    if not z0s:
        sys.exit("error: no slabs to process")

    # collect one centroid per sampled slice; consecutive slabs overlap 32
    # slices in z, so the same global z can be sampled twice - keep the first
    points = []  # (z_L1, cy, cx)
    seen_z = set()
    occupied = []  # slab z0s that contributed >=1 control point
    for z0 in z0s:
        canvases = slab_mask_canvases(run, z0)
        if canvases is None:
            print(f"slab z{z0}: no tiles, skipped", flush=True)
            continue
        got = 0
        for zl in Z_LOCALS:
            zg = z0 + zl
            if zg in seen_z:
                continue
            canvas = canvases[zl]
            n = int(canvas.sum())
            if n < MIN_PIXELS:
                continue
            ys, xs = np.nonzero(canvas)
            points.append((zg, float(ys.mean()), float(xs.mean())))
            seen_z.add(zg)
            got += 1
        if got:
            occupied.append(z0)
        print(f"slab z{z0}: {got} control slices", flush=True)

    if len(points) < 2:
        sys.exit("error: fewer than 2 valid slices - cannot build umbilicus")

    points.sort()
    z_l1 = np.array([p[0] for p in points], dtype=float)
    y_l1 = running_median(np.array([p[1] for p in points]), MEDIAN_WINDOW)
    x_l1 = running_median(np.array([p[2] for p in points]), MEDIAN_WINDOW)
    zf, yf, xf = z_l1 * SCALE, y_l1 * SCALE, x_l1 * SCALE

    # self-checks
    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"self-check {name}: {'PASS' if ok else 'FAIL'} ({detail})")
        if not ok:
            failures.append(name)

    check("z strictly increasing", bool(np.all(np.diff(zf) > 0)),
          f"{len(zf)} points, z {zf[0]:.0f}..{zf[-1]:.0f}")
    in_bounds = bool(np.all((zf >= 0) & (zf < FULL_Z)
                            & (yf >= 0) & (yf < FULL_YX)
                            & (xf >= 0) & (xf < FULL_YX)))
    check("z,y,x in bounds", in_bounds,
          f"z {zf.min():.0f}..{zf.max():.0f} within [0,{FULL_Z}); "
          f"y {yf.min():.0f}..{yf.max():.0f}, x {xf.min():.0f}..{xf.max():.0f}"
          f" within [0,{FULL_YX})")
    jumps = np.hypot(np.diff(yf), np.diff(xf))
    check("consecutive jumps", bool(np.all(jumps < MAX_JUMP)),
          f"max jump {jumps.max():.1f} < {MAX_JUMP} full-res voxels")
    gaps = [z0 for z0 in z0s
            if occupied[0] < z0 < occupied[-1] and z0 not in occupied]
    check("coverage first-to-last occupied slab", not gaps,
          f"occupied z{occupied[0]}..z{occupied[-1]}, "
          f"{len(occupied)} slabs, gaps={gaps if gaps else 'none'}")

    cps = [{"z": float(a), "y": float(b), "x": float(c)}
           for a, b, c in zip(zf, yf, xf)]
    json_path = out_dir / "umbilicus.json"
    with open(json_path, "w") as fh:
        json.dump({"control_points": cps}, fh, indent=2)
    print(f"saved {json_path} ({len(cps)} control points, full-res voxels)")

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(zf, yf, color="#2f6db3", linewidth=1.4, label="y(z)")
    ax.plot(zf, xf, color="#c65a49", linewidth=1.4, label="x(z)")
    ax.set_xlabel("z (full-res voxel)", fontsize=9, color=INK)
    ax.set_ylabel("umbilicus position (full-res voxel)", fontsize=9, color=INK)
    ax.set_title("PHerc1218 umbilicus control points", fontsize=10,
                 color=INK, loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#d5dae2")
    fig.tight_layout()
    png_path = out_dir / "umbilicus_check.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    print(f"saved {png_path}")

    if failures:
        sys.exit(f"self-checks failed: {', '.join(failures)}")
    print("all self-checks PASS")


if __name__ == "__main__":
    main()
