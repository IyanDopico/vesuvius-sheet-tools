"""Synthesize a seed verified tifxyz patch from stitched instance labels.

fit_spiral.py (villa volume-cartographer/scripts/spiral) requires at least one
verified patch: the umbilicus loss is computed inside the patch loss path, which
samples patches unconditionally every step. On a scroll with zero human
segments there is no verified patch to give it - so we synthesize one from the
stitched sheet-instance labels themselves: a z x theta grid laid on one clean
instance, with per-bin dominant-cluster median radii (robust to multi-turn
instances and stacked-sheet bins).

Output layout mirrors villa's tifxyz.save_tifxyz exactly (x/y/z.tif float32 +
meta.json), coordinates in full-resolution voxels, invalid vertices -1.

Usage:
  python scripts/constraints/make_seed_patch.py RUN_DIR PACK_DIR Z0 [UUID]

  RUN_DIR   stitched-label run dir (blocks/, stitch tables, global_table.json)
  PACK_DIR  spiral input pack holding umbilicus.json; the patch is written to
            PACK_DIR/verified_patches/UUID/
  Z0        L1 slab start (e.g. 4928); the 8 sampled slices of that slab become
            the grid rows
  UUID      patch dir name (default: seed-z{Z0}-auto)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pitch_qa  # slab_canvases + Z_LOCALS

STEP_FULLRES = 64.0            # nominal grid step (z rows are 32 L1 = 64 vox)
MIN_PIX_PER_CLUSTER = 3
CLUSTER_GAP = 5.0              # L1 vox: radial gap separating turns
MAX_CLUSTER_SPREAD = 8.0       # L1 vox: dominant cluster tightness gate
MAX_COLS = 40
MIN_VALID_QUADS = 20
GRID_FULLRES = 7593


def dominant_cluster(rr: np.ndarray) -> float:
    """Median of the largest radial cluster (split at gaps > CLUSTER_GAP)."""
    rs = np.sort(rr)
    splits = np.flatnonzero(np.diff(rs) > CLUSTER_GAP)
    starts = np.concatenate([[0], splits + 1])
    ends = np.concatenate([splits, [len(rs) - 1]])
    k = np.argmax(ends - starts)
    cl = rs[starts[k]:ends[k] + 1]
    if len(cl) < MIN_PIX_PER_CLUSTER or cl[-1] - cl[0] > MAX_CLUSTER_SPREAD:
        return np.nan
    return float(np.median(cl))


def main() -> None:
    run = Path(sys.argv[1])
    pack = Path(sys.argv[2])
    z0 = int(sys.argv[3])
    uuid = sys.argv[4] if len(sys.argv) > 4 else f"seed-z{z0}-auto"
    z_locals = pitch_qa.Z_LOCALS

    umb = json.load(open(pack / "umbilicus.json"))["control_points"]
    umb = (np.array([[p["x"], p["y"], p["z"]] for p in umb], dtype=float)
           if isinstance(umb[0], dict) else np.array(umb, dtype=float))

    def umb_yx_l1(z_l1: float):
        i = np.argmin(np.abs(umb[:, 2] - 2.0 * z_l1))
        return umb[i, 1] / 2.0, umb[i, 0] / 2.0

    print("reconstructing slab canvases (parses global table once)...",
          flush=True)
    canvases = pitch_qa.slab_canvases(run, z0)
    assert canvases is not None, f"slab z{z0} not found in {run}"

    counts: dict = {}
    for zl, canvas in canvases.items():
        ids, c = np.unique(canvas[canvas > 0], return_counts=True)
        for i, n in zip(ids, c):
            counts.setdefault(int(i), []).append(int(n))
    cands = [i for i, ns in counts.items() if len(ns) == len(z_locals)
             and min(ns) > 400]
    print(f"{len(cands)} instances present in all {len(z_locals)} slices",
          flush=True)

    best = None
    for gid in sorted(cands, key=lambda i: -min(counts[i]))[:40]:
        rows = []
        r_mids = []
        for zl in z_locals:
            canvas = canvases[zl]
            ys, xs = np.nonzero(canvas == gid)
            uy, ux = umb_yx_l1(z0 + zl)
            th = np.arctan2(ys - uy, xs - ux)
            r = np.hypot(ys - uy, xs - ux)
            rows.append((zl, uy, ux, th, r))
            r_mids.append(np.median(r))
        r_mid = float(np.median(r_mids))
        dth = (STEP_FULLRES / 2.0) / r_mid
        nbins = int(2 * np.pi / dth)
        med_r = np.full((len(z_locals), nbins), np.nan)
        for k, (zl, uy, ux, th, r) in enumerate(rows):
            b = ((th + np.pi) / dth).astype(int) % nbins
            for bb in np.unique(b):
                med_r[k, bb] = dominant_cluster(r[b == bb])
        score = (~np.isnan(med_r)).sum(axis=0).astype(float)
        ext = np.concatenate([score, score])
        w = min(MAX_COLS, nbins)
        csum = np.concatenate([[0.0], np.cumsum(ext)])
        win = csum[w:] - csum[:-w]
        s = int(np.argmax(win[:nbins]))
        vertices = float(win[s])
        if best is None or vertices > best[0]:
            best = (vertices, gid, s, w, dth, nbins, med_r, rows, r_mid)

    assert best is not None, "no candidate instances - check RUN_DIR/Z0"
    vertices, gid, start, ncols, dth, nbins, med_r, rows, r_mid = best
    print(f"instance {gid}: {ncols}-bin window, {vertices:.0f}/"
          f"{ncols * len(z_locals)} valid vertices (r_mid~{r_mid:.0f} L1 vox)",
          flush=True)

    grid = np.full((len(z_locals), ncols, 3), -1.0, dtype=np.float32)  # zyx
    for k, (zl, uy, ux, _th, _r) in enumerate(rows):
        for c in range(ncols):
            bb = (start + c) % nbins
            r = med_r[k, bb]
            if np.isnan(r):
                continue
            theta = -np.pi + (bb + 0.5) * dth
            grid[k, c] = (2.0 * (z0 + zl),
                          2.0 * (uy + r * np.sin(theta)),
                          2.0 * (ux + r * np.cos(theta)))

    valid = np.any(grid != -1.0, axis=-1)
    vq = (valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:])
    print(f"grid {grid.shape[0]}x{grid.shape[1]}: {valid.sum()} vertices, "
          f"{vq.sum()} valid quads", flush=True)
    assert vq.sum() >= MIN_VALID_QUADS, "too few valid quads"
    g = grid[valid]
    assert 0 < g[:, 1].min() and g[:, 1].max() < GRID_FULLRES
    assert 0 < g[:, 2].min() and g[:, 2].max() < GRID_FULLRES

    path = pack / "verified_patches" / uuid
    os.makedirs(path, exist_ok=True)
    Image.fromarray(grid[..., 2]).save(path / "x.tif")
    Image.fromarray(grid[..., 1]).save(path / "y.tif")
    Image.fromarray(grid[..., 0]).save(path / "z.tif")
    area_vx2 = int(vq.sum()) * STEP_FULLRES ** 2
    json.dump({
        "scale": [1 / STEP_FULLRES, 1 / STEP_FULLRES],
        "bbox": [g.min(axis=0)[::-1].tolist(), g.max(axis=0)[::-1].tolist()],
        "area_vx2": area_vx2,
        "area_cm2": area_vx2 * 8.64 ** 2 / 1.0e8,
        "format": "tifxyz",
        "type": "seg",
        "uuid": uuid,
        "source": "vesuvius-sheet-tools make_seed_patch.py "
                  f"(instance {gid}, slab z{z0} L1, dominant-cluster radii)",
    }, open(path / "meta.json", "w"), indent=4)

    # SELF-CHECKS: file round-trip + loader-style validity
    zyxs = np.stack([np.array(Image.open(path / f"{c}.tif"))
                     for c in "zyx"], axis=-1)
    assert zyxs.dtype == np.float32 and np.allclose(zyxs, grid)
    print(f"SELF-CHECKS: PASS - wrote {path}", flush=True)


if __name__ == "__main__":
    main()
