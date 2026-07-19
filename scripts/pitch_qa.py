"""Layer-count QA v2: crossing positions, winding pitch, boundary decomposition.

Implements the three follow-ups from the Discord thread (Diego-dcv) plus the
second calibration anchor pscamillo asked for:

  1. N(z) sampled at 8 slices per slab with slab boundaries marked, so
     deviation clusters can be attributed to stitching seams (on boundaries)
     vs geometry (elsewhere).
  2. Winding pitch per ray: sub-voxel centroids of each crossing run along the
     ray; pitch = median gap between consecutive centroids. Expected physical
     N = span/pitch + 1; counted/expected ratio maps fragmentation (>1) vs
     merge bias (<1) without labels. Pitch outliers ~2x median flag fused
     stacks.
  3. Per-(z,theta) CSV export so raw-prediction counts (pscamillo) can be
     crossed with stitched counts cell by cell.

Usage: python scripts/pitch_qa.py output/scroll_run [theta_step_deg]
Outputs in run dir: pitch_qa.json, pitch_qa_cells.csv, pitch_qa_figure.png
"""

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TILE_RE = re.compile(r"tile_y(\d+)_x(\d+)\.npz")
OVERLAP = 64
STRIDE = 448
GRID = 3797
UM_PER_VOX = 17.28  # L1
SLAB = 256
Z_STRIDE = 224
Z_LOCALS = [16, 48, 80, 112, 144, 176, 208, 240]  # 8 samples per slab

INK = "#3d4451"
MUTED = "#7a8494"
LINE = "#2f6db3"


_GLOBAL_TABLE_CACHE: dict = {}


def slab_canvases(run: Path, z0: int) -> dict[int, np.ndarray] | None:
    """All 8 sampled slices of one slab, stitched to global ids, in one pass."""
    tdir = run / "blocks" / f"z{z0}"
    tpath = run / f"stitch_table_z{z0}.json"
    gpath = run / "global_table.json"
    if not tdir.is_dir() or not tpath.exists():
        return None
    # the global table is ~180 MB of JSON: parse it ONCE per process, not per slab
    if "t" not in _GLOBAL_TABLE_CACHE:
        _GLOBAL_TABLE_CACHE["t"] = (
            json.load(open(gpath)) if gpath.exists() else None)
    global_table = _GLOBAL_TABLE_CACHE["t"]
    slab_table = json.load(open(tpath))
    canvases = {zl: np.zeros((GRID, GRID), dtype=np.int64) for zl in Z_LOCALS}
    tiles = {}
    for p in tdir.glob("tile_*.npz"):
        m = TILE_RE.fullmatch(p.name)
        tiles[(int(m.group(1)), int(m.group(2)))] = p
    for (y0, x0), p in sorted(tiles.items()):
        tkey = f"y{y0}_x{x0}"
        src = None
        if global_table is not None:
            src = global_table.get(f"z{z0}/{tkey}")
        if src is None:
            src = slab_table.get(tkey)
        if not src:
            continue
        with np.load(p) as d:
            labels = d["labels"]
        lut = np.zeros(labels.max() + 1, dtype=np.int64)
        for k, v in src.items():
            if int(k) <= labels.max():
                lut[int(k)] = v
        own_y = OVERLAP // 2 if (y0 - STRIDE, x0) in tiles else 0
        own_x = OVERLAP // 2 if (y0, x0 - STRIDE) in tiles else 0
        for zl in Z_LOCALS:
            sl2d = lut[labels[min(zl, labels.shape[0] - 1), own_y:, own_x:]]
            canvases[zl][y0 + own_y : y0 + own_y + sl2d.shape[0],
                         x0 + own_x : x0 + own_x + sl2d.shape[1]] = sl2d
    return canvases


def ray_metrics(canvas: np.ndarray, theta_step: float):
    """Per ray: distinct-id count, pitch (um) from run centroids, span, ratio."""
    ys, xs = np.nonzero(canvas)
    if len(ys) == 0:
        return None
    cy, cx = ys.mean(), xs.mean()
    r_max = int(np.hypot(max(cy, GRID - cy), max(cx, GRID - cx)))
    rr = np.arange(0, r_max, 1.0)
    out = []
    for th in np.deg2rad(np.arange(0, 360, theta_step)):
        py = np.clip((cy + rr * np.sin(th)).astype(int), 0, GRID - 1)
        px = np.clip((cx + rr * np.cos(th)).astype(int), 0, GRID - 1)
        ids = canvas[py, px]
        n_distinct = len(np.unique(ids[ids > 0]))
        # crossing runs: contiguous nonzero stretches; sub-voxel centroid each
        nz = ids > 0
        if not nz.any():
            out.append((n_distinct, np.nan, np.nan, np.nan))
            continue
        idx = np.flatnonzero(nz)
        breaks = np.flatnonzero(np.diff(idx) > 1)
        run_starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        run_ends = np.concatenate([idx[breaks], [idx[-1]]])
        centroids = (run_starts + run_ends) / 2.0
        if len(centroids) < 3:
            out.append((n_distinct, np.nan, np.nan, np.nan))
            continue
        gaps = np.diff(centroids)
        pitch_vox = float(np.median(gaps))
        span = float(centroids[-1] - centroids[0])
        expected = span / pitch_vox + 1 if pitch_vox > 0 else np.nan
        out.append((n_distinct, pitch_vox * UM_PER_VOX, span * UM_PER_VOX,
                    n_distinct / expected if expected and expected > 0 else np.nan))
    return out, (cy, cx)


def main() -> None:
    run = Path(sys.argv[1])
    theta_step = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    z0s = sorted(int(p.name[1:]) for p in (run / "blocks").iterdir()
                 if p.name.startswith("z"))

    # incremental + resumable: rows are appended per slab; a slab whose last
    # z-local sample (z0+240) is already in the CSV is skipped on restart.
    csv_path = run / "pitch_qa_cells.csv"
    rows = []
    done_z0 = set()
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for r in csv.reader(fh):
                if r and r[0] != "z":
                    rows.append(tuple(float(x) if x != "nan" else np.nan
                                      for x in r))
        zs_in_csv = {int(r[0]) for r in rows}
        done_z0 = {z0 for z0 in z0s if (z0 + Z_LOCALS[-1]) in zs_in_csv}
        print(f"resume: {len(rows)} cells loaded, skipping slabs "
              f"{sorted(done_z0)}", flush=True)
    else:
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["z", "theta_deg", "n_distinct", "pitch_um", "span_um",
                 "counted_over_expected"])

    for z0 in z0s:
        if z0 in done_z0:
            continue
        canvases = slab_canvases(run, z0)
        if canvases is None:
            continue
        slab_rows = []
        for zl, canvas in canvases.items():
            res = ray_metrics(canvas, theta_step)
            if res is None:
                continue
            metrics, _ = res
            for i, (n, pitch, span, ratio) in enumerate(metrics):
                slab_rows.append((z0 + zl, i * theta_step, n, pitch, span,
                                  ratio))
        rows.extend(slab_rows)
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerows(slab_rows)
        p_valid = [r[3] for r in rows if not np.isnan(r[3])]
        print(f"slab z{z0}: {len(rows)} cells so far, "
              f"pitch median {np.median(p_valid):.1f} um" if p_valid else
              f"slab z{z0}: no valid pitch yet", flush=True)

    arr = np.array(rows, dtype=float)
    z = arr[:, 0]
    n = arr[:, 2]
    pitch = arr[:, 3]
    ratio = arr[:, 5]
    valid = ~np.isnan(pitch) & (n >= 5)
    interior = valid & (z > 1024) & (z < 10880)
    summary = {
        "cells_total": int(len(arr)),
        "pitch_um": {
            "median": float(np.nanmedian(pitch[interior])),
            "iqr": [float(np.nanpercentile(pitch[interior], 25)),
                    float(np.nanpercentile(pitch[interior], 75))],
        },
        "counted_over_expected": {
            "median": float(np.nanmedian(ratio[interior])),
            "iqr": [float(np.nanpercentile(ratio[interior], 25)),
                    float(np.nanpercentile(ratio[interior], 75))],
        },
        "pitch_outlier_cells_2x": int(
            (pitch[interior] > 2 * np.nanmedian(pitch[interior])).sum()),
        "theta_step_deg": theta_step,
        "z_samples_per_slab": len(Z_LOCALS),
    }

    # boundary decomposition: deviation of per-slice median N vs local baseline
    z_unique = np.unique(z)
    medN = np.array([np.median(n[z == zz]) for zz in z_unique])
    base = np.convolve(medN, np.ones(9) / 9, mode="same")
    dev = np.abs(medN - base)
    boundaries = set()
    for s in range(Z_STRIDE, 11500, Z_STRIDE):
        boundaries.add(s)
        boundaries.add(s + SLAB - Z_STRIDE)  # overlap edges
    near_b = np.array([min(abs(zz - b) for b in boundaries) <= 32
                       for zz in z_unique])
    inter = (z_unique > 1024) & (z_unique < 10880)
    summary["boundary_decomposition"] = {
        "mean_abs_dev_near_boundary": float(dev[near_b & inter].mean()),
        "mean_abs_dev_interior": float(dev[~near_b & inter].mean()),
        "slices_near_boundary": int((near_b & inter).sum()),
        "slices_interior": int((~near_b & inter).sum()),
    }
    with open(run / "pitch_qa.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    # figure: N(z) dense profile with boundaries + pitch histogram
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150,
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    fig.patch.set_facecolor("white")
    for b in sorted(boundaries):
        if b < z_unique.max():
            ax.axvline(b, color="#e3b04b", linewidth=0.7, alpha=0.55, zorder=1)
    ax.plot(z_unique, medN, color=LINE, linewidth=1.4, zorder=3)
    ax.set_xlabel("z (L1 slice)", fontsize=9, color=INK)
    ax.set_ylabel("median wraps per ray", fontsize=9, color=INK)
    ax.set_title("N(z) at 8 slices/slab — slab boundaries in amber",
                 fontsize=10, color=INK, loc="left")
    ph = pitch[interior]
    ax2.hist(ph[~np.isnan(ph)], bins=np.arange(100, 400, 8),
             color=LINE, alpha=0.85)
    med = np.nanmedian(ph)
    ax2.axvline(med, color=INK, linewidth=1.2)
    ax2.text(med + 6, ax2.get_ylim()[1] * 0.92, f"median {med:.0f} µm",
             fontsize=8.5, color=INK)
    ax2.axvline(207, color="#c65a49", linewidth=1.0, linestyle="--")
    ax2.text(209, ax2.get_ylim()[1] * 0.78, "collection 207 µm\n(pscamillo)",
             fontsize=7.5, color="#c65a49")
    ax2.set_xlabel("winding pitch (µm)", fontsize=9, color=INK)
    ax2.set_title("PHerc1218 stitched pitch", fontsize=10, color=INK, loc="left")
    for a in (ax, ax2):
        a.tick_params(labelsize=8, colors=MUTED)
        for s in a.spines.values():
            s.set_color("#d5dae2")
    fig.tight_layout()
    fig.savefig(run / "pitch_qa_figure.png", bbox_inches="tight",
                facecolor="white")
    print(f"saved {run / 'pitch_qa_figure.png'}")


if __name__ == "__main__":
    main()
