"""Negative control for the layer-count QA (Diego-dcv's suggestion #3).

Injects K known MERGE errors (radially adjacent wraps unified) into one slab's
stitch table, reruns the per-slice ray counts on the target slab and its two
neighbors, and checks that the N(z) deviation fires exactly at the injected
slab with roughly the expected amplitude.

Merge errors are the type per-slice counts can see (a wrongly split wrap keeps
per-slice counts unchanged, since each slice sees exactly one of the halves).
Pairs are chosen radially adjacent along real rays, mimicking genuine
wrap-merge failures.

Usage: python scripts/inject_seam_error.py output/scroll_run [z0_target] [n_pairs]
Output: negative_control.json + printed verdict.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

TILE_RE = re.compile(r"tile_y(\d+)_x(\d+)\.npz")
OVERLAP = 64
STRIDE = 448
GRID = 3797
Z_LOCALS = [16, 48, 80, 112, 144, 176, 208, 240]
THETA_STEP = 6.0


def build_canvases(run: Path, z0: int, table: dict) -> dict[int, np.ndarray]:
    tdir = run / "blocks" / f"z{z0}"
    canvases = {zl: np.zeros((GRID, GRID), dtype=np.int64) for zl in Z_LOCALS}
    tiles = {}
    for p in tdir.glob("tile_*.npz"):
        m = TILE_RE.fullmatch(p.name)
        tiles[(int(m.group(1)), int(m.group(2)))] = p
    for (y0, x0), p in sorted(tiles.items()):
        src = table.get(f"z{z0}/y{y0}_x{x0}")
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


def ray_ids(canvas: np.ndarray):
    """Sequences of distinct ids crossed per ray + mean distinct count."""
    ys, xs = np.nonzero(canvas)
    if len(ys) == 0:
        return [], 0.0
    cy, cx = ys.mean(), xs.mean()
    r_max = int(np.hypot(max(cy, GRID - cy), max(cx, GRID - cx)))
    rr = np.arange(0, r_max, 1.0)
    seqs = []
    counts = []
    for th in np.deg2rad(np.arange(0, 360, THETA_STEP)):
        py = np.clip((cy + rr * np.sin(th)).astype(int), 0, GRID - 1)
        px = np.clip((cx + rr * np.cos(th)).astype(int), 0, GRID - 1)
        ids = canvas[py, px]
        nz = ids[ids > 0]
        counts.append(len(np.unique(nz)))
        # ordered run ids (consecutive distinct crossings)
        if len(nz):
            change = np.flatnonzero(np.diff(nz)) + 1
            seqs.append(nz[np.concatenate([[0], change])])
        else:
            seqs.append(np.array([], dtype=np.int64))
    return seqs, float(np.mean(counts))


def main() -> None:
    run = Path(sys.argv[1])
    z0 = int(sys.argv[2]) if len(sys.argv) > 2 else 4928
    n_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    neighbors = [z0 - 224, z0, z0 + 224]

    table = json.load(open(run / "global_table.json"))

    # pick radially adjacent pairs from the target slab's mid slice
    mid_canvas = build_canvases(run, z0, table)[112]
    seqs, _ = ray_ids(mid_canvas)
    rng = np.random.default_rng(7)
    pairs = []
    seen = set()
    ray_order = rng.permutation(len(seqs))
    for ri in ray_order:
        seq = seqs[ri]
        if len(seq) < 6:
            continue
        j = rng.integers(1, len(seq) - 1)
        a, b = int(seq[j]), int(seq[j + 1])
        if a != b and (a, b) not in seen and (b, a) not in seen:
            pairs.append((a, b))
            seen.add((a, b))
        if len(pairs) >= n_pairs:
            break
    remap = {b: a for a, b in pairs}
    print(f"injecting {len(pairs)} adjacent-wrap merges into slab z{z0}")

    # apply remap to a copy of the table (all slabs - global ids are global)
    injected = {
        cell: {k: remap.get(v, v) for k, v in sub.items()}
        for cell, sub in table.items()
    }

    results = {}
    for zz in neighbors:
        if not (run / "blocks" / f"z{zz}").is_dir():
            continue
        base = build_canvases(run, zz, table)
        mod = build_canvases(run, zz, injected)
        per_slice = {}
        for zl in Z_LOCALS:
            _, n_base = ray_ids(base[zl])
            _, n_mod = ray_ids(mod[zl])
            per_slice[zz + zl] = {"base": round(n_base, 2),
                                  "injected": round(n_mod, 2),
                                  "delta": round(n_mod - n_base, 2)}
        results[f"z{zz}"] = per_slice
        deltas = [v["delta"] for v in per_slice.values()]
        print(f"slab z{zz}: mean dN = {np.mean(deltas):+.2f} "
              f"(per-slice: {[v['delta'] for v in per_slice.values()]})")

    # verdict: target slab must dip, neighbors must not (except shared ids)
    tgt = [v["delta"] for v in results[f"z{z0}"].values()]
    nb = [v["delta"] for k, r in results.items() if k != f"z{z0}"
          for v in r.values()]
    verdict = {
        "injected_pairs": len(pairs),
        "target_mean_delta": float(np.mean(tgt)),
        "neighbor_mean_delta": float(np.mean(nb)) if nb else None,
        "fires_at_target": bool(np.mean(tgt) < -0.5),
        "localized": bool(nb and abs(np.mean(nb)) < abs(np.mean(tgt)) / 3),
    }
    out = {"target_slab": z0, "results": results, "verdict": verdict}
    with open(run / "negative_control.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
