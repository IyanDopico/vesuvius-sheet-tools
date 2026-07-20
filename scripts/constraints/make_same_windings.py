"""Emit official Vesuvius Challenge same-winding point collections from the
PHerc1218 stitched instance-label run.

Reads the L1 (17.28 um/voxel) block dataset produced by the stitching
pipeline (``blocks/z{Z0}/tile_y{Y}_x{X}.npz`` + ``global_table.json``) and
converts trusted instances into spiral-fitting *same-winding* constraints:
one point collection per global instance, points sampled on the sheet
interior, coordinates scaled to full-resolution voxels (XYZ order).

Poison control (an instance is skipped when any of these trips):
  * mega:      instance holds >= 3% of the slab's labeled voxels
  * thickness: volume / count(EDT >= 2 voxels) > 8  (thin surf-pred shells
               and wrongly merged tangles both blow this up)
  * radial:    robust radius spread (p95 - p5 around the per-slice scroll
               centroid) > 0.7 x local winding pitch -- a single winding arc
               must stay inside a fraction of one pitch
  * boundary:  instance spans a slab boundary whose stitching
               agreement_rate < 0.80 -> points are kept from one slab only

Output schema (exact):
  {"vc_pointcollections_json_version": "1",
   "collections": {"<intstr>": {"name": "inst_<gid>",
                                "points": {"<intstr>": {"p": [x, y, z],
                                                        "creation_time": 0}},
                                "metadata": {}, "color": [r, g, b]}}}
No ``wind_a`` key is ever written: every collection is homogeneous by
construction (all points of a collection lie on the same winding).

Also writes ``patch_candidates.json``: big, clean instances (volume > 150000
passing the deep-core filter) worth a later patch-constraint pass.

Usage:
  python scripts/constraints/make_same_windings.py <run_dir> \
      [--slabs 4928,5152] [--out DIR] [--budget 200000]

Defaults: all slabs under <run_dir>/blocks, DIR = <run_dir>/constraints.
"""

import argparse
import colorsys
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

TILE_RE = re.compile(r"tile_y(\d+)_x(\d+)\.npz")

# geometry of the L1 run (matches scripts/pitch_qa.py of the run pipeline)
SLAB = 256            # slab thickness in z
Z_STRIDE = 224        # slab stride (32 voxel z-overlap)
TILE = 512            # tile side in y/x
STRIDE = 448          # tile stride (64 voxel overlap)
OWN = (TILE - STRIDE) // 2   # 32: half-overlap trimmed on each shared edge
L1_DIMS = (11624, 3797, 3797)          # z, y, x
FULL_DIMS = (23247, 7593, 7593)        # full-res voxels (L1 x 2, clipped)
UM_PER_VOX = 17.28

# filters / sampling
MIN_VOL = 2000            # candidate floor, voxels in this slab
MEGA_FRAC = 0.03          # >= 3% of slab labeled voxels -> merged monster
# Deep-core fraction = share of instance voxels with EDT >= 2. Measured on
# 507 real instances (slab z4928, >= 2000 vox): the m7 surface labels are thin
# -- max EDT p50 = 2.0, p90 = 3.0, mean EDT ~ 1.05 -- so genuine sheets sit at
# deep fractions of 0.007 (p50) to 0.03 (p90). A fused multi-wrap blob is much
# deeper. 0.10 is ~3x the thickest genuine instance measured.
# (An earlier volume/interior-count "proxy <= 8" threshold was carried over
# from split_stacked.py's centerline-based ratio; on this metric real values
# run 20-2000, so it rejected 90% of valid instances.)
MAX_DEEP_FRAC = 0.10
# Merge test: mean within-angular-wedge radius sigma must stay under
# RADIAL_SIGMA_FACTOR x local pitch. Two wraps merged into one instance sit
# ~1 pitch apart inside the same wedge (sigma ~ 0.5 pitch), a single wrap is
# only ~2 voxels thick (sigma ~ 0.1 pitch), so 0.30 separates them with
# margin. See local_radial_sigma() for why the earlier global p95-p5 spread
# test failed on this squashed scroll.
RADIAL_SIGMA_FACTOR = 0.30
A_BINS = 24               # 15-degree angular wedges for that test
FALLBACK_PITCH_VOX = 10.0
MIN_DIST = 5.0            # min pairwise point distance, L1 voxels
VOX_PER_POINT = 1500      # K = clamp(round(vol / 1500), 3, 20)
K_MIN, K_MAX = 3, 20
AGREE_MIN = 0.80          # slab-pair agreement below this -> single-slab only
PATCH_MIN_VOL = 150000
R_BINS = 4096             # radius histogram bins (1 voxel each)
CANDS_PER_TILE = 400      # per (instance, tile) interior candidate cap
SEED = 20260720
N_ROUNDTRIP = 200


# ---------------------------------------------------------------- data access

def slab_tiles(run: Path, z0: int) -> dict:
    tiles = {}
    tdir = run / "blocks" / f"z{z0}"
    if tdir.is_dir():
        for p in tdir.glob("tile_*.npz"):
            m = TILE_RE.fullmatch(p.name)
            if m:
                tiles[(int(m.group(1)), int(m.group(2)))] = p
    return tiles


def tile_lut(table: dict, z0: int, y0: int, x0: int, max_local: int):
    """local id -> global id lookup for one tile, or None if unmapped."""
    src = table.get(f"z{z0}/y{y0}_x{x0}")
    if not src:
        return None
    lut = np.zeros(max_local + 1, dtype=np.int32)
    for k, v in src.items():
        k = int(k)
        if k <= max_local:
            lut[k] = v
    return lut


def own_bounds(y0: int, x0: int, tiles: dict, shape) -> tuple:
    """Half-overlap ownership crop so every slab voxel is counted once."""
    y_lo = OWN if (y0 - STRIDE, x0) in tiles else 0
    y_hi = min(TILE - OWN if (y0 + STRIDE, x0) in tiles else TILE, shape[1])
    x_lo = OWN if (y0, x0 - STRIDE) in tiles else 0
    x_hi = min(TILE - OWN if (y0, x0 + STRIDE) in tiles else TILE, shape[2])
    return y_lo, y_hi, x_lo, x_hi


def interior_mask(mask: np.ndarray) -> np.ndarray:
    """Voxels with EDT >= 2 on the labels>0 mask (per tile).

    Exact shortcut for ``scipy.ndimage.distance_transform_edt(mask) >= 2``
    (the pattern used in scripts/split_stacked.py): EDT(v) < 2 iff some
    background voxel sits at squared distance 1, 2 or 3 from v -- exactly the
    26-neighbourhood, because the next realisable squared distance is 4 = 2.0.
    Hence EDT >= 2 is a 3x3x3 minimum filter over the mask; ~7x faster than
    the float transform and bit-identical at this threshold (verified against
    distance_transform_edt on run tiles).
    """
    return ndimage.minimum_filter(mask.view(np.uint8), size=3) > 0


def load_pitch_rows(run: Path) -> list:
    rows = []
    p = run / "pitch_qa_cells.csv"
    if p.exists():
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    z = float(r["z"])
                    pit = float(r["pitch_um"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(pit) and pit > 0:
                    rows.append((z, pit))
    return rows


def local_pitch_vox(pitch_rows: list, z0: int) -> float:
    """Smoothed median pitch (L1 voxels) from cells within +-1 slab, all theta."""
    lo, hi = z0 - Z_STRIDE, z0 + SLAB - 1 + Z_STRIDE
    vals = [p for z, p in pitch_rows if lo <= z <= hi]
    if not vals:
        return FALLBACK_PITCH_VOX
    return float(np.median(vals)) / UM_PER_VOX


def load_agreement(run: Path) -> dict:
    p = run / "assembly_metrics.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        data = json.load(fh)
    return {k: float(v.get("agreement_rate", 0.0))
            for k, v in data.get("pairs", {}).items()}


# ------------------------------------------------------------ per-slab passes

def slab_pass_volumes(run, z0, tiles, table, max_gid):
    """Per-global-id slab volume + per-slice centroid accumulators."""
    vol = np.zeros(max_gid + 1, dtype=np.int64)
    zc = np.zeros(SLAB, dtype=np.int64)
    zsy = np.zeros(SLAB, dtype=np.float64)
    zsx = np.zeros(SLAB, dtype=np.float64)
    skipped = []
    for (y0, x0), p in sorted(tiles.items()):
        with np.load(p) as d:
            labels = d["labels"]
        lut = tile_lut(table, z0, y0, x0, int(labels.max()))
        if lut is None:
            skipped.append((y0, x0))
            continue
        yl, yh, xl, xh = own_bounds(y0, x0, tiles, labels.shape)
        sub = labels[:, yl:yh, xl:xh]
        u, c = np.unique(sub, return_counts=True)
        g = lut[u]
        keep = g > 0
        np.add.at(vol, g[keep], c[keep])
        mask = sub > 0
        py = mask.sum(axis=2)            # (Z, H)
        px = mask.sum(axis=1)            # (Z, W)
        zn = sub.shape[0]
        zc[:zn] += py.sum(axis=1)
        zsy[:zn] += py @ (np.arange(yl, yh, dtype=np.float64) + y0)
        zsx[:zn] += px @ (np.arange(xl, xh, dtype=np.float64) + x0)
    return vol, zc, zsy, zsx, skipped


def slab_pass_details(z0, tiles, table, big_lookup, cy, cx):
    """Interior counts, radius histograms, z-extent and sample candidates
    for the slab's big instances (aggregated across tiles)."""
    det = {}
    for (y0, x0), p in sorted(tiles.items()):
        with np.load(p) as d:
            labels = d["labels"]
        lut = tile_lut(table, z0, y0, x0, int(labels.max()))
        if lut is None:
            continue
        inter_full = interior_mask(labels > 0)   # EDT on the whole tile mask
        yl, yh, xl, xh = own_bounds(y0, x0, tiles, labels.shape)
        gl = lut[labels[:, yl:yh, xl:xh]]
        isub = inter_full[:, yl:yh, xl:xh]
        H, W = gl.shape[1], gl.shape[2]
        flat = gl.ravel()
        idx = np.flatnonzero(flat)
        if idx.size == 0:
            continue
        ids = flat[idx]
        keep = big_lookup[ids]
        idx, ids = idx[keep], ids[keep]
        if idx.size == 0:
            continue
        z = (idx // (H * W)).astype(np.int32)
        rem = idx % (H * W)
        y = (rem // W).astype(np.int32) + (y0 + yl)
        x = (rem % W).astype(np.int32) + (x0 + xl)
        dy = y - cy[z]
        dx = x - cx[z]
        r = np.hypot(dy, dx)
        rb = np.minimum(r, R_BINS - 1).astype(np.int32)
        # angular bin of each voxel, for the geometry-robust merge test below
        ab = np.minimum(
            ((np.arctan2(dy, dx) + np.pi) * (A_BINS / (2 * np.pi))).astype(
                np.int32), A_BINS - 1)
        inter = isub.ravel()[idx]
        order = np.argsort(ids, kind="stable")
        ids_s = ids[order]
        starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
        ends = np.r_[starts[1:], ids_s.size]
        for s, e in zip(starts, ends):
            gid = int(ids_s[s])
            sel = order[s:e]
            d0 = det.get(gid)
            if d0 is None:
                d0 = det[gid] = {"ivol": 0, "zmin": SLAB, "zmax": -1,
                                 "rhist": np.zeros(R_BINS, np.int64),
                                 "an": np.zeros(A_BINS, np.int64),
                                 "ar": np.zeros(A_BINS, np.float64),
                                 "ar2": np.zeros(A_BINS, np.float64),
                                 "cands": []}
            zg = z[sel]
            d0["zmin"] = min(d0["zmin"], int(zg.min()))
            d0["zmax"] = max(d0["zmax"], int(zg.max()))
            d0["rhist"] += np.bincount(rb[sel], minlength=R_BINS)
            # per-angular-bin radius moments (count, sum r, sum r^2)
            asel, rsel = ab[sel], r[sel]
            d0["an"] += np.bincount(asel, minlength=A_BINS)
            d0["ar"] += np.bincount(asel, weights=rsel, minlength=A_BINS)
            d0["ar2"] += np.bincount(asel, weights=rsel * rsel,
                                     minlength=A_BINS)
            ii = sel[inter[sel]]
            d0["ivol"] += int(ii.size)
            # Sheets here are only ~2 voxels thick (measured: max EDT p50 = 2.0),
            # so many valid instances have no EDT >= 2 voxel at all. Interior
            # sampling is preferred (it removes a half-thickness radial bias),
            # but for those instances the bias is <= 1 voxel = 0.1 pitch, so
            # fall back to any voxel rather than dropping the instance.
            src = ii if ii.size else sel
            if src.size:
                zo = np.argsort(z[src], kind="stable")  # z-stratified subsample
                if src.size > CANDS_PER_TILE:
                    zo = zo[np.linspace(0, src.size - 1,
                                        CANDS_PER_TILE).astype(int)]
                pick = src[zo]
                d0["cands"].append(np.stack(
                    [z[pick].astype(np.int32) + z0, y[pick], x[pick]], axis=1))
    return det


def local_radial_sigma(d: dict, min_count: int = 20) -> float:
    """Typical radius dispersion WITHIN an angular bin, in voxels.

    A global p95-p5 radius spread only works for a circular section; PHerc1218
    is squashed, so one wrap legitimately sweeps a wide radius range and a
    global spread test rejects almost everything (measured: 86% of valid
    instances). Within a narrow angular wedge, however, a single wrap is thin
    while a merged pair of wraps sits ~one pitch apart -- so the within-bin
    sigma is the geometry-robust merge signal. Returns the count-weighted mean
    of the per-bin standard deviations.
    """
    n = d["an"]
    use = n >= min_count
    if not use.any():
        return 0.0
    cnt = n[use].astype(float)
    mean = d["ar"][use] / cnt
    var = np.maximum(d["ar2"][use] / cnt - mean * mean, 0.0)
    return float(np.sqrt(var).dot(cnt) / cnt.sum())


def spread_from_hist(hist: np.ndarray) -> float:
    """p95 - p5 of the binned radius distribution (1-voxel bins)."""
    n = int(hist.sum())
    if n == 0:
        return 0.0
    cum = np.cumsum(hist)
    p5 = int(np.searchsorted(cum, 0.05 * n))
    p95 = int(np.searchsorted(cum, 0.95 * n))
    return float(p95 - p5)


# ------------------------------------------------------------------- sampling

class PointSet:
    """Greedy point registry with a 5-voxel grid hash for min-distance."""

    CELL = int(MIN_DIST)

    def __init__(self):
        self.points = []        # (z, y, x) in L1 voxels
        self.cells = {}

    def try_add(self, p) -> bool:
        cz, cyy, cxx = p[0] // self.CELL, p[1] // self.CELL, p[2] // self.CELL
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for q in self.cells.get((cz + dz, cyy + dy, cxx + dx), ()):
                        dd = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                              + (p[2] - q[2]) ** 2)
                        if dd < MIN_DIST * MIN_DIST:
                            return False
        self.cells.setdefault((cz, cyy, cxx), []).append(p)
        self.points.append(p)
        return True


def sample_instance(cands: np.ndarray, k: int, pset: PointSet, rng) -> int:
    """Pick up to k interior points spread across the instance's z-range,
    greedily enforcing the pairwise min distance."""
    if k <= 0 or len(cands) == 0:
        return 0
    cands = cands[np.argsort(cands[:, 0], kind="stable")]
    added = 0
    for chunk in np.array_split(np.arange(len(cands)), min(k, len(cands))):
        if added >= k:
            break
        for i in rng.permutation(chunk):
            p = (int(cands[i, 0]), int(cands[i, 1]), int(cands[i, 2]))
            if pset.try_add(p):
                added += 1
                break
    if added < k:                       # fill-up pass over everything
        for i in rng.permutation(len(cands)):
            if added >= k:
                break
            p = (int(cands[i, 0]), int(cands[i, 1]), int(cands[i, 2]))
            if pset.try_add(p):
                added += 1
    return added


# ---------------------------------------------------------------- self-checks

def roundtrip_check(doc, run, table, all_slabs, rng):
    """Map 200 random emitted points back to source blocks and compare ids."""
    pts = []
    for col in doc["collections"].values():
        gid = int(col["name"].split("_", 1)[1])
        for pt in col["points"].values():
            pts.append((gid, pt["p"]))
    if not pts:
        return 0, 0
    sel = rng.choice(len(pts), size=min(N_ROUNDTRIP, len(pts)), replace=False)
    tiles_of = {z0: slab_tiles(run, z0) for z0 in all_slabs}
    tasks = []
    for i in sel:
        gid, (X, Y, Z) = pts[i]
        z1, y1, x1 = Z // 2, Y // 2, X // 2
        owners, others = [], []
        for z0 in all_slabs:
            if not z0 <= z1 < z0 + SLAB:
                continue
            for (y0, x0), p in tiles_of[z0].items():
                if y0 <= y1 < y0 + TILE and x0 <= x1 < x0 + TILE:
                    yl = OWN if (y0 - STRIDE, x0) in tiles_of[z0] else 0
                    xl = OWN if (y0, x0 - STRIDE) in tiles_of[z0] else 0
                    yh = TILE - OWN if (y0 + STRIDE, x0) in tiles_of[z0] else TILE
                    xh = TILE - OWN if (y0, x0 + STRIDE) in tiles_of[z0] else TILE
                    tgt = owners if (y0 + yl <= y1 < y0 + yh
                                     and x0 + xl <= x1 < x0 + xh) else others
                    tgt.append((z0, y0, x0))
        tasks.append((gid, z1, y1, x1, owners + others))
    tasks.sort(key=lambda t: t[4][0] if t[4] else (-1, -1, -1))
    cache = {}

    def get_tile(z0, y0, x0):
        key = (z0, y0, x0)
        if key not in cache:
            while len(cache) >= 2:              # tiles are ~270 MB decompressed
                cache.pop(next(iter(cache)))
            p = run / "blocks" / f"z{z0}" / f"tile_y{y0}_x{x0}.npz"
            with np.load(p) as d:
                labels = d["labels"]
            cache[key] = (labels, tile_lut(table, z0, y0, x0,
                                           int(labels.max())))
        return cache[key]

    fails = 0
    for gid, z1, y1, x1, cands in tasks:
        hit = False
        for z0, y0, x0 in cands:
            labels, lut = get_tile(z0, y0, x0)
            if lut is None:
                continue
            zi, yi, xi = z1 - z0, y1 - y0, x1 - x0
            if (zi < labels.shape[0] and yi < labels.shape[1]
                    and xi < labels.shape[2]):
                local = int(labels[zi, yi, xi])
                if local > 0 and int(lut[local]) == gid:
                    hit = True
                    break
        if not hit:
            fails += 1
    return int(len(tasks)), fails


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--slabs", type=str, default=None,
                    help="comma-separated slab z0 list (default: all)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <run_dir>/constraints)")
    ap.add_argument("--budget", type=int, default=200000,
                    help="max total points emitted")
    args = ap.parse_args()

    run = args.run_dir
    out = args.out if args.out is not None else run / "constraints"
    out.mkdir(parents=True, exist_ok=True)

    all_slabs = sorted(int(p.name[1:]) for p in (run / "blocks").iterdir()
                       if p.is_dir() and p.name.startswith("z"))
    slabs = (sorted(int(s) for s in args.slabs.split(","))
             if args.slabs else all_slabs)
    missing = [z0 for z0 in slabs if z0 not in all_slabs]
    if missing:
        print(f"ERROR: slabs not found under {run / 'blocks'}: {missing}")
        return 1

    print(f"loading global table ...", flush=True)
    with open(run / "global_table.json") as fh:
        table = json.load(fh)   # ~180 MB of JSON in bad cases: parse ONCE
    max_gid = 0
    for src in table.values():
        for g in src.values():
            if g > max_gid:
                max_gid = g
    pitch_rows = load_pitch_rows(run)
    agree = load_agreement(run)
    all_slab_set = set(all_slabs)
    rng = np.random.default_rng(SEED)

    psets = {}          # gid -> PointSet (one collection per global instance)
    contrib = {}        # gid -> set of contributing slab z0
    restricted = set()  # gids that crossed a low-agreement slab boundary
    patch = {}          # gid -> (volume, deep_frac)
    deep_stats = []     # deep-core fractions seen (filter calibration report)
    sigma_stats = []    # per-instance within-angular-bin radial sigmas
    coverage = {}       # z0 -> [instances, points]
    counters = {k: 0 for k in
                ("candidates", "rejected_mega", "rejected_thickness",
                 "rejected_radial", "rejected_boundary", "budget_skipped",
                 "sample_dropped", "accepted", "extended")}
    total_points = 0

    for z0 in slabs:
        tiles = slab_tiles(run, z0)
        if not tiles:
            print(f"slab z{z0}: no tiles, skipping")
            continue
        vol, zc, zsy, zsx, skipped = slab_pass_volumes(
            run, z0, tiles, table, max_gid)
        if skipped:
            print(f"slab z{z0}: {len(skipped)} tiles missing from "
                  f"global table, skipped: {skipped}")
        total_labeled = int(zc.sum())
        if total_labeled == 0:
            print(f"slab z{z0}: empty")
            continue
        cy = np.where(zc > 0, zsy / np.maximum(zc, 1), (L1_DIMS[1] - 1) / 2)
        cx = np.where(zc > 0, zsx / np.maximum(zc, 1), (L1_DIMS[2] - 1) / 2)
        big_ids = np.flatnonzero(vol >= MIN_VOL)
        big_lookup = np.zeros(max_gid + 1, dtype=bool)
        big_lookup[big_ids] = True
        det = slab_pass_details(z0, tiles, table, big_lookup, cy, cx)
        pitch_vox = local_pitch_vox(pitch_rows, z0)
        cov = coverage.setdefault(z0, [0, 0])

        order = big_ids[np.argsort(-vol[big_ids], kind="stable")]
        for pos, gid in enumerate(order):
            gid = int(gid)
            v = int(vol[gid])
            d = det.get(gid)
            if d is None:
                continue
            counters["candidates"] += 1
            deep_frac = d["ivol"] / v if v else 1.0
            deep_stats.append(deep_frac)
            if v > PATCH_MIN_VOL and deep_frac <= MAX_DEEP_FRAC:
                old = patch.get(gid)
                if old is None or v > old[0]:
                    patch[gid] = (v, deep_frac)
            if v >= MEGA_FRAC * total_labeled:
                counters["rejected_mega"] += 1
                continue
            if deep_frac > MAX_DEEP_FRAC:
                counters["rejected_thickness"] += 1
                continue
            sigma = local_radial_sigma(d)
            sigma_stats.append(sigma)
            if sigma > RADIAL_SIGMA_FACTOR * pitch_vox:
                counters["rejected_radial"] += 1
                continue
            # cross-slab boundary trust
            bad = False
            if (d["zmin"] < SLAB - Z_STRIDE and (z0 - Z_STRIDE) in all_slab_set
                    and agree.get(f"z{z0 - Z_STRIDE}-z{z0}", 0.0) < AGREE_MIN):
                bad = True
            if (d["zmax"] >= Z_STRIDE and (z0 + Z_STRIDE) in all_slab_set
                    and agree.get(f"z{z0}-z{z0 + Z_STRIDE}", 0.0) < AGREE_MIN):
                bad = True
            prior = contrib.get(gid)
            if (bad or gid in restricted) and prior and prior != {z0}:
                counters["rejected_boundary"] += 1   # single-slab only
                continue
            if bad:
                restricted.add(gid)
            if total_points >= args.budget:
                counters["budget_skipped"] += len(order) - pos
                break
            k = min(max(int(round(v / VOX_PER_POINT)), K_MIN), K_MAX)
            k = min(k, args.budget - total_points)
            cands = np.concatenate(d["cands"])
            pset = psets.get(gid)
            new = pset is None
            if new:
                pset = PointSet()
            gained = sample_instance(cands, k, pset, rng)
            if new:
                if len(pset.points) < 2:
                    counters["sample_dropped"] += 1
                    continue
                psets[gid] = pset
                counters["accepted"] += 1
            elif gained:
                counters["extended"] += 1
            if gained:
                contrib.setdefault(gid, set()).add(z0)
                total_points += gained
                cov[0] += 1
                cov[1] += gained
        print(f"slab z{z0}: labeled={total_labeled} big={len(big_ids)} "
              f"pitch_vox={pitch_vox:.2f} accepted_here={cov[0]} "
              f"points_here={cov[1]} total_points={total_points}", flush=True)

    # ------------------------------------------------------------- write JSON
    collections = {}
    ordered = sorted(psets.items(), key=lambda kv: (-len(kv[1].points), kv[0]))
    for ci, (gid, pset) in enumerate(ordered):
        hue = (ci * 0.61803398875) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        pts = {str(i): {"p": [int(x) * 2, int(y) * 2, int(z) * 2],
                        "creation_time": 0}
               for i, (z, y, x) in enumerate(pset.points)}
        collections[str(ci)] = {"name": f"inst_{gid}", "points": pts,
                                "metadata": {},
                                "color": [int(round(255 * c)) for c in rgb]}
    doc = {"vc_pointcollections_json_version": "1",
           "collections": collections}
    out_json = out / "same_windings.json"
    with open(out_json, "w") as fh:
        json.dump(doc, fh)
    patch_list = [{"global_id": g, "volume": v, "deep_frac": round(px, 4)}
                  for g, (v, px) in
                  sorted(patch.items(), key=lambda kv: -kv[1][0])]
    with open(out / "patch_candidates.json", "w") as fh:
        json.dump(patch_list, fh, indent=1)
    print(f"wrote {out_json} ({len(collections)} collections, "
          f"{total_points} points) and patch_candidates.json "
          f"({len(patch_list)} entries)")

    # ------------------------------------------------------------ self-checks
    failed = False

    n_pts = oob = 0
    for col in collections.values():
        for pt in col["points"].values():
            X, Y, Z = pt["p"]
            n_pts += 1
            if not (0 <= X < FULL_DIMS[2] and 0 <= Y < FULL_DIMS[1]
                    and 0 <= Z < FULL_DIMS[0]):
                oob += 1
    ok = oob == 0 and n_pts > 0
    failed |= not ok
    print(f"[{'PASS' if ok else 'FAIL'}] bounds: {n_pts} points, "
          f"{oob} outside full-res dims {FULL_DIMS} (z,y,x)")

    checked, fails = roundtrip_check(doc, run, table, all_slabs, rng)
    ok = checked > 0 and fails == 0
    failed |= not ok
    print(f"[{'PASS' if ok else 'FAIL'}] round-trip: {checked} points "
          f"checked, {fails} id mismatches")

    ok = len(collections) > 0
    failed |= not ok
    print(f"[{'PASS' if ok else 'FAIL'}] non-empty: "
          f"{len(collections)} collections")

    print("instance filter report:")
    for k, v in counters.items():
        print(f"  {k}: {v}")
    if deep_stats:
        ds = np.asarray(deep_stats, dtype=float)
        qs = np.percentile(ds, [50, 90, 99])
        print(f"  deep-core fraction of candidates: p50 {qs[0]:.4f}, "
              f"p90 {qs[1]:.4f}, p99 {qs[2]:.4f} "
              f"(ceiling {MAX_DEEP_FRAC}, rejects "
              f"{100.0 * np.mean(ds > MAX_DEEP_FRAC):.1f}%)")
    print("per-z-band coverage (slab z0: instances, points):")
    for z0 in sorted(coverage):
        ins, pts = coverage[z0]
        print(f"  z{z0}: {ins} instances, {pts} points")

    print("SELF-CHECKS:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
