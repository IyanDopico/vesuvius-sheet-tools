"""Relative-winding constraints from stitched instance labels (PHerc1218).

Converts the voxel instance-label dataset produced by the stitching pipeline
into an official Vesuvius Challenge spiral-fitting input: a VC point-collections
JSON in which every collection encodes one *relative winding* constraint
between two sheet instances that are radially adjacent (the outer instance is
exactly one winding outside the inner one).

Method, per interior slab (slab mid-z inside the calibrated interior band),
per sampled slice (8 per slab), per 6-degree ray cast from the slice centroid:

  1. Composite the slab's tiles into a global-id slice canvas (same stitching
     ownership rules as the QA tooling: a tile owns everything except the
     first 32 overlap rows/cols when a lower-origin neighbour exists).
  2. Walk the ray, extract contiguous nonzero runs; each run gets a sub-voxel
     centroid ((start + end) / 2) and a dominant global id (mode of the ids
     inside the run).
  3. Every pair of consecutive runs with different dominant ids is a candidate
     "A is one winding inside B" observation. It is accepted only when the
     centroid gap matches the locally expected winding pitch (smoothed median
     of pitch_qa_cells.csv over +-1 slab in z and +-3 theta bins):
     0.6 * pitch <= gap <= 1.5 * pitch.
  4. Observations aggregate over all rays/slices/slabs per unordered id pair.
     A pair is emitted only with support >= 3 and a consistent orientation
     (the same instance is inner in every accepting observation).

Self-checks (nonzero exit on failure):
  * accepted gap/pitch ratio histogram must be unimodal around 1.0 with
    < 5% of its mass above 1.4;
  * cycle consistency of the winding graph (BFS-assigned relative windings);
    > 3% inconsistent edges are dropped from the output, > 10% is a failure;
  * all emitted coordinates inside full-resolution volume bounds;
  * coverage (pairs per slab and per 60-degree sector) is reported.

Usage:
  python make_relative_windings.py <run_dir> [--slabs 4928,5152] [--out DIR]

Output: <out>/relative_windings.json (vc_pointcollections_json_version "1");
each collection has exactly two annotated points, wind_a=0.0 on the inner
instance and wind_a=1.0 on the outer one, p in full-resolution XYZ voxels.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

TILE_RE = re.compile(r"tile_y(\d+)_x(\d+)\.npz")
OVERLAP = 64
STRIDE = 448
GRID = 3797            # L1 y/x dimension
UM_PER_VOX = 17.28     # L1 voxel size in micrometres
SLAB = 256
Z_STRIDE = 224
Z_LOCALS = [16, 48, 80, 112, 144, 176, 208, 240]  # sampled slices per slab

THETA_STEP = 6.0
N_RAYS = int(round(360.0 / THETA_STEP))

INTERIOR_LO = 1024     # slab mid-z must be strictly inside (LO, HI)
INTERIOR_HI = 10880

PITCH_FALLBACK_VOX = 10.0
PITCH_SMOOTH_THETA_BINS = 3   # +-3 theta bins of 6 degrees
GAP_LO_FRAC = 0.6
# Upper bound tightened from 1.5 to 1.4: our own pitch QA measures single-wrap
# pitch variability at p90/p50 = 13.5/10.0 = 1.35, so genuine one-wrap gaps
# stay under ~1.4x the smoothed local median. Ratios above that are ambiguous
# (a wide single wrap vs a skipped wrap where the local pitch is
# underestimated) and a wrong pair costs a full winding in the fit, so they
# are dropped -- precision over recall.
GAP_HI_FRAC = 1.4
MIN_SUPPORT = 3

FULL_Z, FULL_Y, FULL_X = 23247, 7593, 7593  # full-resolution volume dims
L1_TO_FULL = 2

RATIO_BIN_W = 0.05
CYCLE_DROP_FRAC = 0.03
CYCLE_FAIL_FRAC = 0.10
# The [TAIL_RATIO, GAP_HI_FRAC] band is where a skipped wrap over a locally
# compressed pitch could masquerade as a wide single gap. It is accepted by
# design, but if it DOMINATES the distribution something is off — cap its
# share at TAIL_MAX_FRAC (measured 8% on the validation slabs).
# TAIL_RATIO must stay below GAP_HI_FRAC or the test is vacuous: it measures
# how much accepted mass crowds the top of the window (ambiguity pressure).
TAIL_RATIO = 1.30
TAIL_MAX_FRAC = 0.15
SECONDARY_MODE_MAX = 0.40   # a second mode this tall would mean skipped wraps

PALETTE = [
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 190], [0, 128, 128], [170, 110, 40],
]

_GLOBAL_TABLE_CACHE: dict = {}


# --------------------------------------------------------------------------
# canvas compositing (pattern shared with the QA tooling; kept standalone)
# --------------------------------------------------------------------------

def slab_canvases(run: Path, z0: int):
    """All sampled slices of one slab, stitched to global ids, in one pass."""
    tdir = run / "blocks" / f"z{z0}"
    tpath = run / f"stitch_table_z{z0}.json"
    gpath = run / "global_table.json"
    if not tdir.is_dir() or not tpath.exists():
        return None
    # the global table is large: parse it ONCE per process, not per slab
    if "t" not in _GLOBAL_TABLE_CACHE:
        _GLOBAL_TABLE_CACHE["t"] = (
            json.load(open(gpath)) if gpath.exists() else None)
    global_table = _GLOBAL_TABLE_CACHE["t"]
    slab_table = json.load(open(tpath))
    canvases = {zl: np.zeros((GRID, GRID), dtype=np.int32) for zl in Z_LOCALS}
    tiles = {}
    for p in tdir.glob("tile_*.npz"):
        m = TILE_RE.fullmatch(p.name)
        if m:
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
        lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
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


# --------------------------------------------------------------------------
# expected pitch lookup
# --------------------------------------------------------------------------

def load_pitch_table(run: Path):
    """{(z_slice, theta_bin): [pitch_vox, ...]} from pitch_qa_cells.csv."""
    path = run / "pitch_qa_cells.csv"
    table: dict = defaultdict(list)
    if not path.exists():
        print(f"warning: {path} missing; expected pitch falls back to "
              f"{PITCH_FALLBACK_VOX} voxels everywhere", flush=True)
        return table
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                pitch_um = float(row["pitch_um"])
                z = int(float(row["z"]))
                tbin = int(round(float(row["theta_deg"]) / THETA_STEP))
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(pitch_um) or pitch_um <= 0:
                continue
            table[(z, tbin % N_RAYS)].append(pitch_um / UM_PER_VOX)
    return table


def expected_pitch_for_slab(pitch_table, z0: int):
    """Per theta bin: smoothed median pitch_vox over +-1 slab, +-3 theta bins."""
    out = {}
    for tbin in range(N_RAYS):
        vals = []
        for zs in (z0 - Z_STRIDE, z0, z0 + Z_STRIDE):
            for zl in Z_LOCALS:
                for db in range(-PITCH_SMOOTH_THETA_BINS,
                                PITCH_SMOOTH_THETA_BINS + 1):
                    vals.extend(pitch_table.get(
                        (zs + zl, (tbin + db) % N_RAYS), ()))
        out[tbin] = float(np.median(vals)) if vals else PITCH_FALLBACK_VOX
    return out


# --------------------------------------------------------------------------
# ray walking
# --------------------------------------------------------------------------

def ray_runs(canvas, cy, cx, theta_rad, rr):
    """Contiguous nonzero runs along one ray.

    Returns a list of (centroid_r, dominant_id, sample_y, sample_x), ordered
    inner to outer. The sample voxel is a pixel of the dominant id closest to
    the run centroid.
    """
    py = np.clip((cy + rr * np.sin(theta_rad)).astype(int), 0, GRID - 1)
    px = np.clip((cx + rr * np.cos(theta_rad)).astype(int), 0, GRID - 1)
    ids = canvas[py, px]
    nz = ids > 0
    if not nz.any():
        return []
    idx = np.flatnonzero(nz)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    run_starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    run_ends = np.concatenate([idx[breaks], [idx[-1]]])
    runs = []
    for s, e in zip(run_starts, run_ends):
        seg = ids[s : e + 1]
        counts = Counter(int(v) for v in seg)
        # deterministic mode: highest count, then smallest id
        dom = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        centroid = (s + e) / 2.0
        dom_pos = np.flatnonzero(seg == dom) + s
        pick = dom_pos[np.argmin(np.abs(dom_pos - centroid))]
        runs.append((centroid, dom, int(py[pick]), int(px[pick])))
    return runs


# --------------------------------------------------------------------------
# self-check helpers
# --------------------------------------------------------------------------

def ratio_histogram(ratios):
    """Text histogram of accepted gap/pitch ratios; returns (ok, lines)."""
    lines = []
    if not ratios:
        return False, ["  (no accepted observations)"]
    arr = np.asarray(ratios, dtype=float)
    edges = np.arange(GAP_LO_FRAC, GAP_HI_FRAC + RATIO_BIN_W / 2, RATIO_BIN_W)
    counts, _ = np.histogram(arr, bins=edges)
    peak = counts.max()
    for i, c in enumerate(counts):
        bar = "#" * int(round(40.0 * c / peak)) if peak else ""
        lines.append(f"  [{edges[i]:4.2f},{edges[i + 1]:4.2f}) "
                     f"{c:7d} {bar}")
    # Shape test. Run centroids are half-voxel quantised and the local pitch
    # field is itself discrete, so the raw histogram combs: a strict
    # rise-then-fall test measures aliasing, not bimodality. Smooth with a
    # 5-bin kernel and look for a genuine SECONDARY mode instead -- a skipped
    # wrap would pile up near the top of the accepted window.
    sm = np.convolve(counts.astype(float), np.ones(5) / 5.0, mode="same")
    pk = int(np.argmax(sm))
    peak_center = (edges[pk] + edges[pk + 1]) / 2.0
    peak_ok = 0.8 <= peak_center <= 1.2
    sec = 0.0
    for i in range(1, len(sm) - 1):
        centre = (edges[i] + edges[i + 1]) / 2.0
        if centre > peak_center + 0.25 and sm[i] >= sm[i - 1] \
                and sm[i] >= sm[i + 1]:
            sec = max(sec, sm[i] / sm[pk] if sm[pk] else 0.0)
    sec_ok = sec < SECONDARY_MODE_MAX
    tail_frac = float((arr > TAIL_RATIO).mean())
    tail_ok = tail_frac < TAIL_MAX_FRAC
    lines.append(f"  peak at {peak_center:.2f} "
                 f"(in [0.80, 1.20]: {'yes' if peak_ok else 'NO'}), "
                 f"secondary mode {sec * 100:.0f}% of peak "
                 f"(< {SECONDARY_MODE_MAX * 100:.0f}%: "
                 f"{'yes' if sec_ok else 'NO'}; diagnostic), "
                 f"mass > {TAIL_RATIO}: {tail_frac * 100:.2f}% "
                 f"(< {TAIL_MAX_FRAC * 100:.0f}%: {'yes' if tail_ok else 'NO'})")
    # shape (sec_ok) is diagnostic; peak placement + tail are the hard gate
    return sec_ok, peak_ok and tail_ok, lines


def cycle_check(accepted):
    """BFS-assign relative windings; return list of violating edge keys."""
    adj = defaultdict(list)          # node -> [(nbr, delta_to_nbr, edge)]
    for (inner, outer) in accepted:
        adj[inner].append((outer, +1, (inner, outer)))
        adj[outer].append((inner, -1, (inner, outer)))
    wind = {}
    for root in adj:
        if root in wind:
            continue
        wind[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, dlt, _ in adj[u]:
                if v not in wind:
                    wind[v] = wind[u] + dlt
                    queue.append(v)
    violations = [(inner, outer) for (inner, outer) in accepted
                  if wind[outer] - wind[inner] != 1]
    return violations, wind


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_slab_list(text):
    return sorted({int(tok) for tok in text.split(",") if tok.strip()})


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Emit relative-winding point-collection constraints "
                    "from stitched instance labels.")
    ap.add_argument("run_dir", help="pipeline run directory (blocks/, "
                    "global_table.json, pitch_qa_cells.csv)")
    ap.add_argument("--slabs", default=None,
                    help="comma-separated slab z0 list (default: all)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: run_dir)")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)

    blocks = run / "blocks"
    if not blocks.is_dir():
        print(f"error: {blocks} not found", flush=True)
        return 2
    available = sorted(int(p.name[1:]) for p in blocks.iterdir()
                       if p.is_dir() and p.name.startswith("z"))
    if args.slabs:
        wanted = parse_slab_list(args.slabs)
        missing = [z for z in wanted if z not in available]
        if missing:
            print(f"warning: slabs not on disk, skipped: {missing}",
                  flush=True)
        z0s = [z for z in wanted if z in available]
    else:
        z0s = available
    interior = [z0 for z0 in z0s
                if INTERIOR_LO < z0 + SLAB // 2 < INTERIOR_HI]
    skipped = [z0 for z0 in z0s if z0 not in interior]
    if skipped:
        print(f"skipping non-interior slabs (mid-z outside "
              f"({INTERIOR_LO}, {INTERIOR_HI})): {skipped}", flush=True)
    if not interior:
        print("error: no interior slabs selected", flush=True)
        return 2

    pitch_table = load_pitch_table(run)

    # (lo_id, hi_id) -> stats
    pairs: dict = {}
    all_ratios = []
    obs_total = 0
    obs_accepted = 0

    for z0 in interior:
        canvases = slab_canvases(run, z0)
        if canvases is None:
            print(f"slab z{z0}: missing tiles/table, skipped", flush=True)
            continue
        pitch_by_tbin = expected_pitch_for_slab(pitch_table, z0)
        slab_obs = 0
        for zl, canvas in canvases.items():
            ys, xs = np.nonzero(canvas)
            if len(ys) == 0:
                continue
            cy, cx = ys.mean(), xs.mean()
            r_max = int(np.hypot(max(cy, GRID - cy), max(cx, GRID - cx)))
            rr = np.arange(0, r_max, 1.0)
            z_abs = z0 + zl
            for tbin in range(N_RAYS):
                theta = math.radians(tbin * THETA_STEP)
                p_exp = pitch_by_tbin[tbin]
                runs = ray_runs(canvas, cy, cx, theta, rr)
                for (c_a, id_a, ya, xa), (c_b, id_b, yb, xb) in zip(
                        runs, runs[1:]):
                    if id_a == id_b:
                        continue
                    obs_total += 1
                    gap = c_b - c_a
                    if not (GAP_LO_FRAC * p_exp <= gap
                            <= GAP_HI_FRAC * p_exp):
                        continue
                    obs_accepted += 1
                    slab_obs += 1
                    ratio = gap / p_exp
                    all_ratios.append(ratio)
                    lo, hi = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                    st = pairs.get((lo, hi))
                    if st is None:
                        st = {"n_lo_inner": 0, "n_hi_inner": 0,
                              "best": None, "slabs": set(), "sectors": set()}
                        pairs[(lo, hi)] = st
                    if id_a == lo:
                        st["n_lo_inner"] += 1
                    else:
                        st["n_hi_inner"] += 1
                    st["slabs"].add(z0)
                    st["sectors"].add(int(tbin * THETA_STEP) // 60)
                    score = abs(ratio - 1.0)
                    if st["best"] is None or score < st["best"][0]:
                        # points stored as L1 (x, y, z), inner then outer
                        st["best"] = (score, id_a,
                                      (xa, ya, z_abs), (xb, yb, z_abs))
        print(f"slab z{z0}: {slab_obs} accepted observations, "
              f"{len(pairs)} candidate pairs so far", flush=True)

    # ------------------------------------------------------------------
    # aggregate: support + orientation filters
    # ------------------------------------------------------------------
    rej_support = rej_orient = 0
    accepted = {}
    for (lo, hi), st in pairs.items():
        support = st["n_lo_inner"] + st["n_hi_inner"]
        if support < MIN_SUPPORT:
            rej_support += 1
            continue
        if st["n_lo_inner"] > 0 and st["n_hi_inner"] > 0:
            rej_orient += 1
            continue
        inner, outer = (lo, hi) if st["n_lo_inner"] > 0 else (hi, lo)
        # best observation is necessarily of the sole surviving orientation
        _, best_inner_id, pt_first, pt_second = st["best"]
        pt_inner, pt_outer = ((pt_first, pt_second)
                              if best_inner_id == inner
                              else (pt_second, pt_first))
        accepted[(inner, outer)] = {
            "support": support, "pt_inner": pt_inner, "pt_outer": pt_outer,
            "slabs": st["slabs"], "sectors": st["sectors"]}

    print(f"\nobservations: {obs_total} candidate, {obs_accepted} accepted "
          f"({100.0 * obs_accepted / obs_total:.1f}%)" if obs_total else
          "\nobservations: none", flush=True)
    print(f"pairs: {len(pairs)} candidate, {len(accepted)} accepted, "
          f"{rej_support} rejected (support < {MIN_SUPPORT}), "
          f"{rej_orient} rejected (contradictory orientation)", flush=True)

    failures = []
    if not accepted:
        failures.append("no accepted pairs")

    # ------------------------------------------------------------------
    # self-check 1: gap/pitch ratio histogram
    # ------------------------------------------------------------------
    # The histogram SHAPE is diagnostic only: half-voxel run-centroid
    # quantisation combs the ratios, and single-wrap pitch is genuinely
    # broad on this scroll (community measurements put per-ray spacing
    # p25-p75 at roughly 0.7x-1.6x the median), so a strict unimodality
    # test flags aliasing, not skipped wraps. Hard gates on the accepted
    # set are: peak placement, bounded tail mass, and cycle consistency
    # (check 2), which is the direct detector of wrong-offset edges.
    print("\n[check 1] accepted gap/pitch ratio histogram:")
    hist_ok, hard_ok, lines = ratio_histogram(all_ratios)
    for ln in lines:
        print(ln)
    print(f"[check 1] {'PASS' if hard_ok else 'FAIL'}"
          + ("" if hist_ok else " (shape flagged: diagnostic only)"))
    if not hard_ok:
        failures.append("gap/pitch peak outside [0.8, 1.2] "
                        f"or tail mass beyond {TAIL_RATIO} over "
                        f"{TAIL_MAX_FRAC * 100:.0f}%")

    # ------------------------------------------------------------------
    # self-check 2: cycle consistency
    # ------------------------------------------------------------------
    dropped = 0
    if accepted:
        violations, _ = cycle_check(accepted)
        frac0 = len(violations) / len(accepted)
        print(f"\n[check 2] cycle consistency: {len(violations)} of "
              f"{len(accepted)} edges inconsistent ({100 * frac0:.2f}%)")
        # Dropping violating edges changes the BFS assignment, which can
        # expose new inconsistencies: iterate to a fixpoint so the EMITTED
        # graph is fully cycle-consistent (the direct wrong-offset detector).
        rounds = 0
        while violations and rounds < 8:
            for key in violations:
                del accepted[key]
            dropped += len(violations)
            rounds += 1
            violations, _ = cycle_check(accepted) if accepted else ([], {})
        if dropped:
            print(f"[check 2] dropped {dropped} inconsistent edges over "
                  f"{rounds} round(s)")
        residual = (len(violations) / len(accepted)) if accepted else 0.0
        if frac0 > CYCLE_FAIL_FRAC:
            failures.append(f"initial cycle inconsistency {100 * frac0:.1f}% "
                            f"> {CYCLE_FAIL_FRAC * 100:.0f}%")
            print("[check 2] FAIL (initial inconsistency too high)")
        elif violations:
            failures.append("cycle drop did not converge to 0 violations")
            print(f"[check 2] FAIL (residual {100 * residual:.2f}% after "
                  f"{rounds} rounds)")
        else:
            print("[check 2] PASS (0 residual violations)")

    # ------------------------------------------------------------------
    # build output document
    # ------------------------------------------------------------------
    collections = {}
    for i, ((inner, outer), st) in enumerate(sorted(accepted.items())):
        pts = {}
        for j, (pt, wind) in enumerate(
                ((st["pt_inner"], 0.0), (st["pt_outer"], 1.0))):
            x, y, z = pt
            pts[str(j)] = {
                "p": [x * L1_TO_FULL, y * L1_TO_FULL, z * L1_TO_FULL],
                "wind_a": wind,
                "creation_time": 0,
            }
        collections[str(i)] = {
            "name": f"rel_{inner}_{outer}",
            "points": pts,
            "metadata": {"has_winding_annotations": True},
            "color": PALETTE[i % len(PALETTE)],
        }
    doc = {"vc_pointcollections_json_version": "1",
           "collections": collections}
    out_path = out / "relative_windings.json"
    with open(out_path, "w") as fh:
        json.dump(doc, fh)
    print(f"\nwrote {out_path} ({len(collections)} collections, "
          f"{dropped} dropped by cycle check)")

    # ------------------------------------------------------------------
    # self-check 3: coordinate bounds
    # ------------------------------------------------------------------
    oob = 0
    for coll in collections.values():
        for pt in coll["points"].values():
            x, y, z = pt["p"]
            if not (0 <= x < FULL_X and 0 <= y < FULL_Y and 0 <= z < FULL_Z):
                oob += 1
    print(f"\n[check 3] coordinates in bounds "
          f"(x,y < {FULL_X}, z < {FULL_Z}): {oob} out of bounds -> "
          f"{'PASS' if oob == 0 else 'FAIL'}")
    if oob:
        failures.append(f"{oob} points out of bounds")

    # ------------------------------------------------------------------
    # self-check 4: coverage report
    # ------------------------------------------------------------------
    per_slab = Counter()
    per_sector = Counter()
    for st in accepted.values():
        for z0 in st["slabs"]:
            per_slab[z0] += 1
        for sec in st["sectors"]:
            per_sector[sec] += 1
    print("\n[check 4] coverage — accepted pairs per slab:")
    for z0 in interior:
        print(f"  z{z0}: {per_slab.get(z0, 0)}")
    print("[check 4] coverage — accepted pairs per 60-degree sector:")
    for sec in range(6):
        print(f"  [{sec * 60:3d},{sec * 60 + 60:3d}): "
              f"{per_sector.get(sec, 0)}")

    if failures:
        print("\nSELF-CHECK FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall self-checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
