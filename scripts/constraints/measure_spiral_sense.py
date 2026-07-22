"""Measure a scroll's spiral winding sense from stitched instance labels.

Method (multi-turn tracking): follow one instance continuously in theta around
the umbilicus for more than a full revolution, with a tight radius-continuity
gate so the track cannot hop turns. Then compare r(t) against r(t - 2*pi) at
the SAME theta mod 2*pi: any cross-section shape term (the squashed-annulus
ellipticity that dominates PHerc1218) cancels exactly, leaving the spiral
advance of +-(local pitch) per turn.

Sign convention matches fit_spiral's `spiral_outward_sense`: with
theta = atan2(y - u_y, x - u_x) in array coordinates, a positive advance
(radius grows with increasing theta) means 'CW'; negative means 'ACW'.

Two failure modes this method surfaces honestly:
  - naive dr/dtheta on a squashed scroll measures the ellipticity, not the
    spiral (30x larger on PHerc1218) - do not use it;
  - instances whose 2-turn advance is ~0 are CLOSED FUSION LOOPS: two adjacent
    turns merged into an annulus by a labeling error. The script reports them
    separately - they are a merge-QA signal in their own right.

Usage:
  python scripts/constraints/measure_spiral_sense.py RUN_DIR PACK_DIR Z0

  RUN_DIR   stitched-label run dir; PACK_DIR holds umbilicus.json;
  Z0        L1 slab start whose 8 sampled slices are analysed.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pitch_qa  # slab_canvases + Z_LOCALS

STEP = 0.02                    # rad per tracking step
TRACK_GATE = 6.0               # L1 vox: max per-step radial jump
MAX_SKIP = 15                  # consecutive missing steps allowed (0.3 rad)
MIN_PIX = 1500
LOOP_THRESHOLD = 2.0           # |advance| below this => closed fusion loop
GRID = 3797


def main() -> None:
    run = Path(sys.argv[1])
    pack = Path(sys.argv[2])
    z0 = int(sys.argv[3])

    umb = json.load(open(pack / "umbilicus.json"))["control_points"]
    umb = (np.array([[p["x"], p["y"], p["z"]] for p in umb], dtype=float)
           if isinstance(umb[0], dict) else np.array(umb, dtype=float))

    def umb_yx_l1(z_l1: float):
        i = np.argmin(np.abs(umb[:, 2] - 2.0 * z_l1))
        return umb[i, 1] / 2.0, umb[i, 0] / 2.0

    print("reconstructing slab canvases...", flush=True)
    canvases = pitch_qa.slab_canvases(run, z0)
    assert canvases is not None, f"slab z{z0} not found in {run}"
    rr = np.arange(0, 2200, 1.0)

    def own_crossings(canvas, uy, ux, th, gid):
        py = np.clip((uy + rr * np.sin(th)).astype(int), 0, GRID - 1)
        px = np.clip((ux + rr * np.cos(th)).astype(int), 0, GRID - 1)
        ids = canvas[py, px]
        nz = ids == gid
        if not nz.any():
            return np.empty(0)
        idx = np.flatnonzero(nz)
        breaks = np.flatnonzero(np.diff(idx) > 2)
        starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        ends = np.concatenate([idx[breaks], [idx[-1]]])
        return (starts + ends) / 2.0

    counts: dict = {}
    for zl, c in canvases.items():
        ids, n = np.unique(c[c > 0], return_counts=True)
        for i, m in zip(ids, n):
            counts.setdefault(int(i), []).append(int(m))
    cands = sorted([i for i, ns in counts.items() if max(ns) > MIN_PIX],
                   key=lambda i: -max(counts[i]))[:40]

    spirals, loops = [], []
    for gid in cands:
        for zl in pitch_qa.Z_LOCALS:
            canvas = canvases[zl]
            if int((canvas == gid).sum()) < MIN_PIX:
                continue
            uy, ux = umb_yx_l1(z0 + zl)
            ys, xs = np.nonzero(canvas == gid)
            ths = np.arctan2(ys - uy, xs - ux)
            th0 = float(np.arctan2(np.sin(ths).mean(), np.cos(ths).mean()))
            seeds = own_crossings(canvas, uy, ux, th0, gid)
            if len(seeds) == 0:
                continue
            for direction in (+1, -1):
                for r_start in seeds:
                    t, r_track = 0.0, float(r_start)
                    ts, rs_ = [0.0], [r_track]
                    miss = 0
                    while abs(t) < 4 * np.pi and miss <= MAX_SKIP:
                        t += direction * STEP
                        cent = own_crossings(canvas, uy, ux, th0 + t, gid)
                        if len(cent) == 0:
                            miss += 1
                            continue
                        j = np.argmin(np.abs(cent - r_track))
                        if abs(cent[j] - r_track) > TRACK_GATE:
                            miss += 1
                            continue
                        miss = 0
                        r_track = float(cent[j])
                        ts.append(t)
                        rs_.append(r_track)
                    ta = np.abs(np.array(ts))
                    ra = np.array(rs_)
                    if ta.max() < 2 * np.pi * 1.03:
                        continue
                    sel = ta > 2 * np.pi
                    if sel.sum() < 10:
                        continue
                    order = np.argsort(ta)
                    r_interp = np.interp(ta[sel] - 2 * np.pi, ta[order],
                                         ra[order])
                    adv = float(np.median(ra[sel] - r_interp)) * direction
                    rec = (gid, z0 + zl, direction, adv, int(sel.sum()))
                    (loops if abs(adv) < LOOP_THRESHOLD else spirals).append(rec)
                    break

    print(f"informative spiral tracks: {len(spirals)}")
    for gid, z, d, adv, n in spirals:
        print(f"  id{gid} z{z} dir{d:+d} advance {adv:+.1f} L1 vox/turn "
              f"(n={n})")
    print(f"closed fusion loops (|advance| < {LOOP_THRESHOLD}): {len(loops)}")
    for gid, z, d, adv, n in loops:
        print(f"  id{gid} z{z} dir{d:+d} advance {adv:+.1f} - two turns "
              f"merged into an annulus")
    if spirals:
        med = float(np.median([s[3] for s in spirals]))
        print(f"VERDICT: median advance {med:+.1f} L1 vox/turn -> "
              f"spiral_outward_sense = '{'CW' if med > 0 else 'ACW'}'")
    else:
        print("VERDICT: no informative tracks on this slab - try another z0")


if __name__ == "__main__":
    main()
