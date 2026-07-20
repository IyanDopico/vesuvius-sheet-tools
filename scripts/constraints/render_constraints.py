"""Render spiral-fit constraint points over stitched PHerc1218 label slices.

QA visualizer for the Phase 4a constraint files (point-collection JSON v1
and umbilicus control points).  For each selected slab the stitched
global-id mid slice (z_local=112) is drawn as a dim grayscale background
(labels > 0 -> gray, background black) and every constraint point whose z
falls within +/-16 L1 slices of it is overlaid:

  * same-winding collections (points without ``wind_a``): one color per
    collection, cycled from a 20-color palette
  * annotated points (``wind_a`` present): diamonds colored by value
    (0 cyan, 1 orange, other magenta); the two points of a pair
    collection are joined by a thin line
  * the umbilicus control point at that z: white cross

Constraint coordinates are FULL-RES voxels (``p = [x, y, z]`` for
collections, ``{z, y, x}`` for the umbilicus); the label volume is L1
(half resolution), so everything is divided by 2 for display.

Usage:
    python scripts/constraints/render_constraints.py output/scroll_run \
        --files coll_a.json,coll_b.json [--umbilicus umbilicus.json] \
        [--slabs 4928,5152] [--out qa_dir] [--zstep 1]

Outputs constraints_z{Z0}.png (~1900 px wide) per slab and prints per-slab
counts of rendered points per file.  Missing or malformed inputs are
reported and skipped, never fatal.
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
from matplotlib import colors as mcolors

TILE_RE = re.compile(r"tile_y(\d+)_x(\d+)\.npz")
OVERLAP = 64
STRIDE = 448
GRID = 3797
MID_Z = 112  # z_local of the rendered slice inside each 256-deep slab
Z_WINDOW = 16  # points within +/- this many L1 slices are drawn
BG_GRAY = 90  # labels > 0 -> this gray level (0..255)

PC_VERSION = "1"
PALETTE = [mcolors.to_hex(c)
           for c in plt.get_cmap("tab20").colors]  # 20 distinct colors
WIND_COLORS = {0: "#00e5ff", 1: "#ff9c1a"}  # cyan / orange
WIND_OTHER = "#ff2fd6"  # magenta

_GLOBAL_TABLE_CACHE: dict = {}


def slab_mid_canvas(run: Path, z0: int) -> np.ndarray | None:
    """Global-id mid slice (z_local=112) of one slab, stitched to GRID^2."""
    tdir = run / "blocks" / f"z{z0}"
    tpath = run / f"stitch_table_z{z0}.json"
    gpath = run / "global_table.json"
    if not tdir.is_dir():
        return None
    # the global table is ~180 MB of JSON: parse it ONCE per process
    if "t" not in _GLOBAL_TABLE_CACHE:
        _GLOBAL_TABLE_CACHE["t"] = (
            json.load(open(gpath)) if gpath.exists() else None)
    global_table = _GLOBAL_TABLE_CACHE["t"]
    slab_table = json.load(open(tpath)) if tpath.exists() else {}
    canvas = np.zeros((GRID, GRID), dtype=np.int64)
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
        lut = np.zeros(labels.max() + 1, dtype=np.int64)
        for k, v in src.items():
            if int(k) <= labels.max():
                lut[int(k)] = v
        # overlap ownership: first 32 rows/cols go to the lower-origin
        # neighbor when one exists (same rule as the stitch pipeline)
        own_y = OVERLAP // 2 if (y0 - STRIDE, x0) in tiles else 0
        own_x = OVERLAP // 2 if (y0, x0 - STRIDE) in tiles else 0
        sl2d = lut[labels[min(MID_Z, labels.shape[0] - 1), own_y:, own_x:]]
        canvas[y0 + own_y : y0 + own_y + sl2d.shape[0],
               x0 + own_x : x0 + own_x + sl2d.shape[1]] = sl2d
    return canvas


def load_pointcollections(path: Path, color_start: int):
    """Parse one point-collection JSON v1 file.

    Returns (collections, n_malformed).  Each collection is a dict with
    ``name``, ``color`` (hex, for its same-winding points) and ``points``:
    a list of (x, y, z, wind_a) tuples in L1 voxels, wind_a None when the
    point carries no annotation.
    """
    data = json.load(open(path))
    ver = data.get("vc_pointcollections_json_version")
    if ver != PC_VERSION:
        print(f"warning: {path.name}: unexpected version {ver!r} "
              f"(expected {PC_VERSION!r}), parsing anyway", flush=True)
    collections = []
    n_bad = 0
    raw_colls = data.get("collections")
    if not isinstance(raw_colls, dict):
        print(f"warning: {path.name}: no 'collections' dict", flush=True)
        return [], 0
    for i, (cid, coll) in enumerate(sorted(raw_colls.items())):
        if not isinstance(coll, dict):
            n_bad += 1
            continue
        points = []
        raw_pts = coll.get("points")
        if not isinstance(raw_pts, dict):
            raw_pts = {}
        for pid, pt in sorted(raw_pts.items()):
            try:
                x, y, z = (float(v) for v in pt["p"])
            except (TypeError, KeyError, ValueError):
                n_bad += 1
                continue
            wind = pt.get("wind_a") if isinstance(pt, dict) else None
            if wind is not None and not isinstance(wind, (int, float)):
                n_bad += 1
                continue
            points.append((x / 2.0, y / 2.0, z / 2.0, wind))  # full-res -> L1
        collections.append({
            "name": str(coll.get("name", cid)),
            "color": PALETTE[(color_start + i) % len(PALETTE)],
            "points": points,
        })
    return collections, n_bad


def load_umbilicus(path: Path) -> np.ndarray | None:
    """(N, 3) array of umbilicus control points as L1 (z, y, x), z-sorted."""
    data = json.load(open(path))
    pts = []
    n_bad = 0
    for cp in data.get("control_points", []):
        try:
            pts.append((float(cp["z"]) / 2.0, float(cp["y"]) / 2.0,
                        float(cp["x"]) / 2.0))
        except (TypeError, KeyError, ValueError):
            n_bad += 1
    if n_bad:
        print(f"warning: {path.name}: {n_bad} malformed control points "
              "skipped", flush=True)
    if not pts:
        return None
    return np.array(sorted(pts), dtype=float)


def umbilicus_at(umb: np.ndarray, z: float) -> tuple[float, float]:
    """(y, x) of the umbilicus at L1 slice z: interpolated, clamped at ends."""
    return (float(np.interp(z, umb[:, 0], umb[:, 1])),
            float(np.interp(z, umb[:, 0], umb[:, 2])))


def render_slab(run: Path, z0: int, files: list, umb, out_dir: Path) -> None:
    canvas = slab_mid_canvas(run, z0)
    if canvas is None:
        print(f"warning: slab z{z0}: no tiles found, skipping", flush=True)
        return
    z_mid = z0 + MID_Z

    fig, ax = plt.subplots(figsize=(19.0, 19.6), dpi=100)
    fig.patch.set_facecolor("black")
    bg = np.where(canvas > 0, BG_GRAY, 0).astype(np.uint8)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    total = 0
    counts = []
    legend_handles = []
    for fname, collections in files:
        n_same = n_annot = 0
        for coll in collections:
            vis = [(x, y, z, w) for x, y, z, w in coll["points"]
                   if abs(z - z_mid) <= Z_WINDOW]
            if not vis:
                continue
            same = [(x, y) for x, y, _, w in vis if w is None]
            annot = [(x, y, w) for x, y, _, w in vis if w is not None]
            if same:
                sx, sy = zip(*same)
                h = ax.scatter(sx, sy, s=90, marker="o", color=coll["color"],
                               edgecolors="black", linewidths=0.5, zorder=3,
                               label=coll["name"])
                if len(legend_handles) < 20:
                    legend_handles.append(h)
                n_same += len(same)
            if annot:
                if len(annot) == 2:  # pair collection: join the two points
                    ax.plot([annot[0][0], annot[1][0]],
                            [annot[0][1], annot[1][1]], color="white",
                            linewidth=1.0, alpha=0.7, zorder=2)
                for x, y, w in annot:
                    ax.scatter([x], [y], s=130, marker="D",
                               color=WIND_COLORS.get(w, WIND_OTHER),
                               edgecolors="black", linewidths=0.5, zorder=4)
                n_annot += len(annot)
        counts.append((fname, n_same, n_annot))
        total += n_same + n_annot

    umb_txt = "no"
    if umb is not None:
        uy, ux = umbilicus_at(umb, z_mid)
        ax.scatter([ux], [uy], s=350, marker="+", color="white",
                   linewidths=1.8, zorder=5)
        umb_txt = f"({ux:.0f}, {uy:.0f})"
    if total == 0:
        ax.text(0.5, 0.04, f"no constraint points within +/-{Z_WINDOW} "
                "slices of this slab", transform=ax.transAxes, color="white",
                fontsize=14, ha="center", zorder=6)

    n_annot_all = sum(a for _, _, a in counts)
    ax.set_xlim(0, GRID)
    ax.set_ylim(GRID, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"slab z{z0} — slice z={z_mid} (L1) — {total} constraint "
                 f"points ({n_annot_all} annotated) — umbilicus {umb_txt}",
                 fontsize=15, color="white", loc="left")
    if legend_handles:
        leg = ax.legend(handles=legend_handles, loc="upper right",
                        fontsize=10, facecolor="black", edgecolor="#555555",
                        framealpha=0.7)
        for t in leg.get_texts():
            t.set_color("white")

    out = out_dir / f"constraints_z{z0}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    for fname, n_same, n_annot in counts:
        print(f"slab z{z0}: {fname}: {n_same + n_annot} points rendered "
              f"({n_same} same-winding, {n_annot} annotated)", flush=True)
    print(f"saved {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay spiral-fit constraint points on stitched "
                    "PHerc1218 label slices (one PNG per slab).")
    ap.add_argument("run_dir", type=Path,
                    help="scroll run dir (contains blocks/ + global_table.json)")
    ap.add_argument("--files", action="append", default=[],
                    help="comma-separated point-collection JSON files "
                         "(repeatable)")
    ap.add_argument("--umbilicus", type=Path,
                    help="umbilicus.json with full-res control_points")
    ap.add_argument("--slabs",
                    help="comma-separated slab z0 list (default: all)")
    ap.add_argument("--out", type=Path,
                    help="output dir (default: run_dir)")
    ap.add_argument("--zstep", type=int, default=1,
                    help="render every Nth available slab when --slabs is "
                         "not given (default 1 = all)")
    args = ap.parse_args()

    run = args.run_dir
    if not (run / "blocks").is_dir():
        sys.exit(f"error: {run / 'blocks'} not found")
    out_dir = args.out if args.out else run
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    color_start = 0
    for group in args.files:
        for token in group.split(","):
            path = Path(token.strip())
            if not path.is_file():
                print(f"warning: {path} not found, skipping", flush=True)
                continue
            try:
                collections, n_bad = load_pointcollections(path, color_start)
            except (json.JSONDecodeError, OSError) as e:
                print(f"warning: {path}: unreadable ({e}), skipping",
                      flush=True)
                continue
            color_start += len(collections)
            n_pts = sum(len(c["points"]) for c in collections)
            msg = (f"loaded {path.name}: {len(collections)} collections, "
                   f"{n_pts} points")
            if n_bad:
                msg += f" ({n_bad} malformed entries skipped)"
            print(msg, flush=True)
            files.append((path.name, collections))
    if not files:
        print("warning: no constraint files loaded — rendering "
              "backgrounds only", flush=True)

    umb = None
    if args.umbilicus:
        if not args.umbilicus.is_file():
            print(f"warning: {args.umbilicus} not found, skipping", flush=True)
        else:
            try:
                umb = load_umbilicus(args.umbilicus)
                if umb is None:
                    print(f"warning: {args.umbilicus.name}: no usable "
                          "control points", flush=True)
            except (json.JSONDecodeError, OSError) as e:
                print(f"warning: {args.umbilicus}: unreadable ({e}), "
                      "skipping", flush=True)

    avail = sorted(int(p.name[1:]) for p in (run / "blocks").iterdir()
                   if p.name.startswith("z") and p.name[1:].isdigit())
    if args.slabs:
        slabs = []
        for token in args.slabs.split(","):
            z0 = int(token.strip())
            if z0 in avail:
                slabs.append(z0)
            else:
                print(f"warning: slab z{z0} not in run, skipping", flush=True)
    else:
        slabs = avail[:: max(1, args.zstep)]

    for z0 in slabs:
        render_slab(run, z0, files, umb, out_dir)


if __name__ == "__main__":
    main()
