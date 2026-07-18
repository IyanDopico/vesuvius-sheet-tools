"""Assemble per-slab outputs of the stitch kernel into one scroll-global table.

The Kaggle kernel labels each z-slab independently (tiles stitched in y/x per
slab). Consecutive slabs overlap Z_OVERLAP slices: this script votes labels in
the facing bands (same global z range seen by both slabs), links them with the
same adaptive mutual-majority rule, and emits a scroll-global instance table.

Input directory layout (accumulated kernel downloads):
    blocks/z{Z0}/tile_y{Y}_x{X}.npz    local int32 labels + origin
    stitch_table_z{Z0}.json            per slab: tile -> local -> slab-global id

Outputs (written next to the input):
    global_table.json       "z{Z0}/y{Y}_x{X}": {local: scroll-global id}
    assembly_metrics.json   per slab-pair links + agreement

Usage:
    python scripts/assemble_scroll.py <run_dir> [slab_size] [z_overlap]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

MIN_VOTES_ABS = 20
DIRECTED_COVER = 0.7  # same rule as the in-slab kernel: +26pp agreement there

SLAB_ID_STEP = 100_000_000  # id space per slab (slab-global ids stay far below)


def adaptive_links(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Mutual-majority links between two co-located label bands (slab-global
    ids). Returns list of (id_a, id_b) plus is used for agreement counting."""
    both = (a > 0) & (b > 0)
    if not both.any():
        return []
    av = a[both].astype(np.int64)
    bv = b[both].astype(np.int64)
    count_a: dict[int, int] = {}
    count_b: dict[int, int] = {}
    for v, c in zip(*np.unique(av, return_counts=True)):
        count_a[int(v)] = int(c)
    for v, c in zip(*np.unique(bv, return_counts=True)):
        count_b[int(v)] = int(c)
    base = int(bv.max()) + 1
    uk, counts = np.unique(av * base + bv, return_counts=True)
    ka = (uk // base).astype(int)
    kb = (uk % base).astype(int)
    best_a: dict[int, tuple[int, int]] = {}
    best_b: dict[int, tuple[int, int]] = {}
    for la, lb, c in zip(ka, kb, counts):
        if c > best_a.get(la, (0, -1))[0]:
            best_a[la] = (int(c), int(lb))
        if c > best_b.get(lb, (0, -1))[0]:
            best_b[lb] = (int(c), int(la))
    links = set()
    # mutual-best links
    for la, (c, lb) in best_a.items():
        if c >= MIN_VOTES_ABS and best_b.get(lb, (0, -1))[1] == la:
            links.add((la, lb))
    # directed coverage links (many-to-one across differing fragmentations)
    for la, (c, lb) in best_a.items():
        if c >= MIN_VOTES_ABS and c >= DIRECTED_COVER * count_a[la]:
            links.add((la, lb))
    for lb, (c, la) in best_b.items():
        if c >= MIN_VOTES_ABS and c >= DIRECTED_COVER * count_b[lb]:
            links.add((la, lb))
    return sorted(links)


def main() -> None:
    run_dir = Path(sys.argv[1])
    slab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    z_overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    slab_dirs = sorted(
        (int(m.group(1)), p)
        for p in (run_dir / "blocks").iterdir()
        if (m := re.fullmatch(r"z(\d+)", p.name))
    )
    tables = {}
    for z0, _ in slab_dirs:
        with open(run_dir / f"stitch_table_z{z0}.json") as fh:
            tables[z0] = json.load(fh)
    print(f"{len(slab_dirs)} slabs: {[z for z, _ in slab_dirs]}")

    # union-find over slab-global ids offset per slab
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    def gid(slab_idx: int, slab_global: int) -> int:
        return (slab_idx + 1) * SLAB_ID_STEP + slab_global

    metrics = {"pairs": {}}
    for i in range(len(slab_dirs) - 1):
        za, dir_a = slab_dirs[i]
        zb, dir_b = slab_dirs[i + 1]
        if za + slab_size - z_overlap != zb:
            print(f"WARNING: slabs z{za} and z{zb} are not contiguous; skipping")
            continue
        links = agree = both = 0
        band_pairs = []
        tiles_b = {p.name: p for p in dir_b.glob("tile_*.npz")}
        for pa in dir_a.glob("tile_*.npz"):
            pb = tiles_b.get(pa.name)
            if pb is None:
                continue
            m = re.fullmatch(r"tile_y(\d+)_x(\d+)\.npz", pa.name)
            tkey = f"y{m.group(1)}_x{m.group(2)}"
            ta = tables[za].get(tkey)
            tb = tables[zb].get(tkey)
            if not ta or not tb:
                continue
            with np.load(pa) as da:
                band_a = da["labels"][-z_overlap:]
            with np.load(pb) as db:
                band_b = db["labels"][:z_overlap]
            # local -> slab-global
            lut_a = np.zeros(band_a.max() + 1, dtype=np.int64)
            for k, v in ta.items():
                if int(k) <= band_a.max():
                    lut_a[int(k)] = v
            lut_b = np.zeros(band_b.max() + 1, dtype=np.int64)
            for k, v in tb.items():
                if int(k) <= band_b.max():
                    lut_b[int(k)] = v
            ga, gb = lut_a[band_a], lut_b[band_b]
            for la, lb in adaptive_links(ga, gb):
                union(gid(i, la), gid(i + 1, lb))
                links += 1
            band_pairs.append((ga, gb))
        for ga, gb in band_pairs:
            m2 = (ga > 0) & (gb > 0)
            if not m2.any():
                continue
            av, bv = ga[m2], gb[m2]
            both += int(m2.sum())
            roots_a = np.fromiter((find(gid(i, int(v))) for v in av), np.int64, len(av))
            roots_b = np.fromiter((find(gid(i + 1, int(v))) for v in bv), np.int64, len(bv))
            agree += int((roots_a == roots_b).sum())
        rate = agree / max(both, 1)
        metrics["pairs"][f"z{za}-z{zb}"] = {
            "links": links, "overlap_voxels": both,
            "agreement_rate": round(rate, 4),
        }
        print(f"z{za} <-> z{zb}: {links} links, agreement {rate:.1%} "
              f"over {both:,} voxels")

    # final scroll-global ids
    global_table: dict[str, dict[str, int]] = {}
    final: dict[int, int] = {}
    next_id = 1
    for i, (z0, _) in enumerate(slab_dirs):
        for tkey, table in tables[z0].items():
            out = {}
            for local, slab_global in table.items():
                root = find(gid(i, slab_global))
                if root not in final:
                    final[root] = next_id
                    next_id += 1
                out[local] = final[root]
            global_table[f"z{z0}/{tkey}"] = out
    metrics["global_instances"] = next_id - 1

    with open(run_dir / "global_table.json", "w") as fh:
        json.dump(global_table, fh)
    with open(run_dir / "assembly_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"scroll-global instances: {next_id - 1}")
    print(f"wrote {run_dir / 'global_table.json'}")


if __name__ == "__main__":
    main()
