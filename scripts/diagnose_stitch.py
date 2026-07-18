"""Diagnose stitching disagreement using a downloaded slab run.

For each adjacent tile pair, rebuilds the overlap-band label pairs, applies
the run's stitch table, and classifies every disagreeing observation:

  frag_split   the a-label's best partner is already linked to another a-label
               (same sheet fragmented differently in the two tiles) -> would be
               fixed by DIRECTED coverage links (many-to-one)
  non_mutual   a's best is b, but b's best is a different a-label AND coverage
               is high -> also candidates for directed links
  weak         best-pair coverage below threshold (genuinely ambiguous)

Usage: python scripts/diagnose_stitch.py output/slab_run
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OVERLAP = 64
STRIDE = 448
COVER = 0.7  # directed link: best pair covers >=70% of the label's band voxels


def band_pairs(run: Path):
    tiles = {}
    for p in (run / "blocks").glob("tile_*.npz"):
        m = re.fullmatch(r"tile_y(\d+)_x(\d+)\.npz", p.name)
        tiles[(int(m.group(1)), int(m.group(2)))] = p
    for (y0, x0), p in sorted(tiles.items()):
        for (ny0, nx0), axis in (((y0, x0 - STRIDE), 2), ((y0 - STRIDE, x0), 1)):
            q = tiles.get((ny0, nx0))
            if q is None:
                continue
            with np.load(q) as d:
                prev = d["labels"]
            with np.load(p) as d:
                cur = d["labels"]
            if axis == 2:
                yield (ny0, nx0), prev[:, :, -OVERLAP:], (y0, x0), cur[:, :, :OVERLAP]
            else:
                yield (ny0, nx0), prev[:, -OVERLAP:, :cur.shape[2]], (y0, x0), cur[:, :OVERLAP, :]


def main() -> None:
    run = Path(sys.argv[1])
    with open(run / "stitch_table.json") as fh:
        table = json.load(fh)

    def to_global(key, arr):
        t = table.get(f"y{key[0]}_x{key[1]}")
        if t is None:
            return None
        lut = np.zeros(arr.max() + 1, dtype=np.int64)
        for k, v in t.items():
            if int(k) <= arr.max():
                lut[int(k)] = v
        return lut[arr]

    stats = defaultdict(int)
    for ka, a_loc, kb, b_loc in band_pairs(run):
        ga = to_global(ka, a_loc)
        gb = to_global(kb, b_loc)
        if ga is None or gb is None:
            continue
        both = (ga > 0) & (gb > 0)
        if not both.any():
            continue
        av, bv = ga[both].astype(np.int64), gb[both].astype(np.int64)
        stats["total"] += len(av)
        agree = av == bv
        stats["agree"] += int(agree.sum())
        dv_a, dv_b = av[~agree], bv[~agree]
        if len(dv_a) == 0:
            continue
        count_a = dict(zip(*[x.tolist() for x in np.unique(av, return_counts=True)]))
        count_b = dict(zip(*[x.tolist() for x in np.unique(bv, return_counts=True)]))
        base = int(bv.max()) + 1
        uk, counts = np.unique(dv_a * base + dv_b, return_counts=True)
        pa = (uk // base).astype(int)
        pb = (uk % base).astype(int)
        # best disagreeing partner per a-label and per b-label
        best_a: dict[int, tuple[int, int]] = {}
        for la, lb, c in zip(pa, pb, counts):
            if c > best_a.get(la, (0, -1))[0]:
                best_a[la] = (int(c), int(lb))
        linked_b = set(av[agree].tolist())  # b-globals equal to some a -> linked
        for la, lb, c in zip(pa, pb, counts):
            cov_a = c / count_a[la]
            if best_a[la][1] == lb and cov_a >= COVER:
                if lb in linked_b:
                    stats["frag_split"] += int(c)
                else:
                    stats["non_mutual"] += int(c)
            else:
                stats["weak"] += int(c)

    total = stats["total"]
    print(f"observations: {total:,}")
    print(f"agree:        {stats['agree']:,} ({stats['agree'] / total:.1%})")
    dis = total - stats["agree"]
    for k in ("frag_split", "non_mutual", "weak"):
        print(f"{k:12s}: {stats[k]:,} ({stats[k] / total:.1%} of total, "
              f"{stats[k] / max(dis, 1):.1%} of disagreement)")
    fixable = stats["frag_split"] + stats["non_mutual"]
    print(f"\nprojected agreement with directed coverage links: "
          f"{(stats['agree'] + fixable) / total:.1%}")


if __name__ == "__main__":
    main()
