"""Sheet-instance separation for surface predictions (wishlist #192/#193).

Takes a crop of the official surface prediction, cleans it (CT gating + small
component removal), then splits the binary "papyrus surface" mask into
individual sheet instances:

  1. Euclidean distance transform D inside the mask.
  2. Cores = D >= core_dist (voxels deep inside a sheet). Each core gets an id.
  3. Watershed on -D, constrained to the mask, grows cores back to full sheets.
     Where two sheets touch, the watershed boundary splits them at the neck.

Outputs an instance-labeled crop (.npy) plus colored mid-slice renders.

Usage:
    python scripts/separate_sheets.py [scroll] [pred_level] [cz] [cy] [cx] [nomerge]
      scroll:     scroll3 | pherc1218 (default pherc1218)
      pred_level: prediction pyramid level (default 1)
      cz, cy, cx: crop center in prediction-level voxels (default scroll center)
      nomerge:    disable watershed-instance merging (may appear anywhere)
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from scipy import ndimage
from skimage.segmentation import watershed

from clean_surface_prediction import SCROLLS

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

CROP = (128, 512, 512)  # z, y, x
CORE_DIST = 2.0  # min distance-to-background for watershed seed cores
MIN_CC = 200  # drop noise components smaller than this (voxels)
MIN_CORE = 60  # drop seed cores smaller than this (voxels)
MERGE_ALPHA = 0.75  # minimum shared-boundary salience relative to both interiors
MIN_BORDER = 20  # minimum number of adjacent voxel pairs on a shared boundary
# Central-crop calibration: 100/75 pairs still percolated (11.84%/7.91% max);
# 60 produced 391 instances with a 4.57% maximum while retaining long sheets.
MAX_BORDER = 60
MAX_MERGED_FRACTION = 0.15  # prevent transitive merges spanning the whole mask
BOUNDARY_CHUNK_SIZE = 4_000_000  # cap temporary boundary reductions


def distinct_colors(n: int, seed: int = 7) -> np.ndarray:
    """n visually distinct RGB colors via golden-ratio hue stepping."""
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.61803398875 + rng.random()) % 1.0
    sat = 0.55 + 0.4 * rng.random(n)
    val = 0.75 + 0.25 * rng.random(n)
    h6 = (hues * 6).astype(int) % 6
    f = hues * 6 - np.floor(hues * 6)
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    h6 = h6[:, None]
    rgb = np.select(
        [h6 == 0, h6 == 1, h6 == 2, h6 == 3, h6 == 4, h6 == 5],
        [
            np.stack([val, t, p], -1), np.stack([q, val, p], -1),
            np.stack([p, val, t], -1), np.stack([p, q, val], -1),
            np.stack([t, p, val], -1), np.stack([val, p, q], -1),
        ],
    )
    return (rgb * 255).astype(np.uint8)


def render(labels: np.ndarray, ct: np.ndarray, path: Path, colors: np.ndarray) -> None:
    """Instance labels in color over the grayscale CT slice."""
    bg = np.clip(ct.astype(np.float32) * (255.0 / max(ct.max(), 1)), 0, 255).astype(np.uint8)
    rgb = np.stack([bg, bg, bg], axis=-1) // 2
    m = labels > 0
    rgb[m] = colors[(labels[m] - 1) % len(colors)]
    Image.fromarray(rgb).save(path)


def merge_instances(
    labels: np.ndarray,
    dist: np.ndarray,
    merge_alpha: float = MERGE_ALPHA,
    min_border: int = MIN_BORDER,
    max_border: int = MAX_BORDER,
    max_merged_fraction: float = MAX_MERGED_FRACTION,
) -> tuple[np.ndarray, int, int, int]:
    """Merge watershed regions separated by a non-salient 6-connected border.

    A shared border is considered artificial when its mean depth is at least
    ``merge_alpha`` times the shallower region's 90th-percentile interior
    depth. Broad contacts are rejected because they more likely run along two
    stacked sheets than cut across one sheet. Border statistics are reduced in
    fixed-size chunks; unions that would reach 15% of the full mask are also
    rejected to prevent a chain of valid contacts from spanning the scroll.
    """
    n_before = int(labels.max())
    if n_before == 0:
        return labels.astype(np.int32, copy=False), 0, 0, 0

    indexes = np.arange(1, n_before + 1)
    depths = ndimage.labeled_comprehension(
        dist,
        labels,
        indexes,
        lambda values: np.percentile(values, 90),
        float,
        0.0,
    )

    # Each key encodes an ordered pair (low_label, high_label). Chunking the
    # leading dimension bounds all dense temporaries even if every pair of
    # adjacent voxels belongs to a boundary.
    key_parts = []
    sum_parts = []
    count_parts = []
    key_base = n_before + 1
    for axis in range(labels.ndim):
        lower_slice = [slice(None)] * labels.ndim
        upper_slice = [slice(None)] * labels.ndim
        lower_slice[axis] = slice(None, -1)
        upper_slice[axis] = slice(1, None)
        lower_slice = tuple(lower_slice)
        upper_slice = tuple(upper_slice)

        lower = labels[lower_slice]
        upper = labels[upper_slice]
        lower_dist = dist[lower_slice]
        upper_dist = dist[upper_slice]
        trailing_size = int(np.prod(lower.shape[1:], dtype=np.int64))
        rows_per_chunk = max(1, BOUNDARY_CHUNK_SIZE // max(trailing_size, 1))
        for start in range(0, lower.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, lower.shape[0])
            lower_chunk = lower[start:stop]
            upper_chunk = upper[start:stop]
            boundary = (
                (lower_chunk != upper_chunk)
                & (lower_chunk > 0)
                & (upper_chunk > 0)
            )
            if not boundary.any():
                continue

            label_a = lower_chunk[boundary]
            label_b = upper_chunk[boundary]
            low = np.minimum(label_a, label_b).astype(np.int64, copy=False)
            high = np.maximum(label_a, label_b).astype(np.int64, copy=False)
            keys = low * key_base + high
            salience = np.minimum(
                lower_dist[start:stop][boundary],
                upper_dist[start:stop][boundary],
            )

            unique_keys, inverse = np.unique(keys, return_inverse=True)
            key_parts.append(unique_keys)
            sum_parts.append(np.bincount(inverse, weights=salience))
            count_parts.append(np.bincount(inverse))

    if not key_parts:
        return labels.astype(np.int32, copy=True), n_before, n_before, 0

    all_keys = np.concatenate(key_parts)
    all_sums = np.concatenate(sum_parts)
    all_counts = np.concatenate(count_parts)
    pair_keys, inverse = np.unique(all_keys, return_inverse=True)
    border_sums = np.bincount(inverse, weights=all_sums)
    border_counts = np.bincount(inverse, weights=all_counts).astype(np.int64)
    saliences = border_sums / border_counts

    parent = np.arange(n_before + 1, dtype=np.int32)
    rank = np.zeros(n_before + 1, dtype=np.uint8)
    component_sizes = np.bincount(labels.ravel(), minlength=n_before + 1)
    max_merged_size = component_sizes[1:].sum() * max_merged_fraction

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = int(parent[label])
        return label

    def union(label_a: int, label_b: int) -> bool:
        root_a, root_b = find(label_a), find(label_b)
        if root_a == root_b:
            return False
        if component_sizes[root_a] + component_sizes[root_b] >= max_merged_size:
            return False
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        component_sizes[root_a] += component_sizes[root_b]
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1
        return True

    eligible = (border_counts >= min_border) & (border_counts <= max_border)
    pair_a = pair_keys // key_base
    pair_b = pair_keys % key_base
    thresholds = merge_alpha * np.minimum(depths[pair_a - 1], depths[pair_b - 1])
    eligible &= saliences >= thresholds
    eligible_pairs = np.flatnonzero(eligible)
    eligible_pairs = eligible_pairs[np.argsort(-saliences[eligible_pairs])]
    for pair_index in eligible_pairs:
        union(int(pair_a[pair_index]), int(pair_b[pair_index]))

    roots = np.fromiter(
        (find(label) for label in range(1, n_before + 1)),
        dtype=np.int32,
        count=n_before,
    )
    _, consecutive = np.unique(roots, return_inverse=True)
    remap = np.zeros(n_before + 1, dtype=np.int32)
    remap[1:] = consecutive + 1
    merged = remap[labels]
    n_after = int(consecutive.max()) + 1
    n_merges = n_before - n_after
    return merged, n_before, n_after, n_merges


def main() -> None:
    merge_enabled = "nomerge" not in sys.argv[1:]
    args = [arg for arg in sys.argv[1:] if arg != "nomerge"]
    scroll = args[0] if args else "pherc1218"
    cfg = SCROLLS[scroll]
    pred_level = int(args[1]) if len(args) > 1 else 1
    ct_level = pred_level + cfg["ct_offset"]

    pred_arr = zarr.open_array(f"{cfg['pred']}/{pred_level}", mode="r")
    ct_arr = zarr.open_array(f"{cfg['ct']}/{ct_level}", mode="r")
    shape = pred_arr.shape
    cz = int(args[2]) if len(args) > 2 else shape[0] // 2
    cy = int(args[3]) if len(args) > 3 else shape[1] // 2
    cx = int(args[4]) if len(args) > 4 else shape[2] // 2

    dz, dy, dx = CROP[0] // 2, CROP[1] // 2, CROP[2] // 2
    sl = np.s_[cz - dz : cz + dz, cy - dy : cy + dy, cx - dx : cx + dx]
    print(f"{scroll} PRED L{pred_level} {shape}, crop center ({cz},{cy},{cx}) size {CROP}")

    t0 = time.time()
    pred = np.asarray(pred_arr[sl])
    ct = np.asarray(ct_arr[sl])
    print(f"  streamed {(pred.nbytes + ct.nbytes) / 1e6:.0f} MB in {time.time() - t0:.1f}s")

    # clean: CT gating + drop small components
    mask = (pred > 0) & (ct > 0)
    lab, _ = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    mask = (sizes >= MIN_CC)[lab] & mask
    print(f"  cleaned mask: {int(mask.sum()):,} voxels")

    # separate instances: distance transform -> cores -> watershed
    t0 = time.time()
    dist = ndimage.distance_transform_edt(mask).astype(np.float32)
    cores, _ = ndimage.label(dist >= CORE_DIST)
    core_sizes = np.bincount(cores.ravel())
    cores = np.where((core_sizes >= MIN_CORE)[cores], cores, 0)
    ids = np.unique(cores)
    remap = np.zeros(cores.max() + 1, dtype=np.int32)
    remap[ids] = np.arange(len(ids))
    markers = remap[cores]
    n_inst = len(ids) - 1
    labels = watershed(-dist, markers, mask=mask)
    print(f"  watershed: {n_inst} sheet instances in {time.time() - t0:.1f}s")

    inst_sizes = np.bincount(labels.ravel())[1:]
    if n_inst:
        print(f"  pre-merge size: median {int(np.median(inst_sizes)):,}, max {int(inst_sizes.max()):,} voxels")

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"{scroll}_L{pred_level}_c{cz}-{cy}-{cx}"
    colors = distinct_colors(max(n_inst, 1))
    render(labels[dz], ct[dz], OUT_DIR / f"sheets_{tag}_zmid_premerge.png", colors)
    render(labels[:, dy, :], ct[:, dy, :], OUT_DIR / f"sheets_{tag}_ymid_premerge.png", colors)

    t0 = time.time()
    if merge_enabled:
        labels, n_before, n_after, n_merges = merge_instances(labels, dist)
    else:
        n_before = n_after = n_inst
        n_merges = 0
    merge_time = time.time() - t0
    print(
        f"  merge: {n_before} -> {n_after} instances, "
        f"{n_merges} fusions in {merge_time:.1f}s"
        + ("" if merge_enabled else " (disabled)")
    )

    merged_sizes = np.bincount(labels.ravel())[1:]
    if n_after:
        mask_size = max(mask.sum(), 1)
        largest_fraction = 100.0 * merged_sizes.max() / mask_size
        print(
            f"  post-merge size: median {int(np.median(merged_sizes)):,}, "
            f"max {int(merged_sizes.max()):,} voxels ({largest_fraction:.1f}% of mask)"
        )
        top_indices = np.argsort(merged_sizes)[-5:][::-1]
        top_sizes = ", ".join(
            f"{int(merged_sizes[index]):,} ({100.0 * merged_sizes[index] / mask_size:.2f}%)"
            for index in top_indices
        )
        print(f"  top-5 post-merge: {top_sizes}")

    colors = distinct_colors(max(n_after, 1))
    render(labels[dz], ct[dz], OUT_DIR / f"sheets_{tag}_zmid_merged.png", colors)
    render(labels[:, dy, :], ct[:, dy, :], OUT_DIR / f"sheets_{tag}_ymid_merged.png", colors)
    np.save(OUT_DIR / f"sheets_{tag}.npy", labels.astype(np.int32, copy=False))
    print(f"Saved renders + instance labels to {OUT_DIR}")


if __name__ == "__main__":
    main()
