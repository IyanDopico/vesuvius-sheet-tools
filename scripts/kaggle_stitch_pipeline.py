"""Vesuvius chunked pipeline with instance stitching (PHerc1218, z-slab @L1).

Processes a 256-slice z-slab of the scroll as overlapping 512x512 (y,x) tiles:
per tile clean -> GPU EDT -> watershed -> calibrated merge -> (mega splitting
with CLAHE + structure tensor when needed), then stitches instances across
tiles by mutual-majority label voting in the 64-voxel overlap bands and a
global union-find.

Outputs to /kaggle/working:
  blocks/tile_y{Y}_x{X}.npz   local int32 labels per tile
  stitch_table.json           (tile, local id) -> global id
  metrics.json                per-tile stats + overlap agreement rate
  slab_zmid.png               stitched color render of the slab's middle slice
  README.txt                  dataset format description

Every GPU step uses the recipe verified by the diagnostics kernel:
upgraded cupy/cucim for EDT, eigh chunked at 500k matrices.
"""

import json
import os
import subprocess
import sys
import time

T_START = time.time()
TIME_BUDGET_S = 7.5 * 3600  # leave margin before Kaggle's 9h cap

t0 = time.time()
pip_res = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off",
     "-U", "cupy-cuda12x", "cucim-cu12", "zarr>=3", "fsspec==2025.3.0", "aiohttp"],
    capture_output=True, text=True,
)
if pip_res.returncode != 0:
    print(pip_res.stdout)
    print(pip_res.stderr)
    raise SystemExit("pip install failed")
print(f"[pip] {time.time() - t0:.0f}s", flush=True)

import cupy as cp
import cupyx.scipy.ndimage as cndi
import numpy as np
import zarr
from cucim.core.operations.morphology import distance_transform_edt as gpu_edt
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.exposure import equalize_adapthist
from skimage.segmentation import watershed

print(f"GPU: {cp.cuda.runtime.getDeviceCount()}x "
      f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} | "
      f"cupy {cp.__version__}", flush=True)

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
CT_URL = f"{S3}/PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr"
PRED_URL = (
    f"{S3}/PHerc1218/representations/predictions/surfaces/"
    "20250521120456-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
LEVEL = 1

# slabs + tiling. Slabs overlap 32 slices in z; cross-slab stitching happens
# in a local assembly step that reads the facing bands from the block npzs.
SLAB = 256
Z_OVERLAP = 32
SLAB_STARTS = [11368]  # tail slab: dense material to the very end
TILE = 512
OVERLAP = 64
STRIDE = TILE - OVERLAP
OCC_MIN = 0.004  # min predicted-voxel fraction for a tile to be processed

# calibrated pipeline constants (see vesuvius-sheet-tools)
CORE_DIST = 2.0
MIN_CC = 200
MIN_CORE = 60
MERGE_ALPHA = 0.75
MIN_BORDER = 20
MAX_BORDER = 60
MAX_MERGED_FRACTION = 0.15
BOUNDARY_CHUNK_SIZE = 4_000_000

# splitting (crushed stacks)
SPLIT_MEGAS = True
MEGA_FRAC = 0.03
CLAHE_KERNEL = 64
CLAHE_CLIP = 0.02
SIGMA_GRAD = 1.2
SIGMA_TENSOR = 2.5
SIGMA_INT = 1.0
NMS_STEP = 1.5
P_MIN = 0.35
D_MAX = 4.0
ALIGN_MIN = 0.90
INPLANE_MAX = 0.35
MIN_CONTACT = 5
MIN_CENTER_CC = 100
EIGH_CHUNK = 500_000

# stitching: mutual-best pairs link with a small absolute floor; additionally
# DIRECTED links fire when a label's best partner covers most of its own band
# presence (many-to-one: fixes sheets fragmented differently in the two tiles
# - measured at +10.8pp agreement on the validation slab).
MIN_VOTES_ABS = 20
DIRECTED_COVER = 0.7

OUT = "/kaggle/working"
os.makedirs(f"{OUT}/blocks", exist_ok=True)

LABEL_STEP = 1_000_000  # global id space per tile


def free_gpu() -> None:
    cp.get_default_memory_pool().free_all_blocks()


def read_retry(arr, sl, attempts: int = 6):
    """S3 chunk reads over hours WILL hit transient 503s; back off and retry."""
    for i in range(attempts):
        try:
            return np.asarray(arr[sl])
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1:
                raise
            wait = 5 * (2 ** i)
            print(f"  read retry {i + 1}/{attempts} in {wait}s: {str(exc)[:110]}",
                  flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------- pipeline ops
def merge_instances(labels, dist):
    n_before = int(labels.max())
    if n_before == 0:
        return labels.astype(np.int32, copy=False), 0
    indexes = np.arange(1, n_before + 1)
    depths = ndimage.labeled_comprehension(
        dist, labels, indexes, lambda v: np.percentile(v, 90), float, 0.0
    )
    key_parts, sum_parts, count_parts = [], [], []
    key_base = n_before + 1
    for axis in range(labels.ndim):
        lo = [slice(None)] * labels.ndim
        up = [slice(None)] * labels.ndim
        lo[axis] = slice(None, -1)
        up[axis] = slice(1, None)
        lo, up = tuple(lo), tuple(up)
        lower, upper = labels[lo], labels[up]
        lower_dist, upper_dist = dist[lo], dist[up]
        trailing = int(np.prod(lower.shape[1:], dtype=np.int64))
        rows = max(1, BOUNDARY_CHUNK_SIZE // max(trailing, 1))
        for start in range(0, lower.shape[0], rows):
            stop = min(start + rows, lower.shape[0])
            lc, uc = lower[start:stop], upper[start:stop]
            boundary = (lc != uc) & (lc > 0) & (uc > 0)
            if not boundary.any():
                continue
            a, b = lc[boundary], uc[boundary]
            low = np.minimum(a, b).astype(np.int64, copy=False)
            high = np.maximum(a, b).astype(np.int64, copy=False)
            keys = low * key_base + high
            sal = np.minimum(lower_dist[start:stop][boundary],
                             upper_dist[start:stop][boundary])
            uk, inv = np.unique(keys, return_inverse=True)
            key_parts.append(uk)
            sum_parts.append(np.bincount(inv, weights=sal))
            count_parts.append(np.bincount(inv))
    if not key_parts:
        return labels.astype(np.int32, copy=True), n_before
    pair_keys, inv = np.unique(np.concatenate(key_parts), return_inverse=True)
    border_sums = np.bincount(inv, weights=np.concatenate(sum_parts))
    border_counts = np.bincount(inv, weights=np.concatenate(count_parts)).astype(np.int64)
    saliences = border_sums / border_counts
    parent = np.arange(n_before + 1, dtype=np.int32)
    rank = np.zeros(n_before + 1, dtype=np.uint8)
    comp = np.bincount(labels.ravel(), minlength=n_before + 1)
    cap = comp[1:].sum() * MAX_MERGED_FRACTION

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb or comp[ra] + comp[rb] >= cap:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        comp[ra] += comp[rb]
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    eligible = (border_counts >= MIN_BORDER) & (border_counts <= MAX_BORDER)
    pa = pair_keys // key_base
    pb = pair_keys % key_base
    eligible &= saliences >= MERGE_ALPHA * np.minimum(depths[pa - 1], depths[pb - 1])
    order = np.flatnonzero(eligible)
    order = order[np.argsort(-saliences[order])]
    for i in order:
        union(int(pa[i]), int(pb[i]))
    roots = np.fromiter((find(x) for x in range(1, n_before + 1)), np.int32, n_before)
    _, consecutive = np.unique(roots, return_inverse=True)
    remap = np.zeros(n_before + 1, dtype=np.int32)
    remap[1:] = consecutive + 1
    return remap[labels], int(consecutive.max()) + 1


def structure_normals_gpu(eq_np):
    """Sheet normals + planarity on GPU with chunked eigh (verified recipe)."""
    vol = cp.asarray(eq_np, dtype=cp.float32)
    grads = [cndi.gaussian_filter(vol, SIGMA_GRAD,
                                  order=tuple(int(i == ax) for i in range(3)))
             for ax in range(3)]
    J = cp.empty(vol.shape + (3, 3), dtype=cp.float32)
    for i in range(3):
        for j in range(i, 3):
            Jij = cndi.gaussian_filter(grads[i] * grads[j], SIGMA_TENSOR)
            J[..., i, j] = Jij
            J[..., j, i] = Jij
    del grads, vol
    J_flat = J.reshape(-1, 3, 3)
    del J
    n_vox = J_flat.shape[0]
    normals = cp.empty((n_vox, 3), dtype=cp.float32)
    planarity = cp.empty(n_vox, dtype=cp.float32)
    for s in range(0, n_vox, EIGH_CHUNK):
        e = min(s + EIGH_CHUNK, n_vox)
        w, v = cp.linalg.eigh(J_flat[s:e])
        normals[s:e] = v[..., :, 2]
        planarity[s:e] = (w[..., 2] - w[..., 1]) / (w[..., 2] + 1e-6)
    del J_flat
    out_n = cp.asnumpy(normals).reshape(eq_np.shape + (3,))
    out_p = cp.asnumpy(planarity).reshape(eq_np.shape)
    del normals, planarity
    free_gpu()
    return out_n, out_p


def consolidate_centerlines(center_lab, coords_r, normals_r, labels_r):
    n_lab = int(center_lab.max())
    lut = np.arange(n_lab + 1, dtype=np.int32)
    if len(coords_r) == 0:
        return lut
    tree = cKDTree(coords_r)
    dist, idx = tree.query(coords_r, k=10, distance_upper_bound=D_MAX)
    n_pts = coords_r.shape[0]
    src = np.repeat(np.arange(n_pts), idx.shape[1])
    dst = idx.ravel()
    valid = np.isfinite(dist.ravel()) & (dst < n_pts)
    src, dst = src[valid], dst[valid]
    keep = (src < dst) & (labels_r[src] != labels_r[dst])
    src, dst = src[keep], dst[keep]
    if len(src) == 0:
        return lut
    disp = coords_r[dst] - coords_r[src]
    disp /= np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
    inplane = np.abs(np.einsum("ij,ij->i", disp, normals_r[src]))
    align = np.abs(np.einsum("ij,ij->i", normals_r[src], normals_r[dst]))
    la = labels_r[src].astype(np.int64)
    lb = labels_r[dst].astype(np.int64)
    lo, hi = np.minimum(la, lb), np.maximum(la, lb)
    key = lo * (n_lab + 1) + hi
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    ok = ((cnt >= MIN_CONTACT)
          & (np.bincount(inv, weights=inplane) / cnt <= INPLANE_MAX)
          & (np.bincount(inv, weights=align) / cnt >= ALIGN_MIN))
    parent = np.arange(n_lab + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for k in uk[ok]:
        a, b = find(int(k // (n_lab + 1))), find(int(k % (n_lab + 1)))
        if a != b:
            parent[b] = a
    roots = np.fromiter((find(x) for x in range(n_lab + 1)), np.int32, n_lab + 1)
    _, lut = np.unique(roots, return_inverse=True)
    return lut.astype(np.int32)


def split_megas(labels, n_inst, ct, mask):
    """Split mega-instances with CLAHE + tensor normals (verified recipe)."""
    inst_sizes = np.bincount(labels.ravel())
    mask_total = int(mask.sum())
    mega_ids = [i for i in range(1, len(inst_sizes))
                if inst_sizes[i] >= MEGA_FRAC * mask_total]
    if not mega_ids:
        return labels, n_inst, 0
    eq = np.empty(ct.shape, dtype=np.float32)
    for z in range(ct.shape[0]):
        eq[z] = equalize_adapthist(ct[z], kernel_size=CLAHE_KERNEL,
                                   clip_limit=CLAHE_CLIP).astype(np.float32)
    normals, planarity = structure_normals_gpu(eq)
    eq_s = ndimage.gaussian_filter(eq, SIGMA_INT)
    del eq
    mega_mask = np.isin(labels, mega_ids)
    vz, vy, vx = np.nonzero(mega_mask)
    coords = np.stack([vz, vy, vx]).astype(np.float32)
    nrm = normals[vz, vy, vx].T
    del normals
    val = eq_s[vz, vy, vx]
    up = ndimage.map_coordinates(eq_s, coords + NMS_STEP * nrm, order=1, mode="nearest")
    dn = ndimage.map_coordinates(eq_s, coords - NMS_STEP * nrm, order=1, mode="nearest")
    coherent = planarity[vz, vy, vx] >= P_MIN
    del planarity
    is_ridge = (val > up) & (val > dn) & coherent
    centerline = np.zeros(mask.shape, dtype=bool)
    centerline[vz[is_ridge], vy[is_ridge], vx[is_ridge]] = True
    center_lab, _ = ndimage.label(centerline)
    lut = consolidate_centerlines(
        center_lab,
        coords.T[is_ridge],
        nrm.T[is_ridge],
        center_lab[vz[is_ridge], vy[is_ridge], vx[is_ridge]],
    )
    center_lab = lut[center_lab]
    c_sizes = np.bincount(center_lab.ravel())
    center_lab = np.where((c_sizes >= MIN_CENTER_CC)[center_lab], center_lab, 0)
    assigned = watershed(-eq_s, center_lab, mask=mega_mask)
    new_labels = labels.astype(np.int32, copy=True)
    offset = int(labels.max())
    zone = mega_mask & (assigned > 0)
    new_labels[zone] = offset + assigned[zone]
    _, compact = np.unique(new_labels, return_inverse=True)
    new_labels = compact.reshape(labels.shape).astype(np.int32)
    return new_labels, int(new_labels.max()), len(mega_ids)


def process_tile(pred_arr, ct_arr, z0, z1, y0, x0, ny, nx):
    """Full pipeline on one tile; returns local labels + stats."""
    sl = np.s_[z0:z1, y0:y0 + ny, x0:x0 + nx]
    pred = read_retry(pred_arr, sl)
    ct = read_retry(ct_arr, sl)
    mask = (pred > 0) & (ct > 0)
    del pred
    if not mask.any():
        return None, ct, {}
    m = cp.asarray(mask)
    lab, _ = cndi.label(m)
    sizes = cp.bincount(lab.ravel())
    mask = cp.asnumpy((sizes >= MIN_CC)[lab] & m)
    del m, lab, sizes
    free_gpu()
    if not mask.any():
        return None, ct, {}
    dist = cp.asnumpy(gpu_edt(cp.asarray(mask))).astype(np.float32)
    free_gpu()
    cores, _ = ndimage.label(dist >= CORE_DIST)
    core_sizes = np.bincount(cores.ravel())
    cores = np.where((core_sizes >= MIN_CORE)[cores], cores, 0)
    ids = np.unique(cores)
    if len(ids) < 2:
        return None, ct, {}
    remap = np.zeros(cores.max() + 1, dtype=np.int32)
    remap[ids] = np.arange(len(ids))
    labels = watershed(-dist, remap[cores], mask=mask)
    del cores, remap
    labels, n_after = merge_instances(labels, dist)
    del dist
    n_megas = 0
    if SPLIT_MEGAS:
        labels, n_after, n_megas = split_megas(labels, n_after, ct, mask)
    stats = {
        "instances": n_after,
        "mask_voxels": int(mask.sum()),
        "megas_split": n_megas,
    }
    return labels, ct, stats


# ------------------------------------------------------------------ stitching
parent: dict[int, int] = {}


def find_g(x: int) -> int:
    while parent.setdefault(x, x) != x:
        parent[x] = parent.get(parent[x], parent[x])
        x = parent[x]
    return x


def union_g(a: int, b: int) -> None:
    ra, rb = find_g(a), find_g(b)
    if ra != rb:
        parent[rb] = ra


def stitch_band(prev_lab, cur_lab, prev_gid0, cur_gid0, bands):
    """Mutual-majority vote between two label arrays over the same voxels.
    Links mutual-best pairs and records the raw pair counts in `bands` so the
    agreement rate can be recomputed once ALL unions are known (a
    traversal-time measurement would be order-dependent). Returns n_links."""
    both = (prev_lab > 0) & (cur_lab > 0)
    if not both.any():
        return 0
    a = prev_lab[both].astype(np.int64)
    b = cur_lab[both].astype(np.int64)
    band_count_a = np.bincount(a)
    band_count_b = np.bincount(b)
    base = int(b.max()) + 1
    keys = a * base + b
    uk, counts = np.unique(keys, return_counts=True)
    ka = (uk // base).astype(np.int64)
    kb = (uk % base).astype(np.int64)
    bands.append((prev_gid0, cur_gid0, ka, kb, counts))
    best_for_a: dict[int, tuple[int, int]] = {}
    best_for_b: dict[int, tuple[int, int]] = {}
    for la, lb, c in zip(ka, kb, counts):
        if c > best_for_a.get(la, (0, -1))[0]:
            best_for_a[la] = (int(c), int(lb))
        if c > best_for_b.get(lb, (0, -1))[0]:
            best_for_b[lb] = (int(c), int(la))
    n_links = 0
    # mutual-best links
    for la, (c, lb) in best_for_a.items():
        if c >= MIN_VOTES_ABS and best_for_b.get(lb, (0, -1))[1] == la:
            union_g(prev_gid0 + la, cur_gid0 + lb)
            n_links += 1
    # directed coverage links (many-to-one across differing fragmentations)
    for la, (c, lb) in best_for_a.items():
        if c >= MIN_VOTES_ABS and c >= DIRECTED_COVER * band_count_a[la]:
            union_g(prev_gid0 + la, cur_gid0 + lb)
            n_links += 1
    for lb, (c, la) in best_for_b.items():
        if c >= MIN_VOTES_ABS and c >= DIRECTED_COVER * band_count_b[lb]:
            union_g(prev_gid0 + la, cur_gid0 + lb)
            n_links += 1
    return n_links


# ---------------------------------------------------------------------- main
pred_arr = zarr.open_array(f"{PRED_URL}/{LEVEL}", mode="r")
ct_arr = zarr.open_array(f"{CT_URL}/{LEVEL}", mode="r")
FULL_Y, FULL_X = pred_arr.shape[1], pred_arr.shape[2]

occ_arr = zarr.open_array(f"{PRED_URL}/5", mode="r")
metrics: dict = {"slabs": {}, "run": {}}
stopped_early = False


def run_slab(z0: int, z1: int) -> dict:
    """Process one z-slab: tiles -> stitch -> tables/render. Returns stats."""
    global stopped_early
    parent.clear()
    # occupancy from L5 (each L5 voxel = 16^3 L1 voxels)
    occ5 = read_retry(occ_arr, np.s_[z0 // 16 : max(-(-z1 // 16), z0 // 16 + 1)])
    occ2d = (occ5 > 0).mean(axis=0)
    tiles = []
    for y0 in range(0, FULL_Y - OVERLAP, STRIDE):
        for x0 in range(0, FULL_X - OVERLAP, STRIDE):
            ny = min(TILE, FULL_Y - y0)
            nx = min(TILE, FULL_X - x0)
            o = occ2d[y0 // 16 : -(-(y0 + ny) // 16), x0 // 16 : -(-(x0 + nx) // 16)]
            if o.size and o.mean() >= OCC_MIN:
                tiles.append((y0, x0, ny, nx))
    print(f"slab z[{z0}:{z1}] -> {len(tiles)} occupied tiles", flush=True)
    os.makedirs(f"{OUT}/blocks/z{z0}", exist_ok=True)

    tile_gid0: dict[tuple[int, int], int] = {}
    right_strips: dict[tuple[int, int], np.ndarray] = {}
    bottom_strips: dict[tuple[int, int], np.ndarray] = {}
    zmid_slices: dict[tuple[int, int], np.ndarray] = {}
    zmid_ct: dict[tuple[int, int], np.ndarray] = {}
    tile_stats: dict = {}
    bands: list = []
    total_links = 0
    processed = 0

    for t_idx, (y0, x0, ny, nx) in enumerate(tiles):
        if time.time() - T_START > TIME_BUDGET_S:
            print(f"TIME BUDGET at tile {t_idx}/{len(tiles)} of slab z{z0}",
                  flush=True)
            stopped_early = True
            break
        t0 = time.time()
        labels, ct, stats = process_tile(pred_arr, ct_arr, z0, z1, y0, x0, ny, nx)
        key = (y0, x0)
        gid0 = (t_idx + 1) * LABEL_STEP
        tile_gid0[key] = gid0
        if labels is None:
            continue
        processed += 1
        np.savez_compressed(f"{OUT}/blocks/z{z0}/tile_y{y0}_x{x0}.npz",
                            labels=labels, z0=z0, y0=y0, x0=x0)
        left = (y0, x0 - STRIDE)
        if left in right_strips:
            total_links += stitch_band(
                right_strips.pop(left), labels[:, :, :OVERLAP],
                tile_gid0[left], gid0, bands)
        top = (y0 - STRIDE, x0)
        if top in bottom_strips:
            total_links += stitch_band(
                bottom_strips.pop(top)[:, :, : labels.shape[2]],
                labels[:, :OVERLAP, :],
                tile_gid0[top], gid0, bands)
        right_strips[key] = labels[:, :, -OVERLAP:].copy()
        bottom_strips[key] = labels[:, -OVERLAP:, :].copy()
        zmid_slices[key] = labels[labels.shape[0] // 2].copy()
        zmid_ct[key] = ct[ct.shape[0] // 2].copy()
        stats["seconds"] = round(time.time() - t0, 1)
        tile_stats[f"y{y0}_x{x0}"] = stats
        print(f"  z{z0} tile {t_idx + 1}/{len(tiles)} y{y0} x{x0}: "
              f"{stats['instances']} inst, {stats['megas_split']} megas, "
              f"{stats['seconds']}s", flush=True)

    # per-slab global table
    stitch_table: dict[str, dict[str, int]] = {}
    final_ids: dict[int, int] = {}
    next_id = 1
    for key, gid0 in tile_gid0.items():
        tkey = f"y{key[0]}_x{key[1]}"
        stats = tile_stats.get(tkey)
        if not stats:
            continue
        table = {}
        for local in range(1, stats["instances"] + 1):
            root = find_g(gid0 + local)
            if root not in final_ids:
                final_ids[root] = next_id
                next_id += 1
            table[str(local)] = final_ids[root]
        stitch_table[tkey] = table

    # agreement recomputed AFTER all unions (order-independent). Counts are
    # pairwise band observations; corner voxels appear in up to 4 bands.
    total_agree = total_both = 0
    for prev_gid0, cur_gid0, ka, kb, counts in bands:
        for la, lb, c in zip(ka, kb, counts):
            total_both += int(c)
            if find_g(prev_gid0 + int(la)) == find_g(cur_gid0 + int(lb)):
                total_agree += int(c)

    with open(f"{OUT}/stitch_table_z{z0}.json", "w") as fh:
        json.dump(stitch_table, fh)

    # stitched render of the slab's middle slice (2x downsampled)
    canvas = np.zeros((FULL_Y, FULL_X), dtype=np.int32)
    ct_canvas = np.zeros((FULL_Y, FULL_X), dtype=np.uint8)
    for (y0, x0), sl2d in zmid_slices.items():
        table = stitch_table.get(f"y{y0}_x{x0}")
        if table is None:
            continue
        lut = np.zeros(sl2d.max() + 1, dtype=np.int32)
        for lstr, g in table.items():
            li = int(lstr)
            if li <= sl2d.max():
                lut[li] = g
        own_y = OVERLAP // 2 if (y0 - STRIDE, x0) in zmid_slices else 0
        own_x = OVERLAP // 2 if (y0, x0 - STRIDE) in zmid_slices else 0
        gy, gx = y0 + own_y, x0 + own_x
        piece = lut[sl2d[own_y:, own_x:]]
        canvas[gy : gy + piece.shape[0], gx : gx + piece.shape[1]] = piece
        ctp = zmid_ct[(y0, x0)][own_y:, own_x:]
        ct_canvas[gy : gy + ctp.shape[0], gx : gx + ctp.shape[1]] = ctp
    rng = np.random.default_rng(7)
    n_glob = max(next_id - 1, 1)
    palette = rng.integers(60, 255, size=(n_glob + 1, 3), dtype=np.uint8)
    palette[0] = 0
    small = canvas[::2, ::2]
    bg = ct_canvas[::2, ::2] // 2
    rgb = np.stack([bg, bg, bg], -1)
    m2 = small > 0
    rgb[m2] = palette[small[m2] % (n_glob + 1)]
    Image.fromarray(rgb).save(f"{OUT}/slab_z{z0}.png")

    slab_stats = {
        "tiles": tile_stats,
        "expected_tiles": len(tiles),
        "processed_tiles": processed,
        "links": total_links,
        "overlap_pair_observations": total_both,
        "agreement_rate": round(total_agree / max(total_both, 1), 4),
        "global_instances": next_id - 1,
    }
    print(f"slab z{z0}: {total_links} links, agreement "
          f"{slab_stats['agreement_rate']:.1%}, "
          f"{next_id - 1} instances, {processed}/{len(tiles)} tiles", flush=True)
    return slab_stats


for z0 in SLAB_STARTS:
    if stopped_early:
        break
    z1 = min(z0 + SLAB, pred_arr.shape[0])
    metrics["slabs"][f"z{z0}"] = run_slab(z0, z1)

metrics["run"] = {
    "slab_starts": SLAB_STARTS,
    "slab_size": SLAB,
    "z_overlap": Z_OVERLAP,
    "complete": not stopped_early,
    "level": LEVEL,
}
with open(f"{OUT}/metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

with open(f"{OUT}/README.txt", "w") as fh:
    fh.write(
        "PHerc1218 sheet-instance labels @L1 (17.28 um/voxel)\n"
        "blocks/zZ/tile_yY_xX.npz: local int32 'labels' + block origin\n"
        "stitch_table_zZ.json: per tile, local label id -> global id (per slab)\n"
        "Slabs overlap {} slices in z; cross-slab stitching is done by the\n"
        "assembly script in vesuvius-sheet-tools.\n"
        "metrics.json: per-slab per-tile stats + overlap agreement rates\n"
        "Produced by vesuvius-sheet-tools (github.com/IyanDopico/vesuvius-sheet-tools)\n"
        .format(Z_OVERLAP)
    )

print(f"TOTAL {time.time() - T_START:.0f}s", flush=True)
if stopped_early:
    print("=== PARTIAL RUN (time budget) ===", flush=True)
else:
    print("=== ALL DONE, ZERO ERRORS ===", flush=True)
