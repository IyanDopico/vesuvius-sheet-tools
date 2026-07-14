"""Split watershed mega-instances (fused sheet stacks) using local orientation.

Community-informed prototype (thanks: sean/bruniss for CLAHE+windowing,
nvining for the skepticism): in badly compressed regions the watershed
produces one mega-instance spanning many physical wraps. This script splits
those by exploiting that stacked sheets still show distinct intensity ridges:

  1. CLAHE per z-slice amplifies the faint seams between stacked sheets.
  2. Structure tensor of the equalized volume -> sheet normal per voxel.
  3. Non-maximum suppression of smoothed intensity ALONG the normal: voxels
     that are ridge maxima along their own normal = per-sheet centerlines.
     Two fused sheets yield two separate centerline components.
  4. Every voxel of the mega-instance is reassigned to its nearest centerline
     component; non-mega instances are left untouched.

Usage:
    python scripts/split_stacked.py [scroll] [pred_level] [cz] [cy] [cx]
      defaults: pherc1218, level 1, crushed-tip crop (5812, 2320, 860)
"""

import sys
import time
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.exposure import equalize_adapthist
from skimage.segmentation import watershed

from clean_surface_prediction import SCROLLS
from separate_sheets import (
    CORE_DIST,
    MIN_CC,
    MIN_CORE,
    distinct_colors,
    merge_instances,
    render,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

CROP = (128, 512, 512)
MEGA_FRAC = 0.03  # instances above this fraction of the mask get split
CLAHE_KERNEL = 64
CLAHE_CLIP = 0.02
SIGMA_GRAD = 1.2
SIGMA_TENSOR = 2.5
SIGMA_INT = 1.0  # smoothing of the equalized intensity used for ridge NMS
NMS_STEP = 1.5  # voxels along the normal for the ridge test
MIN_CENTER_CC = 100  # drop centerline components smaller than this
EIGH_CHUNK = 16  # z-slices per eigh batch

# iteration 2: coherence gating + orientation-guided centerline consolidation
P_MIN = 0.35  # minimum tensor planarity for a voxel to seed a ridge
D_MAX = 4.0  # max gap (voxels) bridged between centerline fragments
ALIGN_MIN = 0.90  # min |n1.n2| between fragments to consider them one sheet
INPLANE_MAX = 0.35  # max |disp_unit.normal|: bridge along the sheet, not across
MIN_CONTACT = 5  # min close voxel pairs before a fragment pair may merge


def structure_normals(eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sheet normal + planarity per voxel from the smoothed structure tensor
    of the (CLAHE-equalized) intensity. Planarity = (l3-l2)/l3 in [0,1]."""
    grads = [
        ndimage.gaussian_filter(eq, SIGMA_GRAD, order=tuple(int(i == ax) for i in range(3)))
        for ax in range(3)
    ]
    J = np.empty(eq.shape + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(i, 3):
            Jij = ndimage.gaussian_filter(grads[i] * grads[j], SIGMA_TENSOR)
            J[..., i, j] = Jij
            J[..., j, i] = Jij
    del grads
    normals = np.empty(eq.shape + (3,), dtype=np.float32)
    planarity = np.empty(eq.shape, dtype=np.float32)
    for z0 in range(0, eq.shape[0], EIGH_CHUNK):
        z1 = min(z0 + EIGH_CHUNK, eq.shape[0])
        w, v = np.linalg.eigh(J[z0:z1])
        normals[z0:z1] = v[..., :, 2]
        planarity[z0:z1] = (w[..., 2] - w[..., 1]) / (w[..., 2] + 1e-6)
    return normals, planarity


def consolidate_centerlines(
    center_lab: np.ndarray,
    coords_r: np.ndarray,
    normals_r: np.ndarray,
    labels_r: np.ndarray,
) -> np.ndarray:
    """Merge centerline fragments that continue each other ALONG the sheet.

    Two fragments merge only when they are close (<= D_MAX), their normals
    agree (>= ALIGN_MIN) and the displacement between them lies in the sheet
    plane (|disp.normal| <= INPLANE_MAX) — never across the normal, which is
    what keeps stacked sheets separate. Returns a relabel lookup table.
    """
    n_lab = int(center_lab.max())
    tree = cKDTree(coords_r)
    dist, idx = tree.query(coords_r, k=10, distance_upper_bound=D_MAX)
    n_pts = coords_r.shape[0]
    src = np.repeat(np.arange(n_pts), idx.shape[1])
    dst = idx.ravel()
    valid = np.isfinite(dist.ravel()) & (dst < n_pts)
    src, dst = src[valid], dst[valid]
    keep = (src < dst) & (labels_r[src] != labels_r[dst])
    src, dst = src[keep], dst[keep]
    lut = np.arange(n_lab + 1, dtype=np.int32)
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
    ok = (
        (cnt >= MIN_CONTACT)
        & (np.bincount(inv, weights=inplane) / cnt <= INPLANE_MAX)
        & (np.bincount(inv, weights=align) / cnt >= ALIGN_MIN)
    )

    parent = np.arange(n_lab + 1, dtype=np.int32)

    def find(x: int) -> int:
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
    lut = lut.astype(np.int32)
    lut[0] = 0
    return lut


def main() -> None:
    scroll = sys.argv[1] if len(sys.argv) > 1 else "pherc1218"
    cfg = SCROLLS[scroll]
    pred_level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ct_level = pred_level + cfg["ct_offset"]
    cz = int(sys.argv[3]) if len(sys.argv) > 3 else 5812
    cy = int(sys.argv[4]) if len(sys.argv) > 4 else 2320
    cx = int(sys.argv[5]) if len(sys.argv) > 5 else 860

    dz, dy, dx = CROP[0] // 2, CROP[1] // 2, CROP[2] // 2
    sl = np.s_[cz - dz : cz + dz, cy - dy : cy + dy, cx - dx : cx + dx]

    t0 = time.time()
    pred = np.asarray(zarr.open_array(f"{cfg['pred']}/{pred_level}", mode="r")[sl])
    ct = np.asarray(zarr.open_array(f"{cfg['ct']}/{ct_level}", mode="r")[sl])
    print(f"streamed {(pred.nbytes + ct.nbytes) / 1e6:.0f} MB in {time.time() - t0:.1f}s")

    # baseline pipeline: clean -> EDT -> watershed -> merge
    mask = (pred > 0) & (ct > 0)
    del pred
    lab, _ = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    mask = (sizes >= MIN_CC)[lab] & mask
    del lab
    dist = ndimage.distance_transform_edt(mask).astype(np.float32)
    cores, _ = ndimage.label(dist >= CORE_DIST)
    core_sizes = np.bincount(cores.ravel())
    cores = np.where((core_sizes >= MIN_CORE)[cores], cores, 0)
    ids = np.unique(cores)
    remap = np.zeros(cores.max() + 1, dtype=np.int32)
    remap[ids] = np.arange(len(ids))
    labels = watershed(-dist, remap[cores], mask=mask)
    del cores, remap
    labels, n_before, n_after, _ = merge_instances(labels, dist)
    print(f"baseline: {n_before} -> {n_after} instances after merge")

    inst_sizes = np.bincount(labels.ravel())
    mask_total = int(mask.sum())
    mega_ids = [
        i for i in range(1, len(inst_sizes)) if inst_sizes[i] >= MEGA_FRAC * mask_total
    ]
    print(f"mega-instances (>{MEGA_FRAC:.0%} of mask): "
          f"{[(i, int(inst_sizes[i])) for i in mega_ids]}")
    if not mega_ids:
        print("nothing to split in this crop")
        return

    # 1. CLAHE per z-slice (bruniss' suggestion) on the raw CT
    t0 = time.time()
    eq = np.empty(ct.shape, dtype=np.float32)
    for z in range(ct.shape[0]):
        eq[z] = equalize_adapthist(
            ct[z], kernel_size=CLAHE_KERNEL, clip_limit=CLAHE_CLIP
        ).astype(np.float32)
    print(f"CLAHE: {time.time() - t0:.1f}s")

    # 2. orientation + coherence on the equalized volume
    t0 = time.time()
    normals, planarity = structure_normals(eq)
    print(f"structure tensor + normals: {time.time() - t0:.1f}s")

    # 3. ridge NMS along the normal, gated by tensor coherence (P_MIN)
    t0 = time.time()
    eq_s = ndimage.gaussian_filter(eq, SIGMA_INT)
    mega_mask = np.isin(labels, mega_ids)
    vz, vy, vx = np.nonzero(mega_mask)
    coords = np.stack([vz, vy, vx]).astype(np.float32)
    nrm = normals[vz, vy, vx].T  # (3, N)
    val = eq_s[vz, vy, vx]
    up = ndimage.map_coordinates(eq_s, coords + NMS_STEP * nrm, order=1, mode="nearest")
    dn = ndimage.map_coordinates(eq_s, coords - NMS_STEP * nrm, order=1, mode="nearest")
    coherent = planarity[vz, vy, vx] >= P_MIN
    is_ridge = (val > up) & (val > dn) & coherent
    centerline = np.zeros(mask.shape, dtype=bool)
    centerline[vz[is_ridge], vy[is_ridge], vx[is_ridge]] = True
    print(f"ridge NMS: {time.time() - t0:.1f}s, "
          f"{int(is_ridge.sum()):,}/{len(val):,} mega voxels are coherent ridge "
          f"({int((~coherent).sum()):,} below P_MIN)")

    # 4a. label centerline fragments, then consolidate along the sheet plane
    t0 = time.time()
    center_lab, n_center = ndimage.label(centerline)
    coords_r = coords.T[is_ridge]  # (N_r, 3)
    normals_r = nrm.T[is_ridge]
    labels_r = center_lab[vz[is_ridge], vy[is_ridge], vx[is_ridge]]
    lut = consolidate_centerlines(center_lab, coords_r, normals_r, labels_r)
    center_lab = lut[center_lab]
    n_consolidated = int(center_lab.max())
    c_sizes = np.bincount(center_lab.ravel())
    center_lab = np.where((c_sizes >= MIN_CENTER_CC)[center_lab], center_lab, 0)
    kept = np.unique(center_lab)
    print(f"centerline components: {n_center} -> {n_consolidated} consolidated "
          f"-> {len(kept) - 1} after size filter ({time.time() - t0:.1f}s)")

    # 4b. grow each centerline sheet with an intensity watershed: boundaries
    # land in the CLAHE valleys (the seams), not at euclidean midpoints
    t0 = time.time()
    assigned = watershed(-eq_s, center_lab, mask=mega_mask)
    new_labels = labels.astype(np.int32, copy=True)
    offset = int(labels.max())
    split_zone = mega_mask & (assigned > 0)
    new_labels[split_zone] = offset + assigned[split_zone]
    # compact ids
    _, compact = np.unique(new_labels, return_inverse=True)
    new_labels = compact.reshape(new_labels.shape).astype(np.int32)
    n_final = int(new_labels.max())
    print(f"reassign: {time.time() - t0:.1f}s -> {n_final} total instances "
          f"({n_final - n_after:+d} vs baseline)")

    final_sizes = np.bincount(new_labels.ravel())[1:]
    print(f"largest instance now {100.0 * final_sizes.max() / mask_total:.2f}% of mask "
          f"(was {100.0 * inst_sizes[mega_ids[0]] / mask_total:.2f}%)")

    # thickness diagnostic: instance volume / centerline voxels ~ mean sheet
    # thickness. Real single sheets at L1 sit around 4-8; stacked blobs higher.
    cl_per_new = np.bincount(
        new_labels[centerline].ravel(), minlength=len(final_sizes) + 1
    )[1:]
    split_ids = np.flatnonzero(cl_per_new > 0)
    order = split_ids[np.argsort(-final_sizes[split_ids])][:5]
    print("top split instances (size, centerline vox, thickness proxy):")
    for i in order:
        print(f"  #{i + 1}: {int(final_sizes[i]):,} vox, "
              f"{int(cl_per_new[i]):,} cl, ratio {final_sizes[i] / cl_per_new[i]:.1f}")

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"{scroll}_L{pred_level}_c{cz}-{cy}-{cx}"
    colors = distinct_colors(max(n_final, 1))
    render(labels[dz], ct[dz], OUT_DIR / f"split_{tag}_before.png",
           distinct_colors(max(n_after, 1)))
    render(new_labels[dz], ct[dz], OUT_DIR / f"split_{tag}_after.png", colors)
    # zoom on the biggest mega instance for the money shot
    ys, xs = np.nonzero(labels[dz] == mega_ids[0])
    if len(ys):
        y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, CROP[1])
        x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, CROP[2])
        render(labels[dz, y0:y1, x0:x1], ct[dz, y0:y1, x0:x1],
               OUT_DIR / f"split_{tag}_zoom_before.png", distinct_colors(max(n_after, 1)))
        render(new_labels[dz, y0:y1, x0:x1], ct[dz, y0:y1, x0:x1],
               OUT_DIR / f"split_{tag}_zoom_after.png", colors)
    np.save(OUT_DIR / f"split_{tag}.npy", new_labels)
    print(f"saved renders + labels to {OUT_DIR}")


if __name__ == "__main__":
    main()
