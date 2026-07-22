"""Patch-mode entry point: OFF/ON sheet-instance splitting on a GT surface patch.

Consumes the patch layout of Jinhojeong's vesuvius-surface-geometry-diagnostic
(release patches-v1; see IyanDopico/vesuvius-sheet-tools#1):

  imagesTr/{scroll}_z{Z}_y{Y}_x{X}_0000.tif   uint8 raw CT, zyx
  labelsTr/{scroll}_z{Z}_y{Y}_x{X}.tif        uint8 binary GT surface, same grid

and produces {stem}_off.npz / {stem}_on.npz (int32 'labels'; the nonzero mask
is exactly the GT mask in both) plus {stem}_params.json documenting the
calibration. Intended consumer: eval_patch.py in that same diagnostic repo
(scripts/eval_patch.py there), via --pred <off.npz> --pred-b <on.npz>.

OFF = EDT watershed + calibrated salience merge (separate_sheets.py); ON = the
same, then the CLAHE -> structure tensor -> ridge NMS -> orientation-guided
consolidation -> intensity watershed splitter (split_stacked.py) applied to
mega-instances. The GT mask is never filtered: components that get no watershed
seed keep their own connected-component id, so mask-level metrics compare the
same voxels and only the instance decomposition differs between OFF and ON.

Self-calibration: every voxel-unit parameter was calibrated on PHerc1218 at L1
(17.28 um/vox), where the median sheet half-thickness - EDT p90 inside the
mask - is ~1.2 vox. We measure that statistic on the GT patch and scale
lengths by s = ht/1.2, areas by s^2, volumes by s^3. No per-scroll voxel-size
table required.

Usage: python scripts/split_patch.py CT_TIF GT_TIF OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.exposure import equalize_adapthist
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import split_stacked
from separate_sheets import merge_instances
from split_stacked import consolidate_centerlines, structure_normals

HT_REF = 1.2          # L1-PHerc1218 median sheet half-thickness (EDT p90)
MEGA_FRAC = split_stacked.MEGA_FRAC
P_MIN = split_stacked.P_MIN


def main() -> None:
    ct_path, gt_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(gt_path).stem

    ct = tifffile.imread(ct_path)
    gt = tifffile.imread(gt_path) > 0
    assert ct.shape == gt.shape, f"grid mismatch {ct.shape} vs {gt.shape}"
    print(f"{stem}: grid {gt.shape}, GT voxels {int(gt.sum()):,}", flush=True)

    # ---- self-calibration from the GT's own sheet thickness
    dist = ndimage.distance_transform_edt(gt).astype(np.float32)
    ht = float(np.percentile(dist[gt], 90))
    s = float(np.clip(ht / HT_REF, 0.5, 8.0))
    print(f"half-thickness p90 = {ht:.2f} vox -> scale s = {s:.2f}", flush=True)

    # Core threshold from the depth distribution itself: on the L1-PHerc1218
    # reference, CORE_DIST=2.0 sits at ~p97 of in-mask EDT. Scaling 2.0 by s
    # overshoots on thin single-sheet GT bands (p90 depth < 2s -> zero seeds),
    # so we take the distribution's own p97, bounded by the naive scaling.
    core_dist = float(min(np.percentile(dist[gt], 97), 2.0 * s))
    min_core = max(int(round(60 * s ** 2)), 8)  # cores in thin bands are
    # pancake-shaped: area scaling, not volume
    min_border = max(int(round(20 * s ** 2)), 4)
    max_border = max(int(round(60 * s ** 2)), 12)
    clahe_kernel = max(int(round(64 * s)), 16)
    sigma_int = 1.0 * s
    nms_step = 1.5 * s
    min_center_cc = max(int(round(100 * s ** 2)), 20)
    # module constants read by the imported helpers
    split_stacked.SIGMA_GRAD = 1.2 * s
    split_stacked.SIGMA_TENSOR = 2.5 * s
    split_stacked.D_MAX = 4.0 * s

    # ---- OFF: EDT watershed + calibrated merge, full GT coverage
    t0 = time.time()
    cores, _ = ndimage.label(dist >= core_dist)
    core_sizes = np.bincount(cores.ravel())
    cores = np.where((core_sizes >= min_core)[cores], cores, 0)
    ids = np.unique(cores)
    remap = np.zeros(cores.max() + 1, dtype=np.int32)
    remap[ids] = np.arange(len(ids))
    labels = watershed(-dist, remap[cores], mask=gt)
    # coverage fallback: GT components with no seed keep their own id
    leftover = gt & (labels == 0)
    if leftover.any():
        extra, n_extra = ndimage.label(leftover)
        labels = labels + np.where(extra > 0, extra + labels.max(), 0)
        print(f"coverage fallback: {n_extra} seedless components kept",
              flush=True)
    labels, n_before, n_after, _ = merge_instances(
        labels.astype(np.int32), dist,
        min_border=min_border, max_border=max_border)
    assert bool(((labels > 0) == gt).all()), "OFF mask must equal GT mask"
    off = labels.astype(np.int32)
    print(f"OFF: {n_before} -> {n_after} instances after merge "
          f"({time.time() - t0:.1f}s)", flush=True)

    # ---- ON: CLAHE splitter on mega-instances (split_stacked recipe)
    inst_sizes = np.bincount(off.ravel())
    mask_total = int(gt.sum())
    mega_ids = [i for i in range(1, len(inst_sizes))
                if inst_sizes[i] >= MEGA_FRAC * mask_total]
    print(f"mega-instances (>{MEGA_FRAC:.0%}): "
          f"{[(i, int(inst_sizes[i])) for i in mega_ids]}", flush=True)
    on = off.copy()
    n_final = int(on.max())
    if mega_ids:
        t0 = time.time()
        eq = np.empty(ct.shape, dtype=np.float32)
        for z in range(ct.shape[0]):
            eq[z] = equalize_adapthist(
                ct[z], kernel_size=clahe_kernel,
                clip_limit=split_stacked.CLAHE_CLIP).astype(np.float32)
        normals, planarity = structure_normals(eq)
        eq_s = ndimage.gaussian_filter(eq, sigma_int)
        mega_mask = np.isin(off, mega_ids)
        vz, vy, vx = np.nonzero(mega_mask)
        coords = np.stack([vz, vy, vx]).astype(np.float32)
        nrm = normals[vz, vy, vx].T
        val = eq_s[vz, vy, vx]
        up = ndimage.map_coordinates(eq_s, coords + nms_step * nrm, order=1,
                                     mode="nearest")
        dn = ndimage.map_coordinates(eq_s, coords - nms_step * nrm, order=1,
                                     mode="nearest")
        is_ridge = (val > up) & (val > dn) & (planarity[vz, vy, vx] >= P_MIN)
        centerline = np.zeros(gt.shape, dtype=bool)
        centerline[vz[is_ridge], vy[is_ridge], vx[is_ridge]] = True
        center_lab, n_center = ndimage.label(centerline)
        coords_r = coords.T[is_ridge]
        normals_r = nrm.T[is_ridge]
        labels_r = center_lab[vz[is_ridge], vy[is_ridge], vx[is_ridge]]
        lut = consolidate_centerlines(center_lab, coords_r, normals_r,
                                      labels_r)
        center_lab = lut[center_lab]
        c_sizes = np.bincount(center_lab.ravel())
        center_lab = np.where((c_sizes >= min_center_cc)[center_lab],
                              center_lab, 0)
        assigned = watershed(-eq_s, center_lab, mask=mega_mask)
        split_zone = mega_mask & (assigned > 0)
        on[split_zone] = int(off.max()) + assigned[split_zone]
        _, compact = np.unique(on, return_inverse=True)
        on = compact.reshape(on.shape).astype(np.int32)
        n_final = int(on.max())
        print(f"ON: centerlines {n_center} -> {int(center_lab.max())} kept; "
              f"{n_after} -> {n_final} instances ({time.time() - t0:.1f}s)",
              flush=True)
    else:
        print("ON: no mega-instances; ON == OFF", flush=True)
    assert bool(((on > 0) == gt).all()), "ON mask must equal GT mask"

    np.savez_compressed(out_dir / f"{stem}_off.npz", labels=off)
    np.savez_compressed(out_dir / f"{stem}_on.npz", labels=on)
    off_sizes = np.bincount(off.ravel())[1:]
    on_sizes = np.bincount(on.ravel())[1:]
    params = {
        "stem": stem, "grid": list(gt.shape),
        "gt_voxels": mask_total,
        "half_thickness_p90_vox": ht, "scale": s,
        "core_dist": core_dist, "min_core": min_core,
        "min_border": min_border, "max_border": max_border,
        "clahe_kernel": clahe_kernel, "sigma_grad": split_stacked.SIGMA_GRAD,
        "sigma_tensor": split_stacked.SIGMA_TENSOR, "sigma_int": sigma_int,
        "nms_step": nms_step, "min_center_cc": min_center_cc,
        "d_max": split_stacked.D_MAX,
        "instances_off": int(off.max()), "instances_on": n_final,
        "largest_share_off": float(off_sizes.max() / mask_total),
        "largest_share_on": float(on_sizes.max() / mask_total),
        "mega_ids": [int(i) for i in mega_ids],
    }
    json.dump(params, open(out_dir / f"{stem}_params.json", "w"), indent=2)
    print(f"SELF-CHECKS: PASS - masks identical to GT; "
          f"OFF {params['instances_off']} / ON {params['instances_on']} "
          f"instances; largest {params['largest_share_off']:.1%} -> "
          f"{params['largest_share_on']:.1%}", flush=True)


if __name__ == "__main__":
    main()
