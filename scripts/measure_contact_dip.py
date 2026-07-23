"""Measure the CT intensity profile across sheet-contact sites (PHerc1218).

Tests Jinhojeong's no-dip prediction (villa #191: raw CT shows no intensity
dip between contacting sheets - median dip -4.5 grey levels on Dataset059,
0% of pairs above a quarter of local contrast) on PHerc1218, at flagged
contact sites - e.g. the per-site list of the ADL-RW repair
(IyanDopico/vesuvius-sheet-tools#1) or any CSV with site coordinates.

Per site: streams a small L1 CT crop from the open S3 bucket, estimates the
local sheet normal via structure tensor (same sigmas as split_stacked, on the
raw and CLAHE'd crop), samples the intensity profile +-PROFILE_HALF voxels
along the normal, and reports the center-vs-shoulder dip in grey levels and
as a fraction of local contrast (p90-p10 of the profile), for both raw and
CLAHE intensity - the disambiguation signal the CLAHE splitter relies on.

Usage:
  python scripts/measure_contact_dip.py SITES_CSV [OUT_CSV]
    SITES_CSV columns: any of (gz,gy,gx) [full-res voxels] or (z,y,x) [L1].
  python scripts/measure_contact_dip.py --selftest
    runs on a handful of hardcoded crushed-interior points.
"""
import csv
import sys

import numpy as np
import zarr
from scipy import ndimage
from skimage.exposure import equalize_adapthist

CT_URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1218/"
          "volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr")
CT_LEVEL = "1"          # L1 = 17.28 um/vox, matches the instance labels
CROP_HALF = 24          # 48^3 crop per site
PROFILE_HALF = 8        # +-8 vox along the normal (~138 um)
SIGMA_GRAD, SIGMA_TENSOR = 1.2, 2.5
SHOULDER = (4, 7)       # shoulder band along the profile (vox from center)

SELFTEST_SITES_L1 = [   # crushed interior, near the known fused-stack crop
    (5812, 2320, 860), (5812, 2290, 900), (5780, 2350, 820),
    (5840, 2300, 940), (5812, 2260, 780),
]


def profile_stats(vol, center, normal):
    t = np.arange(-PROFILE_HALF, PROFILE_HALF + 1, 0.5)
    pts = center[:, None] + normal[:, None] * t[None, :]
    prof = ndimage.map_coordinates(vol, pts, order=1, mode="nearest")
    lo, hi = np.percentile(prof, [10, 90])
    contrast = max(hi - lo, 1e-6)
    c0 = PROFILE_HALF * 2  # index of t=0 (0.5 step)
    center_val = float(prof[c0 - 1:c0 + 2].mean())
    sh = np.concatenate([
        prof[c0 - SHOULDER[1] * 2:c0 - SHOULDER[0] * 2],
        prof[c0 + SHOULDER[0] * 2:c0 + SHOULDER[1] * 2],
    ])
    shoulder_val = float(np.median(sh))
    dip = shoulder_val - center_val  # positive = darker valley at the contact
    return dip, dip / contrast, contrast


def measure(sites_l1):
    arr = zarr.open(CT_URL, mode="r")[CT_LEVEL]
    rows = []
    for (z, y, x) in sites_l1:
        z0, y0, x0 = (int(z - CROP_HALF), int(y - CROP_HALF), int(x - CROP_HALF))
        crop = np.asarray(arr[z0:z0 + 2 * CROP_HALF,
                              y0:y0 + 2 * CROP_HALF,
                              x0:x0 + 2 * CROP_HALF]).astype(np.float32)
        if crop.size == 0 or crop.max() <= 0:
            rows.append((z, y, x, *[np.nan] * 5))
            continue
        eq = np.stack([equalize_adapthist(s / max(crop.max(), 1), kernel_size=24,
                                          clip_limit=0.02)
                       for s in crop]).astype(np.float32)
        g = [ndimage.gaussian_filter(ndimage.sobel(
            ndimage.gaussian_filter(crop, SIGMA_GRAD), axis=a), SIGMA_TENSOR)
            for a in range(3)]
        c = np.array([CROP_HALF] * 3, dtype=float)
        J = np.zeros((3, 3))
        sl = tuple(slice(CROP_HALF - 4, CROP_HALF + 5) for _ in range(3))
        for a in range(3):
            for b in range(3):
                J[a, b] = float((g[a][sl] * g[b][sl]).mean())
        w, v = np.linalg.eigh(J)
        normal = v[:, -1]  # largest-eigenvalue direction = across sheets
        dip_r, dipc_r, con_r = profile_stats(crop, c, normal)
        dip_e, dipc_e, _ = profile_stats(eq, c, normal)
        rows.append((z, y, x, dip_r, dipc_r, con_r, dip_e, dipc_e))
    return rows


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sites = SELFTEST_SITES_L1
        out = None
    else:
        src = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else None
        sites = []
        for r in csv.DictReader(open(src, newline="")):
            if "gz" in r and r.get("gz"):
                sites.append((float(r["gz"]) / 2, float(r["gy"]) / 2,
                              float(r["gx"]) / 2))
            else:
                sites.append((float(r["z"]), float(r["y"]), float(r["x"])))
    rows = measure(sites)
    hdr = ["z_l1", "y_l1", "x_l1", "dip_raw_grey", "dip_raw_over_contrast",
           "contrast_raw", "dip_clahe", "dip_clahe_over_contrast"]
    if out:
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            w.writerows(rows)
    a = np.array([r[3:] for r in rows], dtype=float)
    ok = ~np.isnan(a[:, 0])
    print(f"{ok.sum()}/{len(rows)} sites measured")
    if ok.any():
        print(f"raw dip: median {np.nanmedian(a[ok, 0]):+.1f} grey "
              f"({np.nanmedian(a[ok, 1]) * 100:+.0f}% of local contrast); "
              f">=25% of contrast: {(a[ok, 1] >= 0.25).mean() * 100:.0f}%")
        print(f"CLAHE dip: median {np.nanmedian(a[ok, 3]) * 100:+.1f}% "
              f"(units of equalized intensity); "
              f">=25% of contrast: {(a[ok, 4] >= 0.25).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
