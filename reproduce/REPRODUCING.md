# Reproducing the PHerc1218 spiral fit

One command reproduces the published window fit end-to-end (download, patch,
fit, outputs):

```
python reproduce/spiral_fit_window.py
```

Requirements: CUDA GPU (an 800-slice window peaks well under 6 GB; a free
Kaggle T4 runs it in ~25 min at ~24 it/s), Python >= 3.11, internet, and:

```
pip install torch numpy scipy pillow tqdm einops kornia trimesh pyro-ppl torchdiffeq wandb "zarr==2.18.3" "numcodecs<0.16" fsspec aiohttp
```

(wandb stays offline — `WANDB_MODE=disabled` is set by the runner. The zarr
pin only matters if you later enable the lasagna inputs; for the PCL-only
config any importable zarr works.)

## What it runs

- villa `scripts/spiral` pinned to commit `61bd95c` of the
  [IyanDopico/villa fork](https://github.com/IyanDopico/villa/tree/fix-atlas-lookup-oob)
  = upstream `6e78421` plus the atlas-lookup bounds fix of
  [ScrollPrize/villa#1207](https://github.com/ScrollPrize/villa/pull/1207)
  (since merged upstream; the fork pin is kept because it is the exact code
  state the published runs used — without the fix, small-patch runs crash
  deterministically around step 1.2k).
- The [spiral_input_pherc1218 pack](../data/spiral_input_pherc1218/) from this
  repo (umbilicus + same/relative windings from the stitched 686k-instance
  labels + synthesized seed patches).
- Stable PCL-only config: dense/lasagna and shell losses off, stock dt
  schedule, `stratified_pcl_sampling` off, seed 1, 30k steps.

## Windows

`FIT_Z_BEGIN`/`FIT_Z_END` select the window (full-resolution voxels, 8.64 µm).
The published window is z 9700–10500. **A window must contain at least one
verified seed patch** — `fit_spiral.py` requires one (its umbilicus loss lives
inside the patch-loss path; see docs/constraints.md). The pack ships the
published window's patch (`seed-z4928-pherc1218`, full-res z 9888–10336), and
per-slab patches for the remaining scroll are being generated and added (ids
`seed-z{SLAB}-pherc1218`, full-res z ≈ 2×SLAB..2×(SLAB+256)) — the runner
downloads whatever the pack carries. To synthesize one for any slab yourself:
`scripts/constraints/make_seed_patch.py RUN_DIR PACK_DIR Z0` against the
[published dataset](https://www.kaggle.com/datasets/iyndopicomartnez/pherc1218-sheet-instance-labels).

## Expected results (band, not bit-parity)

Same-seed satisfaction moves several points across code versions and hardware;
our seed-variance probes on one code state put relative-winding satisfaction
at ±0.3 pt and same-winding at ±6 pt. Published runs on the z 9700–10500
window, 30k steps:

| run | relative-winding pts | same-winding pts | seed patch |
|---|---|---|---|
| pinned pre-refactor code, seed 1 | 97.8% | 98.4% | 100% |
| fork tip, seed 1 | 95.0% | 92.1% | 100% |
| fork tip, seed 2 | 95.3% | 98.1% | 100% |

A reproduction **passes** if: seed patch ≥ 97%, relative windings ≥ 94%,
same windings ≥ 91%, and the fitted spiral's median dr between consecutive
windings lands at 10.1 ± 0.5 L1 voxels (= the independently measured 173 µm
pitch). `satisfied_fitted.json` in the output directory carries the first
three; the winding meshes (`meshes/*/w*/`) carry the fourth.

## Honest caveats

- `spiral_outward_sense = 'CW'` was measured from one multi-turn stitched
  instance (weak evidence). The PCL constraints are mirror-symmetric, so a
  wrong sense only mirrors the parametrization — fixable at render time.
- The fit models the geometric spiral including voids: its winding count sits
  above the per-ray counted N by design ("don't count the air").
- Dense lasagna losses (official lasagna pack for this scroll) are NOT part of
  the stable config yet: upstream structurally gates them behind an outer
  shell, and unlocking them shifts the PCL numbers — active work, see the
  repo issues/thread.
