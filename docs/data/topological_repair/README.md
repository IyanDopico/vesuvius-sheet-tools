# Topological repair: site records for the v1 fusion population

Two CSVs documenting where the v1 dataset's fused sheet stacks are, and where a
continuity repair reassigned them. They sit here so the failure population is
documented next to the dataset it belongs to, and so `split_stacked` can be run
on exactly these coordinates for the site-by-site signal comparison in issue #1.

The repaired label volume itself is published separately as a Kaggle dataset
(CC-BY-NC, derived from the v1 labels):
https://www.kaggle.com/datasets/jhjeong0815/pherc1218-topological-repair

## repaired_sites.csv

14,131 sites (5,855 tier 1, 8,276 tier 2) where the repair reassigned a fused
column to neighbouring instances.

| column | meaning |
|---|---|
| `slab`, `tile` | which v1 tile the site is in |
| `gz`, `gy`, `gx` | scroll-global voxel coordinate of the flag point |
| `lz`, `ly`, `lx` | the same point in tile-local coordinates |
| `orig_id` | the fused instance id at the site in v1 |
| `assigned_A`, `assigned_B` | the two neighbour ids it was split between |
| `thickness_ratio` | column thickness over the local median, which is the flag |
| `confidence` | median absolute potential of the solve, see the note below |
| `tier` | 1 conservative (cap 40), 2 extended (cap 300) |
| `decision` | SPLIT |

Both coordinate systems are written out because the repair records are kept in
tile-local space while the diagnostic reports in scroll space. Global equals
local plus the tile's `(z0, y0, x0)`, so either set indexes the same voxel.

Tier 1 validates at 78.8% by ray recast, tier 2 at 62.1%, combined 69.0% over
all 14,131 sites.

### On `confidence`, which runs the opposite way to its name

It measures how decisive the solve was, not whether the site came out split.
Recasting the diagnostic ray at every site, the split rate by confidence
quartile is 76.8, 73.6, 71.2 and 54.4 percent, and the same inversion holds
inside each tier (Spearman -0.216 in tier 1, -0.174 in tier 2; mean confidence
0.342 and 0.341, so it is not a tier artifact). A decisive solve is one that
pushed the whole fused column onto a single neighbour, which leaves it
single-id. Sort ascending and keep the low end: dropping the top quarter takes
tier 1 from 78.8 to 84.1 percent, where dropping a random quarter leaves it at
78.8. Tier 2 remains the shakier half.

## fused_suspects.csv

9,716 rows from the diagnostic pass, a sampled subset of the flagged fused
population rather than the full set, with global coordinates, ids and thickness
ratios. These are candidates, not repairs; some fall below the repair's
confidence gate.

## Notes

- Coordinates are in the same L1 voxel grid as the v1 labels.
- The repair reassigns voxels between existing instance ids. It adds and removes
  nothing and mints no new id, so the mask and the instance count match v1.
- Method: https://github.com/Jinhojeong/vesuvius-unmerge
- Eval tooling: https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic

## tier3_sites.csv (batch 2, with pre-registered predictions)

The 40,246 tier-3 sites from the cap-lift pass, same columns as
repaired_sites.csv plus three:

| column | meaning |
|---|---|
| `scorable` | 1 if the site sat far enough from a tile face for feature extraction (34,135 of 40,246); 0 otherwise |
| `predicted_agreement` | mask-only model probability that this site would be `was_mega` under the baseline above, trained on the tier-1/2 verdicts (CV AUC 0.865); empty where `scorable` is 0 |
| `mass_context` | the crop-scale component volume-to-surface feature driving that model |

Tier 3 validates at 57.3% by ray recast, and the confidence inversion above
holds there too (Spearman -0.199).

The predictions are the point of this file. Two rival pre-registered
predictions for the tier-3 `was_mega` rate exist before any batch-2 run: the
tier-monotonicity argument says under 31.9 percent, the site-level model here
says 37.5 percent on the scorable subset (mean probability 0.376). Committing
the per-site numbers in this file is what makes the registration checkable
rather than claimed. When batch 2 runs, the comparison must be site-matched
against `scorable = 1` rows, since the two subsets differ.

## split_stacked_baseline_verdicts.csv (independent corroboration)

Per-site verdicts from an independent intensity/watershed segmentation
(split_stacked's baseline: clean -> EDT -> watershed -> merge, 128x512x512
crops) at all 14,131 repaired sites plus a +30-vox-y matched control.
`was_mega` = the point lands inside a fused mega-instance (>= 3% of the local
mask) that this unrelated method also flags. Result: flagged 37.2% vs control
12.9% (2.89x); tier 1 44.7% vs 15.2% (2.93x), tier 2 31.9% vs 11.2% (2.85x) —
the tier ordering matches the ray-recast validation ordering. Ran on a Kaggle
CPU kernel, 1,255 crops, density-ordered, full coverage.

## mega_coverage.csv (the converse measurement)

One row per fused mega-instance (>=3% of crop mask) with the count of flagged
sites inside it, over two populations: `site` = the 1,255 site-centred crops
of the baseline run (biased toward megas containing sites), `random` = 423
crops sampled across the scroll independent of the site list (the unbiased
arm). Random population: 77.1% of megas contain zero flagged sites; zero-count
megas are ~4x smaller (median 91k vs 374k vox) and the zero fraction falls
96.1/89.6/79.1/43.6% across volume quartiles — most of the gap is sampling,
with the top-quartile 43.6% as the upper bound on structure the ray sample
misses at sizes it should reach.
