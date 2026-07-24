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
