# PHerc1218 spiral-fit sweep — per-window results

Fitted surfaces for both anchored bands of PHerc1218 (volume
20250521120456, 8.64 µm/voxel, grid 23247×7593×7593 zyx) — a Grand
Prize-eligible scroll with **zero human segments** — produced with the
unmodified official `fit_spiral.py` via this repo's
`reproduce/spiral_fit_window.py`, off the machine-generated spiral-fit
input pack (constraints from stitched instance labels).

Compute & replication: Paulo Sergio Camillo (pscamillo), RTX 5070
(sm_120), torch 2.11.0+cu128. Pack, method and runner: Iyán Dopico
(IyanDopico). Sweep executed 2026-07-23/24.

## Setup (frozen)

- Runner: `reproduce/spiral_fit_window.py`, `FIT_PACK_REF=7116a75`
  (25 per-slab seed patches), seed 1, 30k steps, PCL-only config,
  `unattached_pcl_num_per_step=800` (pre-scale; the fit z-scales it to
  67 effective strips/step on an 800-slice window).
- Windows: 800 slices, step 640 (160 overlap). Band A ≈ z 1344–10368
  full-res; Band B ≈ 15680–21120. Crushed middle (≈10368–15680) has no
  anchors and is out of scope here.
- Reading: rel/same split by `source_file` from `satisfied_fitted.json`;
  seed fractions from the same file; median dr per winding measured at
  3 z's (window center ±200) against the interpolated umbilicus, from
  `meshes/fitted_*/w*/` tifxyz (see `tools/`).

## Pass criteria (pre-registered; seed-satisfaction semantics agreed
with the pack author on 2026-07-24)

| metric | criterion |
|---|---|
| relative-winding points | ≥ 94% (where denominator ≥ 50 pts) |
| same-winding points | ≥ 91% (idem) |
| median dr per winding | 10.1 ± 0.8 L1 vox |
| final loss / exit | < 5–10 range, exit 0, ≥1 anchored slab in window |
| per-seed satisfaction | **recorded diagnostic, not a pass criterion** |

Rationale for the seed rule: per-slab synthesized seeds vary in
geometric tightness — the generator's cluster gates bound radial
spread at 8 L1 vox *per theta bin*, but nothing bounds arc-level
spread where source instances get noisy in the crushed direction.
Measured extreme (seed-z4256): the fitted winding passes at Δ+0.7 vox
(next winding −21.0 = exactly one pitch, so no misassignment), while
the seed's own radial spread is p10–p90 = 49 vox ≈ 2.5 pitches at a
single z. Seed satisfaction therefore measures seed tightness, not fit
quality.

## Results

**21 sweep windows + the published gate window. Exit 0 on 21/21.
19 windows validate on constraints+dr; 2 flagged.**

Aggregate over the 19 validated windows: relative 95.6–98.7%
(19/19 ≥ 94), same 92.1–100% (19/19 ≥ 91), median dr 9.48–10.64 L1
across 57 spot-checks (all within 10.1±0.8), final loss 0.1–7.2
(recorded per window as a range; one window not recorded), no
structural failures. Total sweep GPU time 175.1 min.

Every figure in this section is produced by `tools/aggregate.py` from
`windows.csv`; rerun it after any change to the table.

Verdict labels used in `windows.csv`:

| label | n | meaning |
|---|---|---|
| `full PASS` | 6 | gate passed; every seed patch ≥ 97% satisfied |
| `PASS` | 11 | gate passed; one or more seed patches < 97% |
| `PASS-with-note` | 2 | identical to `PASS` — see below |
| `PROVISIONAL` | 2 | not validated; listed below |

The gate is constraints+dr. The `full PASS` / `PASS` split records
seed-patch tightness, which per the seed rule above is a property of the
input pack, not of the fit — it is kept because it is informative about
the pack, not because it ranks the fits. `PASS-with-note` carries no
measured difference from `PASS`: those two windows were first recorded
FAIL under the stricter provisional seed criterion and reclassified on
2026-07-24 when the semantics were agreed; the label preserves that
provenance.

Flagged windows:
1. **1380–2180 (tip) — PROVISIONAL.** Relative constraints collapse to
   zero near the taper (5 pcls / 14 pts, all same-winding): the cone
   breaks the constraint geometry (expected physics per the pack
   author). Fit is anchored only by 2 mid-quality seeds + umbilicus.
2. **17600–18400 — PROVISIONAL.** Falls in an intra-band anchor gap
   (no slab between L1 z8288 and z9184); ran with **zero patches** due
   to a check-ordering gap in `fit_spiral.py` (see the upstream issue
   draft / villa issue). Constraints excellent (rel 98.5) but no
   absolute seed anchoring.

Physical trend recorded (not a fit artifact): median dr decreases
with z within windows and increases toward the base across Band B
(~10.2–10.6) vs Band A (~9.5–10.4) — consistent with real local pitch
variation; cross-check against the pack's per-ray pitch CSV pending.

Full per-window table: `windows.csv` (machine-readable) — columns
include per-seed fractions, rel/same numerators and denominators, dr
at 3 z's, wall-clock and verdicts.

## Surfaces

Per-window fitted meshes (both `fitted` and `fitted_spliced` series,
tifxyz per winding), `satisfied_fitted.json`, per-slice QA PNGs and
run metadata: Kaggle dataset **https://www.kaggle.com/datasets/pscamillo/pherc1218-spiral-fit-sweep-surfaces** (~2.1 GB; training
checkpoints withheld for size, available on request).

## Known caveats

- Main-pack (multi-seed) results are a separate series from the
  published single-seed band, per REPRODUCING.md — anchoring differs.
- Seed fractions over small window∩slab intersections (< ~50k vx²)
  are noisy; several windows demonstrate the same seed scoring
  differently on different portions.
- Windows must overlap ≥1 anchored slab; the practical log guard is
  `fitting N patches, N >= 1` (`loaded N patches` refers to the pack).
- `loss_final` is the observed range over the run, not a single value;
  one window (3940–4740) has no loss recorded (`n/r`).
