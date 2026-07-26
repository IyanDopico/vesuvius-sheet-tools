# results/pherc1218_sweep

At-scale sweep of the PHerc1218 spiral-fit input pack: 21 windows
across both anchored bands, run with this repo's
`reproduce/spiral_fit_window.py` unmodified (pack @ 7116a75, seed 1,
30k steps). 18 windows validate on constraints+dr; per-window table,
criteria and flagged cases in `RESULTS_en.md` / `windows.csv`.
Surfaces (~2.1 GB): https://www.kaggle.com/datasets/pscamillo/pherc1218-spiral-fit-sweep-surfaces.

`tools/` — the QA instruments the sweep shipped with:
- `window_report.py RUN_DIR` — per-seed fractions, rel/same split by
  source_file, median dr per winding at 3 z's (window center ±200,
  against the interpolated umbilicus), and a ready table row.
- `dr_multi_z.py RUN_DIR [z ...]` — standalone dr measurement
  (auto-derives the 3 z's from the run directory name).
- `sweep_band.sh Z1 Z2 ...` — sequences 800-slice windows, preserves
  full logs, accumulates the essentials into a single summary.

The dr band used (10.1 ± 0.8 L1 vox) reflects measured physical pitch
variation with z and theta; a window failing only on dr should be
cross-checked against the pack's per-ray pitch CSV before being called
a fit failure. The practical zero-patch guard: the log must show
`fitting N patches, N >= 1` (`loaded N patches` refers to the pack,
not the window).

Compute & replication: pscamillo. Pack, method, runner: IyanDopico.
