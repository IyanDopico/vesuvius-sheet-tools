# Winding constraints for spiral fitting

Tools that turn the [PHerc1218 sheet-instance
labels](https://www.kaggle.com/datasets/iyndopicomartnez/pherc1218-sheet-instance-labels)
into **sparse winding constraints** for the official spiral-fitting pipeline.

> **Interface stability**: the scripts live in `scripts/constraints/` and are
> under active development. This page documents the *contract* — inputs,
> outputs, file formats and filtering guarantees. Exact flags may evolve;
> `--help` on each script is authoritative.

## Why

The endorsed path from surface predictions to per-sheet winding numbers
(tifxyz windings) is the official spiral fit
([tutorial](https://scrollprize.org/tutorial_spiral)): given an umbilicus and
a set of sparse winding annotations, `fit_spiral.py` solves for a global
spiral that assigns a winding number to every sheet point. The annotations it
consumes are point collections saying "these points lie on the same wrap" and
"these two groups of points lie on adjacent wraps".

Producing those annotations is currently a manual job — it is an open problem
([winding annotations](https://scrollprize.org/open_problems/winding_annotations)).
Our whole-scroll instance labels already encode exactly that information:
points inside one instance share a wrap, and radially adjacent instances are
candidate consecutive wraps. These tools extract it automatically, filter it
conservatively, and pack it in the official input layout — to our knowledge
the first such machine-generated spiral input pack for PHerc1218, a scroll
with no human segments to bootstrap from.

## Toolchain

| Script | Arguments | Output |
|---|---|---|
| `make_umbilicus.py` | `<run_dir> [--slabs] [--out]` | `umbilicus.json` |
| `make_same_windings.py` | `<run_dir> [--slabs] [--out] [--budget]` | `same_windings.json` + `patch_candidates.json` |
| `make_relative_windings.py` | `<run_dir> [--slabs] [--out]` | `relative_windings.json` |
| `validate_constraints.py` | `<file_or_dir> [--blocks] [--roundtrip N] [--selftest]` | validation report (schema + trap checks) |
| `pack_spiral_input.py` | `<in_dir> [--out]` | `spiral_input_pherc1218/` (official layout) |
| `render_constraints.py` | `<run_dir> --files ... [--umbilicus] [--slabs] [--out]` | QA PNGs (constraints over CT slices) |

`<run_dir>` is an assembled whole-scroll run directory (the layout produced by
`assemble_scroll.py` / published in the Kaggle dataset: `blocks/z*/tile_*.npz`
tiles, per-slab stitch tables, `global_table.json`). `--slabs` restricts
processing to a z-slab subset; `--budget` caps the number of emitted
same-winding collections.

`patch_candidates.json` is an auxiliary output: instance pairs that look like
fragments of one wrap but did not pass every filter. They are recorded for
review and are **not** part of the packed spiral input.

## File formats

### Point collections (`vc_pointcollections_json` v1)

`same_windings.json` and `relative_windings.json` both use the official
point-collection schema (see `point_collection.py` in
[ScrollPrize/villa](https://github.com/ScrollPrize/villa),
`volume-cartographer/scripts/spiral/`):

```json
{
  "vc_pointcollections_json_version": "1",
  "collections": {
    "<collection_id>": {
      "name": "<string>",
      "points": {
        "<point_id>": {
          "p": [x, y, z],
          "wind_a": 0.0,
          "creation_time": 0
        }
      },
      "metadata": {},
      "color": [r, g, b]
    }
  }
}
```

`wind_a` is optional per point — but see trap 3 below.

Three properties of the official loader are easy to get wrong, and all three
fail *silently* (no error, just wrong or missing points):

1. **`p` is `[x, y, z]` order.** The official loader reverses it to zyx
   internally. Writing zyx yourself places every point at a transposed
   location with no warning.
2. **Coordinates are full-resolution voxels.** Our labels are L1
   (half-resolution), so every coordinate is multiplied by 2 before writing.
   PHerc1218 full-resolution dimensions: z = 23247, y = x = 7593, at
   8.64 µm/voxel.
3. **Collections must be homogeneous in `wind_a`.** Within one collection,
   either every point carries `wind_a` or none does. In mixed collections the
   official loader strips points, so part of the constraint quietly vanishes.

`validate_constraints.py` checks all three (plus schema conformance), and
`--roundtrip N` re-reads N sampled points through the official loader
conventions to confirm they land where intended.

### `umbilicus.json`

```json
{"control_points": [{"z": 0, "y": 0, "x": 0}]}
```

A list of control points tracing the scroll's central axis, again in
full-resolution voxel coordinates; the official `umbilicus.py` interpolates
between them along z.

### Constraint semantics

- **`same_windings.json`** — collections *without* `wind_a`: all points in a
  collection lie on the same physical wrap, winding number unknown.
- **`relative_windings.json`** — *pairs* of collections *with* `wind_a`: the
  inner collection gets `0.0` and the outer `1.0`, meaning "the outer group is
  exactly one wrap outside the inner group". The values are relative anchors
  for the pair, not absolute winding numbers.

## Anti-poison design

`fit_spiral.py` solves a global problem, so a wrong constraint is worse than
a missing one — a single bad adjacent-wrap pair can bend the fit far from
where it was emitted. Generation is therefore deliberately conservative:
anything ambiguous is dropped (or diverted to `patch_candidates.json`), and
every emitted constraint passes independent geometric checks.

**Relative (adjacent-wrap) pairs:**

| Filter | Value | Rationale |
|---|---|---|
| Gap validation | radial gap ∈ [0.6, 1.5] × smoothed local pitch | Local pitch comes from our pitch QA (whole-scroll median 10.0 L1 vox = 173 µm). A gap below 0.6× means the "pair" is likely one split sheet; above 1.5× a wrap probably sits undetected between the two — either way the offset-1 claim would be false. |
| Ray support | ≥ 3 rays | The pair must be observed as radially adjacent along at least 3 independent rays; one-off adjacencies are noise-prone. |
| Orientation consistency | reject contradictions | If the inner/outer ordering flips between supporting rays, the pair is dropped. |
| Offset-2 pairs | never emitted | Only adjacent (offset-1) pairs are produced. Larger offsets add little signal to the fit but multiply exposure to miscounted intermediate wraps. |

**Same-winding collections** (which instances qualify as trustworthy
single-wrap sources):

| Filter | Value | Rationale |
|---|---|---|
| Instance size | ≥ 2000 vox | Small fragments have unreliable geometry. |
| Thickness proxy | ≤ 8 | Volume / centerline voxels ≈ mean sheet thickness; genuine single sheets at L1 measure 4–8 (see README), higher indicates a fused stack that may span wraps. |
| Mega-instances | excluded | Known fused stacks spanning many wraps must not contribute "same wrap" points. |
| Radial spread | ≤ 0.7 × local pitch | Points of one instance must stay within a fraction of the wrap-to-wrap distance radially; a wider spread suggests the label crosses wraps. |

**Point sampling** (per accepted instance): K = clamp(volume / 1500, 3, 20)
points, sampled where the Euclidean distance transform is ≥ 2 (deep interior,
never the ambiguous surface), with a minimum spacing of 5 voxels between
points.

## Pipeline

```bash
cd scripts/constraints

python make_umbilicus.py         output/scroll_run   # → umbilicus.json
python make_same_windings.py     output/scroll_run   # → same_windings.json (+ patch_candidates.json)
python make_relative_windings.py output/scroll_run   # → relative_windings.json

python validate_constraints.py   output/constraints  # schema + trap checks + roundtrip
python render_constraints.py     output/scroll_run --files output/constraints/*.json  # QA PNGs

python pack_spiral_input.py      output/constraints  # → spiral_input_pherc1218/
```

Generate, validate, eyeball the renders, pack. The next phase — running the
official `fit_spiral.py` on the pack to obtain tifxyz windings — is not part
of this toolchain yet.

## Known limitations

- **Constraints inherit stitching errors.** Cross-slab instance agreement is
  82.1% mean (see README); a same-winding collection whose instance was
  mis-stitched across a slab boundary can join two different physical wraps.
  The filters above reduce this risk, they do not eliminate it.
- **Fragments give only locally-scoped information.** A wrap fragmented into
  several instances yields several small same-winding collections instead of
  one long-range one, so the fit gets no "these two distant points share a
  wrap" signal from fragmented regions.
- **No absolute winding numbers yet.** Relative pairs anchor adjacency only;
  absolute winding assignment is left to the spiral fit itself (umbilicus +
  spiral model). An `abs_winding`-style anchoring pass may come later.

## References

- Spiral fitting tutorial — <https://scrollprize.org/tutorial_spiral>
- Winding annotations open problem — <https://scrollprize.org/open_problems/winding_annotations>
- Official implementation — [ScrollPrize/villa](https://github.com/ScrollPrize/villa),
  `volume-cartographer/scripts/spiral/` (`point_collection.py`,
  `umbilicus.py`, `fit_spiral.py`)
- Input labels — [PHerc1218 sheet-instance labels on Kaggle](https://www.kaggle.com/datasets/iyndopicomartnez/pherc1218-sheet-instance-labels)
