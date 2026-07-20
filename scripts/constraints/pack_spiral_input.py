"""Pack generated PHerc1218 spiral-fit constraint files into the official
spiral-input dataset layout (as documented in spiral-input-PHercParis4-README).

Collects whichever constraint files exist in <in_dir> (sibling tools may not
have produced all of them yet), sanity-checks each JSON, and assembles:

    <out>/
        umbilicus.json            (copied if present and valid)
        same_windings.json        (copied if present and valid)
        relative_windings.json    (copied if present and valid)
        abs_winding.json          (created empty-but-valid if absent)
        verified_patches/         (empty, with placeholder note)
        unverified_patches/       (empty, with placeholder note)
        README.txt                (provenance + per-file counters)

Usage:
    python scripts/constraints/pack_spiral_input.py <in_dir> \
        [--out spiral_input_pherc1218] [--force] [--date YYYY-MM-DD]

Exit codes: 0 = packed umbilicus + at least one constraints file,
            2 = not enough valid inputs to make a usable pack.
Never modifies <in_dir>. Refuses --out inside <in_dir> unless --force.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCROLL_VOLUME_ID = "20250521120456-8.640um-1.2m-116keV"
VOXEL_SIZE_UM = 8.64
KAGGLE_URL = "https://www.kaggle.com/datasets/iyndopicomartnez/pherc1218-sheet-instance-labels"
PC_VERSION_KEY = "vc_pointcollections_json_version"

# file name in in_dir -> required top-level keys
CONSTRAINT_FILES = {
    "umbilicus.json": ("control_points",),
    "same_windings.json": (PC_VERSION_KEY, "collections"),
    "relative_windings.json": (PC_VERSION_KEY, "collections"),
}
OPTIONAL_EXTRA_FILES = {
    "abs_winding.json": (PC_VERSION_KEY, "collections"),  # created empty if absent
    "patch_candidates.json": (),  # expected to be a JSON list
}
EMPTY_POINT_COLLECTION = {PC_VERSION_KEY: "1", "collections": {}}
EMPTY_DIRS = ("verified_patches", "unverified_patches")
PLACEHOLDER_NOTE = (
    "Placeholder so this directory exists in the packed dataset.\n"
    "Patches produced by the spiral-fit pipeline go here.\n"
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="pack_spiral_input.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("in_dir", help="directory with generated constraint JSONs")
    ap.add_argument("--out", default="spiral_input_pherc1218",
                    help="output dataset directory (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out / allow --out inside in_dir")
    ap.add_argument("--date", default=None,
                    help="generation date to record in README.txt "
                         "(default: newest input file mtime, printed)")
    return ap.parse_args(argv)


def load_json(path: Path) -> tuple[object | None, str | None]:
    """Return (parsed, None) or (None, error string)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def check_keys(data: object, required: tuple[str, ...]) -> str | None:
    """Return an error string if `data` lacks the expected shape, else None."""
    if not required:  # patch_candidates.json: expect a list
        return None if isinstance(data, list) else "expected a JSON list"
    if not isinstance(data, dict):
        return "expected a JSON object"
    missing = [k for k in required if k not in data]
    if missing:
        return f"missing top-level keys: {', '.join(missing)}"
    if "control_points" in required and not isinstance(data["control_points"], list):
        return "control_points is not a list"
    if "collections" in required and not isinstance(data["collections"], dict):
        return "collections is not an object"
    return None


def counters(name: str, data: object) -> str:
    """Human-readable content counters for the README."""
    if name == "umbilicus.json":
        pts = data.get("control_points", [])
        return f"{len(pts)} control points"
    if name == "patch_candidates.json":
        return f"{len(data)} patch candidates"
    cols = data.get("collections", {})
    n_points = 0
    for col in cols.values():
        if isinstance(col, dict):
            pts = col.get("points", col)
            n_points += len(pts) if isinstance(pts, (list, dict)) else 0
        elif isinstance(col, list):
            n_points += len(col)
    return f"{len(cols)} collections, {n_points} points"


def resolve_date(explicit: str | None, packed_paths: list[Path]) -> str:
    """Generation date: --date wins; otherwise newest input mtime (UTC)."""
    if explicit:
        return explicit
    if packed_paths:
        mtime = max(p.stat().st_mtime for p in packed_paths)
        stamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return stamp.strftime("%Y-%m-%d") + " (from input file mtimes)"
    return "unknown (no inputs packed, no --date given)"


def write_readme(out: Path, packed: dict[str, str], skipped: dict[str, str],
                 missing: list[str], created: list[str], date_str: str) -> None:
    lines = [
        "PHerc1218 spiral-fit input pack",
        "=" * 31,
        "",
        "Provenance",
        "----------",
        "Generated by vesuvius-sheet-tools (scripts/constraints/pack_spiral_input.py)",
        "from the PHerc1218 sheet-instance dataset:",
        f"  {KAGGLE_URL}",
        f"Scroll volume id: {SCROLL_VOLUME_ID} (full resolution, {VOXEL_SIZE_UM} um/voxel)",
        "Coordinate convention: points are p = [x, y, z] in full-resolution voxels.",
        f"Generation date: {date_str}",
        "",
        "Contents",
        "--------",
    ]
    for name, info in packed.items():
        lines.append(f"  {name}: {info}")
    for name in created:
        lines.append(f"  {name}: created empty (valid placeholder, no data yet)")
    lines += [
        "  verified_patches/: empty (placeholder)",
        "  unverified_patches/: empty (placeholder)",
    ]
    if skipped:
        lines += ["", "Skipped inputs (corrupt or wrong shape)", "-" * 38]
        for name, why in skipped.items():
            lines.append(f"  {name}: {why}")
    if missing:
        lines += ["", "Missing pieces (not present in the input directory)", "-" * 50]
        for name in missing:
            lines.append(f"  {name}")
    lines.append("")
    (out / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir).resolve()
    out = Path(args.out).resolve()

    if not in_dir.is_dir():
        print(f"ERROR: input directory not found: {in_dir}")
        return 2
    if out == in_dir or in_dir in out.parents:
        if not args.force:
            print(f"ERROR: --out {out} is inside the input directory; "
                  "refusing without --force")
            return 2
        print(f"WARNING: --out {out} is inside the input directory (--force)")
    if out.exists() and not out.is_dir():
        print(f"ERROR: --out {out} exists and is not a directory")
        return 2
    if out.is_dir() and any(out.iterdir()):
        if not args.force:
            print(f"ERROR: output directory {out} exists and is not empty; "
                  "use --force to overwrite")
            return 2
        print(f"WARNING: overwriting existing output directory {out}")
        # Remove the layout files we manage so a stale copy from a previous
        # run can't survive alongside a README that no longer mentions it.
        for name in list(CONSTRAINT_FILES) + list(OPTIONAL_EXTRA_FILES) + ["README.txt"]:
            stale = out / name
            if stale.is_file():
                stale.unlink()

    out.mkdir(parents=True, exist_ok=True)
    for d in EMPTY_DIRS:
        (out / d).mkdir(exist_ok=True)
        (out / d / "PLACEHOLDER.txt").write_text(PLACEHOLDER_NOTE, encoding="utf-8")

    packed: dict[str, str] = {}     # name -> counters text
    skipped: dict[str, str] = {}    # name -> reason
    missing: list[str] = []
    packed_paths: list[Path] = []

    all_inputs = {**CONSTRAINT_FILES, **OPTIONAL_EXTRA_FILES}
    for name, required in all_inputs.items():
        src = in_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        data, err = load_json(src)
        if err is None:
            err = check_keys(data, required)
        if err is not None:
            print(f"WARNING: skipping {name}: {err}")
            skipped[name] = err
            continue
        shutil.copy2(src, out / name)
        packed[name] = counters(name, data)
        packed_paths.append(src)

    created: list[str] = []
    if "abs_winding.json" not in packed:
        abs_path = out / "abs_winding.json"
        abs_path.write_text(json.dumps(EMPTY_POINT_COLLECTION, indent=2) + "\n",
                            encoding="utf-8")
        created.append("abs_winding.json")
        if "abs_winding.json" in missing:
            missing.remove("abs_winding.json")  # we created it; not a gap

    date_str = resolve_date(args.date, packed_paths)
    write_readme(out, packed, skipped, missing, created, date_str)

    print(f"\nPacked spiral input -> {out}")
    print(f"Generation date recorded: {date_str}")
    print(f"{'file':<28} {'status':<10} details")
    print("-" * 70)
    for name in all_inputs:
        if name in packed:
            print(f"{name:<28} {'packed':<10} {packed[name]}")
        elif name in skipped:
            print(f"{name:<28} {'SKIPPED':<10} {skipped[name]}")
        elif name in created:
            print(f"{name:<28} {'created':<10} empty valid placeholder")
        else:
            print(f"{name:<28} {'missing':<10} not present in input dir")
    for d in EMPTY_DIRS:
        print(f"{d + '/':<28} {'created':<10} empty (placeholder)")

    have_umbilicus = "umbilicus.json" in packed
    have_constraints = any(n in packed for n in ("same_windings.json",
                                                 "relative_windings.json"))
    if have_umbilicus and have_constraints:
        return 0
    print("\nERROR: incomplete pack: need umbilicus.json plus at least one of "
          "same_windings.json / relative_windings.json (valid).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
