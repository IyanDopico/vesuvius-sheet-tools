"""Strict validator for official Vesuvius spiral-fit constraint files.

Validates point-collection JSONs (vc_pointcollections_json_version "1") and
umbilicus.json files generated from our PHerc1218 instance labels, enforcing
the exact schema of the official ScrollPrize/villa loaders:

  * point collection: top key "vc_pointcollections_json_version" must be the
    STRING "1"; "collections" maps strkey -> {name, points, [metadata], [color]};
    each point has "p" = [x, y, z] (3 finite numbers, XYZ order, FULL-RES
    voxels), optional "wind_a" number, optional "creation_time". Within one
    collection either ALL points carry wind_a or NONE (the official loader
    strips mixed collections, so mixed is an ERROR here).
  * umbilicus: {"control_points": [{"z", "y", "x"}, ...]}, >= 2 points.
  * bounds (PHerc1218 full-res): z in [0, 23247), y/x in [0, 7593).

With --blocks it additionally round-trips a sample of points from collections
named "inst_<globalid>" back into the half-res slab/tile label blocks and
checks the stitched GLOBAL id matches; hit rate below 90% is an ERROR.

Usage:
  python scripts/constraints/validate_constraints.py <file_or_dir>
      [--blocks output/scroll_run] [--roundtrip 100] [--report validation_report.json]
  python scripts/constraints/validate_constraints.py --selftest

Writes validation_report.json (per-file errors/warnings/stats) and exits 0
only if no file has errors.
"""

import argparse
import json
import math
import random
import re
import sys
import tempfile
from pathlib import Path

# PHerc1218 full-resolution volume bounds (exclusive).
FULL_Z_MAX = 23247
FULL_YX_MAX = 7593

# Half-res (L1) block layout produced by our stitching pipeline.
SLAB_SIZE = 256
SLAB_STRIDE = 224
SLAB_OVERLAP = 32
SLAB_Z0S = list(range(0, 11201, SLAB_STRIDE)) + [11368]
SLAB_OWN = SLAB_OVERLAP // 2  # a slab "owns" z_local >= 16 when possible
TILE_SIZE = 512
TILE_STRIDE = 448
TILE_OVERLAP = 64
TILE_OWN = TILE_OVERLAP // 2  # a tile "owns" y/x_local >= 32 when possible

INST_NAME_RE = re.compile(r"inst_(\d+)")
MIN_HIT_RATE = 0.90
DEFAULT_ROUNDTRIP_N = 100
POINT_KEYS = {"p", "wind_a", "creation_time"}
COLLECTION_KEYS = {"name", "points", "metadata", "color"}


def is_num(v) -> bool:
    """Finite JSON number (bool is not a number)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_bounds(x, y, z, where: str, errors: list) -> None:
    if not (0 <= x < FULL_YX_MAX):
        errors.append(f"{where}: x={x!r} out of bounds [0, {FULL_YX_MAX})")
    if not (0 <= y < FULL_YX_MAX):
        errors.append(f"{where}: y={y!r} out of bounds [0, {FULL_YX_MAX})")
    if not (0 <= z < FULL_Z_MAX):
        errors.append(f"{where}: z={z!r} out of bounds [0, {FULL_Z_MAX})")


def detect_kind(data) -> str:
    if not isinstance(data, dict):
        return "unknown"
    if "control_points" in data:
        return "umbilicus"
    if "vc_pointcollections_json_version" in data or "collections" in data:
        return "point_collection"
    return "unknown"


def validate_point(pt, where: str, errors: list, warnings: list) -> bool:
    """Validate one point dict. Returns True iff it carries wind_a."""
    if not isinstance(pt, dict):
        errors.append(f"{where}: expected object, got {type(pt).__name__}")
        return False
    p = pt.get("p")
    if "p" not in pt:
        errors.append(f"{where}.p: missing required key")
    elif not (isinstance(p, list) and len(p) == 3 and all(is_num(c) for c in p)):
        errors.append(f"{where}.p: expected 3 finite numbers [x, y, z], got {p!r}")
    else:
        check_bounds(p[0], p[1], p[2], f"{where}.p", errors)
    has_wind = "wind_a" in pt
    if has_wind and not is_num(pt["wind_a"]):
        errors.append(f"{where}.wind_a: expected finite number, got {pt['wind_a']!r}")
    if "creation_time" in pt and not is_num(pt["creation_time"]):
        warnings.append(f"{where}.creation_time: expected number, got "
                        f"{type(pt['creation_time']).__name__}")
    for k in sorted(set(pt) - POINT_KEYS):
        warnings.append(f"{where}.{k}: unknown key")
    return has_wind


def validate_point_collection(data: dict, errors: list, warnings: list) -> dict:
    stats = {"collections": 0, "points": 0, "annotated_collections": 0}
    ver = data.get("vc_pointcollections_json_version")
    if ver is None:
        errors.append("vc_pointcollections_json_version: missing required key")
    elif not (isinstance(ver, str) and ver == "1"):
        errors.append(f'vc_pointcollections_json_version: must be the string "1", got {ver!r}')
    colls = data.get("collections")
    if colls is None:
        errors.append("collections: missing required key")
        return stats
    if not isinstance(colls, dict):
        errors.append(f"collections: expected object, got {type(colls).__name__}")
        return stats
    for k in sorted(set(data) - {"vc_pointcollections_json_version", "collections"}):
        warnings.append(f"{k}: unknown top-level key")
    for ck in colls:
        where = f"collections.{ck}"
        coll = colls[ck]
        if not isinstance(coll, dict):
            errors.append(f"{where}: expected object, got {type(coll).__name__}")
            continue
        stats["collections"] += 1
        name = coll.get("name")
        if name is None:
            errors.append(f"{where}.name: missing required key")
        elif not isinstance(name, str):
            errors.append(f"{where}.name: expected string, got {type(name).__name__}")
        points = coll.get("points")
        if "points" not in coll:
            errors.append(f"{where}.points: missing required key")
            points = {}
        elif not isinstance(points, dict):
            errors.append(f"{where}.points: expected object, got {type(points).__name__}")
            points = {}
        elif not points:
            warnings.append(f"{where}.points: empty collection")
        if "metadata" in coll and not isinstance(coll["metadata"], dict):
            errors.append(f"{where}.metadata: expected object, got "
                          f"{type(coll['metadata']).__name__}")
        if "color" in coll:
            color = coll["color"]
            if not (isinstance(color, list) and len(color) == 3
                    and all(is_num(c) for c in color)):
                errors.append(f"{where}.color: expected 3 numbers, got {color!r}")
        for k in sorted(set(coll) - COLLECTION_KEYS):
            warnings.append(f"{where}.{k}: unknown key")
        n_wind = 0
        for pk in points:
            if validate_point(points[pk], f"{where}.points.{pk}", errors, warnings):
                n_wind += 1
        stats["points"] += len(points)
        if 0 < n_wind < len(points):
            errors.append(f"{where}: mixed wind_a ({n_wind} points with, "
                          f"{len(points) - n_wind} without) - official loader "
                          f"strips these; must be all or none")
        elif points and n_wind == len(points):
            stats["annotated_collections"] += 1
    return stats


def validate_umbilicus(data: dict, errors: list, warnings: list) -> dict:
    cps = data.get("control_points")
    if not isinstance(cps, list):
        errors.append(f"control_points: expected array, got {type(cps).__name__}")
        return {"control_points": 0}
    if len(cps) < 2:
        errors.append(f"control_points: need >= 2 points, got {len(cps)}")
    for i, cp in enumerate(cps):
        where = f"control_points.{i}"
        if not isinstance(cp, dict):
            errors.append(f"{where}: expected object, got {type(cp).__name__}")
            continue
        for axis in ("z", "y", "x"):
            if axis not in cp:
                errors.append(f"{where}.{axis}: missing required key")
            elif not is_num(cp[axis]):
                errors.append(f"{where}.{axis}: expected finite number, got {cp[axis]!r}")
        if all(is_num(cp.get(a)) for a in ("z", "y", "x")):
            check_bounds(cp["x"], cp["y"], cp["z"], where, errors)
        for k in sorted(set(cp) - {"z", "y", "x"}):
            warnings.append(f"{where}.{k}: unknown key")
    for k in sorted(set(data) - {"control_points"}):
        warnings.append(f"{k}: unknown top-level key")
    return {"control_points": len(cps)}


def _axis_candidates(v: int, stride: int, size: int, own: int) -> list:
    """Origins covering local coord v, owner-preferred (local >= own, smallest)."""
    cands = []
    k = v // stride
    for origin in (k * stride, (k - 1) * stride):
        if origin >= 0 and 0 <= v - origin < size:
            cands.append(origin)
    return sorted(cands, key=lambda o: (0 if v - o >= own else 1, v - o))


def _slab_candidates(z: int) -> list:
    cands = [z0 for z0 in SLAB_Z0S if 0 <= z - z0 < SLAB_SIZE]
    return sorted(cands, key=lambda z0: (0 if z - z0 >= SLAB_OWN else 1, z - z0))


class _TileCache:
    """Tiny LRU over decompressed label arrays (each ~256 MB)."""

    def __init__(self, maxsize: int = 2):
        self.maxsize = maxsize
        self.cache = {}

    def get(self, path: Path):
        import numpy as np
        if path in self.cache:
            return self.cache[path]
        if len(self.cache) >= self.maxsize:
            self.cache.pop(next(iter(self.cache)))
        with np.load(path) as d:
            arr = d["labels"]
        self.cache[path] = arr
        return arr


def roundtrip_check(data: dict, blocks: Path, n_samples: int,
                    errors: list, warnings: list) -> dict:
    """Sample points from inst_<gid> collections and re-look-up their global id."""
    tile_root = blocks / "blocks" if (blocks / "blocks").is_dir() else blocks
    table_path = (tile_root.parent if tile_root.name == "blocks" else blocks) \
        / "global_table.json"
    if not tile_root.is_dir() or not table_path.exists():
        errors.append(f"roundtrip: blocks dir or global_table.json not found under {blocks}")
        return {}
    with open(table_path) as fh:
        global_table = json.load(fh)

    samples = []  # (expected_gid, x, y, z, where)
    for ck, coll in data.get("collections", {}).items():
        if not isinstance(coll, dict):
            continue
        m = INST_NAME_RE.fullmatch(str(coll.get("name", "")))
        if not m:
            continue
        gid = int(m.group(1))
        points = coll.get("points")
        if not isinstance(points, dict):
            continue
        for pk, pt in points.items():
            p = pt.get("p") if isinstance(pt, dict) else None
            if isinstance(p, list) and len(p) == 3 and all(is_num(c) for c in p):
                samples.append((gid, p[0], p[1], p[2], f"collections.{ck}.points.{pk}"))
    if not samples:
        warnings.append("roundtrip: no points in inst_<globalid> collections; skipped")
        return {"sampled": 0}
    if n_samples <= 0:
        warnings.append("roundtrip: sample size <= 0; skipped")
        return {"sampled": 0}
    rng = random.Random(0)
    if len(samples) > n_samples:
        samples = rng.sample(samples, n_samples)

    def lookup_plan(x, y, z):
        zl1, yl1, xl1 = int(z // 2), int(y // 2), int(x // 2)
        plan = []
        for z0 in _slab_candidates(zl1):
            for y0 in _axis_candidates(yl1, TILE_STRIDE, TILE_SIZE, TILE_OWN):
                for x0 in _axis_candidates(xl1, TILE_STRIDE, TILE_SIZE, TILE_OWN):
                    plan.append((z0, y0, x0, zl1 - z0, yl1 - y0, xl1 - x0))
        return plan

    # Group by preferred tile so each ~256 MB npz decompresses once.
    samples.sort(key=lambda s: lookup_plan(s[1], s[2], s[3])[0][:3]
                 if lookup_plan(s[1], s[2], s[3]) else ())
    cache = _TileCache()
    hits = misses = unresolved = 0
    mismatch_examples = []
    for gid, x, y, z, where in samples:
        verdict = None  # (found_gid,) once a definitive nonzero label is read
        for z0, y0, x0, zl, yl, xl in lookup_plan(x, y, z):
            npz = tile_root / f"z{z0}" / f"tile_y{y0}_x{x0}.npz"
            entry = global_table.get(f"z{z0}/y{y0}_x{x0}")
            if entry is None or not npz.exists():
                continue
            arr = cache.get(npz)
            if zl >= arr.shape[0] or yl >= arr.shape[1] or xl >= arr.shape[2]:
                continue
            local = int(arr[zl, yl, xl])
            if local == 0:
                continue  # background here; an overlapping tile may own it
            found = entry.get(str(local))
            if found is None:
                continue  # unmapped local id; try the alternate tile
            verdict = (int(found),)
            break
        if verdict is None:
            unresolved += 1
        elif verdict[0] == gid:
            hits += 1
        else:
            misses += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(f"{where}: expected {gid}, block has {verdict[0]}")
    rate = hits / len(samples)
    stats = {"sampled": len(samples), "hits": hits, "misses": misses,
             "unresolved": unresolved, "hit_rate": round(rate, 4)}
    for ex in mismatch_examples:
        warnings.append(f"roundtrip mismatch: {ex}")
    if rate < MIN_HIT_RATE:
        errors.append(f"roundtrip: hit rate {rate:.1%} below {MIN_HIT_RATE:.0%} "
                      f"({hits}/{len(samples)} hits, {misses} mismatches, "
                      f"{unresolved} unresolved)")
    return stats


def validate_file(path: Path, blocks, roundtrip_n: int) -> dict:
    errors, warnings, stats = [], [], {}
    kind = "unknown"
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        errors.append(f"unreadable JSON: {exc}")
        data = None
    if data is not None:
        kind = detect_kind(data)
        if kind == "point_collection":
            stats = validate_point_collection(data, errors, warnings)
            if blocks is not None:
                stats["roundtrip"] = roundtrip_check(data, Path(blocks), roundtrip_n,
                                                     errors, warnings)
        elif kind == "umbilicus":
            stats = validate_umbilicus(data, errors, warnings)
        else:
            errors.append("unrecognized schema: expected top-level "
                          "vc_pointcollections_json_version/collections or control_points")
    return {"path": str(path), "kind": kind, "errors": errors,
            "warnings": warnings, "stats": stats}


# ---------------------------------------------------------------- selftest --

def _selftest_fixtures() -> list:
    """(name, payload, expect_valid) triples for the built-in test suite."""
    def pc(colls):
        return {"vc_pointcollections_json_version": "1", "collections": colls}

    good_coll = {"name": "inst_7", "points": {
        "0": {"p": [100.0, 200.0, 300.0], "wind_a": 1.5},
        "1": {"p": [4000, 5000, 20000], "wind_a": 2.5, "creation_time": 1721469600.0},
    }}
    return [
        ("valid_point_collection", pc({"0": good_coll}), True),
        ("valid_umbilicus",
         {"control_points": [{"z": 0, "y": 3796.5, "x": 3796.5},
                             {"z": 11623, "y": 3800.0, "x": 3790.0},
                             {"z": 23246, "y": 3796.5, "x": 3796.5}]}, True),
        ("wrong_version_number",
         {"vc_pointcollections_json_version": 1, "collections": {"0": good_coll}}, False),
        ("wrong_version_string", pc({"0": good_coll}) | {
            "vc_pointcollections_json_version": "2"}, False),
        ("mixed_wind_a", pc({"0": {"name": "inst_7", "points": {
            "0": {"p": [1, 2, 3], "wind_a": 0.5},
            "1": {"p": [4, 5, 6]}}}}), False),
        ("p_length_2", pc({"0": {"name": "inst_7", "points": {
            "0": {"p": [1, 2]}}}}), False),
        ("out_of_bounds_x", pc({"0": {"name": "inst_7", "points": {
            "0": {"p": [7593, 10, 10]}}}}), False),
        ("out_of_bounds_z", pc({"0": {"name": "inst_7", "points": {
            "0": {"p": [10, 10, 23247]}}}}), False),
        ("umbilicus_single_point",
         {"control_points": [{"z": 5, "y": 5, "x": 5}]}, False),
        ("missing_name", pc({"0": {"points": {"0": {"p": [1, 2, 3]}}}}), False),
    ]


def run_selftest() -> int:
    failures = 0
    with tempfile.TemporaryDirectory(prefix="constraints_selftest_") as tmp:
        for name, payload, expect_valid in _selftest_fixtures():
            path = Path(tmp) / f"{name}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            rec = validate_file(path, blocks=None, roundtrip_n=0)
            ok = (len(rec["errors"]) == 0) == expect_valid
            tag = "PASS" if ok else "FAIL"
            want = "accept" if expect_valid else "reject"
            print(f"SELFTEST {tag}: {name} (expected {want}, "
                  f"{len(rec['errors'])} errors)")
            if not ok:
                failures += 1
                for e in rec["errors"]:
                    print(f"    error: {e}")
    print(f"selftest: {'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


# -------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strict validator for spiral-fit constraint JSONs "
                    "(point collections + umbilicus).")
    ap.add_argument("target", nargs="?", help="constraint .json file or directory")
    ap.add_argument("--blocks", help="scroll run dir (contains blocks/ and "
                    "global_table.json) for round-trip id checks")
    ap.add_argument("--roundtrip", type=int, default=DEFAULT_ROUNDTRIP_N,
                    help="points to sample per file for the round-trip check "
                    f"(default {DEFAULT_ROUNDTRIP_N})")
    ap.add_argument("--report", default="validation_report.json",
                    help="report output path (default validation_report.json)")
    ap.add_argument("--selftest", action="store_true",
                    help="run built-in fixture tests and exit")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()
    if not args.target:
        ap.error("target is required unless --selftest is given")
    target = Path(args.target)
    if target.is_dir():
        files = sorted(p for p in target.glob("*.json")
                       if p.name != "validation_report.json")
        if not files:
            print(f"no *.json files found in {target}")
            return 1
    elif target.is_file():
        files = [target]
    else:
        print(f"not found: {target}")
        return 1

    records = [validate_file(p, args.blocks, args.roundtrip) for p in files]
    total_errors = sum(len(r["errors"]) for r in records)
    report = {"target": str(target), "files": records,
              "ok": total_errors == 0}
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    for r in records:
        status = "OK" if not r["errors"] else "INVALID"
        print(f"[{status}] {r['path']} ({r['kind']}) "
              f"errors={len(r['errors'])} warnings={len(r['warnings'])} "
              f"stats={r['stats']}")
        for e in r["errors"][:20]:
            print(f"    error: {e}")
        if len(r["errors"]) > 20:
            print(f"    ... {len(r['errors']) - 20} more errors (see report)")
        for w in r["warnings"][:5]:
            print(f"    warning: {w}")
    print(f"report written to {args.report}; "
          f"{len(records)} file(s), {total_errors} error(s)")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
