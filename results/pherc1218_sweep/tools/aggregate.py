#!/usr/bin/env python3
"""Derive the RESULTS_en.md aggregate straight from windows.csv.

Every number quoted in the Results section of RESULTS_en.md is produced
here, so the prose cannot drift from the table. Run after any change to
windows.csv and paste the output.

    python3 tools/aggregate.py [windows.csv]
"""
import csv
import re
import sys

REL_MIN = 94.0
SAME_MIN = 91.0
DR_C, DR_TOL = 10.1, 0.8
DR_COLS = ("dr_L1_zm200", "dr_L1_c", "dr_L1_zp200")


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def loss_bounds(s):
    """loss_final is a per-window range: '2.2-3.6', '~2.8', '2-3', 'n/r'."""
    vals = [float(x) for x in re.findall(r"\d+\.?\d*", s or "")]
    return (min(vals), max(vals)) if vals else None


def rng(vals, fmt="{:.2f}"):
    return fmt.format(min(vals)) + "-" + fmt.format(max(vals))


def main(path):
    rows = list(csv.DictReader(open(path)))

    gate = [r for r in rows if "gate" in r["verdict"].lower()]
    sweep = [r for r in rows if r not in gate]
    prov = [r for r in sweep if r["verdict"].startswith("PROVISIONAL")]
    ok = [r for r in sweep if r not in prov]

    def label(r):
        v = r["verdict"]
        if v.startswith("full PASS"):
            return "full PASS"
        if "with-note" in v:
            return "PASS-with-note"
        return "PASS"

    counts = {k: sum(1 for r in ok if label(r) == k)
              for k in ("full PASS", "PASS", "PASS-with-note")}

    rel = [num(r["rel_pct"]) for r in ok]
    same = [num(r["same_pct"]) for r in ok]
    assert all(v is not None for v in rel + same), "non-numeric rel/same in a PASS-class row"

    dr = [num(r[c]) for r in ok for c in DR_COLS]
    dr_missing = sum(1 for v in dr if v is None)
    dr = [v for v in dr if v is not None]

    lb = [loss_bounds(r["loss_final"]) for r in ok]
    loss_nr = sum(1 for b in lb if b is None)
    lo = [b[0] for b in lb if b] + [b[1] for b in lb if b]

    wall = sum(num(r["wall_s"]) or 0.0 for r in sweep) / 60.0

    n = len(ok)
    print(f"sweep windows       : {len(sweep)}  (+{len(gate)} gate reference)")
    print(f"validated (gate)    : {n}   "
          f"[full PASS {counts['full PASS']}, PASS {counts['PASS']}, "
          f"PASS-with-note {counts['PASS-with-note']}]")
    print(f"PROVISIONAL         : {len(prov)}")
    print()
    print(f"relative            : {rng(rel, '{:.1f}')} %   "
          f"({sum(1 for v in rel if v >= REL_MIN)}/{n} >= {REL_MIN:g})")
    print(f"same                : {rng(same, '{:.1f}')} %   "
          f"({sum(1 for v in same if v >= SAME_MIN)}/{n} >= {SAME_MIN:g})")
    inb = sum(1 for v in dr if abs(v - DR_C) <= DR_TOL)
    print(f"median dr           : {rng(dr)} L1 over {len(dr)} spot-checks   "
          f"({inb}/{len(dr)} within {DR_C}+-{DR_TOL})"
          + (f"   [{dr_missing} not recorded]" if dr_missing else ""))
    print(f"final loss          : {rng(lo, '{:.1f}')}"
          + (f"   [{loss_nr} window(s) not recorded]" if loss_nr else ""))
    print(f"sweep GPU time      : {wall:.1f} min")

    bad = [r["window_z"] for r in ok
           if num(r["rel_pct"]) < REL_MIN or num(r["same_pct"]) < SAME_MIN
           or any(num(r[c]) is not None and abs(num(r[c]) - DR_C) > DR_TOL
                  for c in DR_COLS)]
    print()
    print("gate check on PASS-class rows: " + ("OK" if not bad else "VIOLATION " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "windows.csv"))
