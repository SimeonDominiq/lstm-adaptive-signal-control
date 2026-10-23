"""
Diagnose the detector count excess observed in free-flowing conditions.

In the low, moderate and surge scenarios the loops register roughly 4.5 per cent
more vehicles than the generator released. Over a flat profile with no queueing
this should not occur: a count taken 200 m downstream should approximately equal
the count released, or fall slightly short because vehicles are still in transit
when the run ends.

This script does not assume a cause. It runs one scenario under fixed-time
control, keeps the per-cycle log, and reports four things that between them
distinguish the candidate explanations:

  1.  Totals, to confirm the excess is present in this run.
  2.  Per-cycle differences over time, to show whether the excess is spread
      evenly across the run or concentrated at its start or end.
  3.  The distribution of per-cycle differences, to show whether a small number
      of cycles carry the excess or whether every cycle contributes.
  4.  A shifted comparison, in which detector counts are compared against the
      arrivals of the preceding cycle, to test whether the excess is an artefact
      of aligning a lagged observation against an unlagged truth.

What each outcome would indicate is stated at the end of the output. No
conclusion is drawn here that the numbers do not support.

    python3 scripts/diagnose_detector_excess.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "demand"))

from generate_demand import CYCLE, N_CYCLES, MOVEMENTS, turning_ratios, sample_departures  # noqa: E402
from build_scenarios import SCENARIOS, build_rows  # noqa: E402


def true_arrivals(scenario_name, seed):
    scn = SCENARIOS[scenario_name]
    turns = turning_ratios(seed, True)
    rows, _ = build_rows(scn, seed, turns)
    vehicles = sample_departures(rows, seed, turns)
    arr = np.zeros((N_CYCLES, len(MOVEMENTS)), dtype=float)
    idx = {m: i for i, m in enumerate(MOVEMENTS)}
    n_out_of_range = 0
    for v in vehicles:
        c = int(v["depart"] // CYCLE)
        if 0 <= c < N_CYCLES and v["movement"] in idx:
            arr[c, idx[v["movement"]]] += 1
        else:
            n_out_of_range += 1
    return arr, len(vehicles), n_out_of_range


def run(scenario, seed, route):
    workdir = tempfile.mkdtemp(prefix="diag_")
    try:
        label = f"diag_{scenario}_{seed}"
        cmd = [sys.executable, os.path.join(HERE, "run_controller.py"),
               "--controller", "fixed", "--route", route,
               "--label", label, "--outdir", workdir]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"run failed:\n{proc.stderr[-1500:]}")
        cycles = json.load(open(os.path.join(workdir, f"{label}_cycles.json")))
        det = np.zeros((N_CYCLES, len(MOVEMENTS)), dtype=float)
        idx = {m: i for i, m in enumerate(MOVEMENTS)}
        logged = []
        for rec in cycles:
            c = rec["cycle"]
            if 1 <= c <= N_CYCLES:
                logged.append(c)
                for m, v in rec["counts"].items():
                    if m in idx:
                        det[c - 1, idx[m]] = v
        return det, sorted(logged)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="low")
    ap.add_argument("--scenarios-dir", default=os.path.join(ROOT, "demand", "scenarios"))
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.scenarios_dir, "manifest.json")))
    run_spec = manifest["scenarios"][args.scenario]["runs"][0]
    seed = run_spec["seed"]
    route = os.path.join(args.scenarios_dir, run_spec["route_file"])

    truth, n_vehicles, n_out = true_arrivals(args.scenario, seed)
    det, logged = run(args.scenario, seed, route)

    print(f"scenario {args.scenario}, seed {seed}, fixed-time control\n")

    # ---- 1. totals -------------------------------------------------------
    print("1.  TOTALS")
    print(f"    vehicles generated              {n_vehicles}")
    print(f"    of which outside the cycle grid {n_out}")
    print(f"    true arrivals in grid           {int(truth.sum())}")
    print(f"    detector counts logged          {int(det.sum())}")
    print(f"    ratio                           {det.sum()/truth.sum():.4f}")
    print(f"    cycles logged                   {len(logged)}  "
          f"(first {logged[0] if logged else '-'}, last {logged[-1] if logged else '-'})")

    # restrict to cycles actually logged, as the divergence script does
    used = np.zeros(N_CYCLES, dtype=bool)
    for c in logged:
        used[c - 1] = True
    d, t = det[used], truth[used]
    print(f"    ratio over logged cycles only   {d.sum()/t.sum():.4f}")

    # ---- 2. where the excess sits over time ------------------------------
    per_cycle = d.sum(axis=1) - t.sum(axis=1)
    n = len(per_cycle)
    thirds = [per_cycle[:n//3], per_cycle[n//3:2*n//3], per_cycle[2*n//3:]]
    print("\n2.  EXCESS OVER TIME  (detector minus true, vehicles per cycle)")
    for name, seg in zip(["first third ", "middle third", "last third  "], thirds):
        print(f"    {name}  mean {seg.mean():+7.3f}   total {seg.sum():+8.1f}")
    print(f"    first cycle  {per_cycle[0]:+.1f}        last cycle {per_cycle[-1]:+.1f}")

    # ---- 3. distribution -------------------------------------------------
    print("\n3.  DISTRIBUTION OF PER-CYCLE DIFFERENCES")
    print(f"    cycles where detector > true    {(per_cycle > 0).sum()} of {n}")
    print(f"    cycles where detector < true    {(per_cycle < 0).sum()}")
    print(f"    cycles equal                    {(per_cycle == 0).sum()}")
    print(f"    mean {per_cycle.mean():+.3f}   sd {per_cycle.std(ddof=1):.3f}   "
          f"min {per_cycle.min():+.0f}   max {per_cycle.max():+.0f}")

    # ---- 4. shifted comparison -------------------------------------------
    print("\n4.  SHIFTED COMPARISON  (detector cycle c against true cycle c-1)")
    if n > 2:
        d_shift = d[1:].sum(axis=1)
        t_shift = t[:-1].sum(axis=1)
        print(f"    unshifted ratio                 {d.sum()/t.sum():.4f}")
        print(f"    shifted   ratio                 {d_shift.sum()/t_shift.sum():.4f}")
        print(f"    unshifted correlation           "
              f"{np.corrcoef(d.sum(axis=1), t.sum(axis=1))[0,1]:.4f}")
        print(f"    shifted   correlation           "
              f"{np.corrcoef(d_shift, t_shift)[0,1]:.4f}")

    # ---- interpretation guide -------------------------------------------
    print("\n" + "=" * 74)
    print("WHAT EACH OUTCOME WOULD INDICATE")
    print("=" * 74)
    print("  If the excess is concentrated in the first cycles, the run begins with")
    print("  vehicles already present or the first logged cycle covers more than one")
    print("  cycle of releases.")
    print("  If it is spread evenly across every cycle, the counting or the release")
    print("  accounting differs systematically, not at the boundaries.")
    print("  If a small number of cycles carry it, those cycles should be inspected")
    print("  individually.")
    print("  If the shifted ratio is markedly closer to one than the unshifted ratio,")
    print("  the comparison is misaligned by one cycle and the divergence figures")
    print("  must be recomputed on the shifted basis.")
    print("  If none of these holds, the cause is not established by this test and")
    print("  further work is required before the figures are used.")


if __name__ == "__main__":
    main()
