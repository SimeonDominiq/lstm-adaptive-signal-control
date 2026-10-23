"""
Establish why the controller logs 100 cycles rather than 102.

A 14,400 second run at a 140 second cycle contains 102.86 cycles, so 102 whole
cycles begin within the run. The controller logs 100. The detector totals
reconcile exactly against the vehicles released, so nothing is being miscounted;
the question is which cycles are absent from the log and why.

This script does not assume an answer. It runs one scenario under fixed-time
control, keeps the per-cycle log, and reports:

  1.  Which cycle indices appear in the log and which are missing.
  2.  The simulation time recorded against the first and last logged cycles,
      compared with where those cycles should fall if the first began at t = 0.
  3.  The interval between consecutive logged cycles, to show whether the
      programme is holding the nominal 140 seconds throughout.
  4.  How many vehicles are released in the cycles that are absent, and what
      share of the run that represents.

The controller detects a new cycle when the signal programme returns to its
first phase. Two mechanisms could remove cycles from the log, and the output
distinguishes them: cycles that never began before the run ended, and a cycle
whose start was not detected because the run began part-way through a phase.

    python3 scripts/check_missing_cycles.py
    python3 scripts/check_missing_cycles.py --scenario oversaturated
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

RUN_END = 14400


def released_per_cycle(scenario_name, seed):
    scn = SCENARIOS[scenario_name]
    turns = turning_ratios(seed, True)
    rows, _ = build_rows(scn, seed, turns)
    vehicles = sample_departures(rows, seed, turns)
    counts = {}
    for v in vehicles:
        c = int(v["depart"] // CYCLE)
        counts[c] = counts.get(c, 0) + 1
    return counts, len(vehicles)


def run(scenario, seed, route):
    workdir = tempfile.mkdtemp(prefix="miss_")
    try:
        label = f"miss_{scenario}_{seed}"
        cmd = [sys.executable, os.path.join(HERE, "run_controller.py"),
               "--controller", "fixed", "--route", route,
               "--label", label, "--outdir", workdir]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"run failed:\n{proc.stderr[-1500:]}")
        return json.load(open(os.path.join(workdir, f"{label}_cycles.json")))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="low")
    ap.add_argument("--scenarios-dir", default=os.path.join(ROOT, "demand", "scenarios"))
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.scenarios_dir, "manifest.json")))
    spec = manifest["scenarios"][args.scenario]["runs"][0]
    seed = spec["seed"]
    route = os.path.join(args.scenarios_dir, spec["route_file"])

    cycles = run(args.scenario, seed, route)
    released, total_vehicles = released_per_cycle(args.scenario, seed)

    logged = sorted(r["cycle"] for r in cycles)
    times = {r["cycle"]: r["t"] for r in cycles}

    whole_cycles = int(RUN_END // CYCLE)
    print(f"scenario {args.scenario}, seed {seed}, fixed-time control\n")

    # ---- 1. which indices are present ------------------------------------
    print("1.  CYCLE INDICES")
    print(f"    run length                    {RUN_END} s")
    print(f"    nominal cycle                 {int(CYCLE)} s")
    print(f"    whole cycles within the run   {whole_cycles}")
    print(f"    cycles the generator uses     {N_CYCLES}")
    print(f"    cycles logged                 {len(logged)}")
    print(f"    first / last logged index     {logged[0]} / {logged[-1]}")
    expected = set(range(1, whole_cycles + 1))
    missing = sorted(expected - set(logged))
    extra = sorted(set(logged) - expected)
    print(f"    indices 1..{whole_cycles} absent from the log   {missing if missing else 'none'}")
    if extra:
        print(f"    indices logged beyond {whole_cycles}        {extra}")
    gaps = [b - a for a, b in zip(logged, logged[1:]) if b - a != 1]
    print(f"    breaks in the logged sequence {'none' if not gaps else gaps}")

    # ---- 2. timing of first and last -------------------------------------
    print("\n2.  TIMING OF THE LOGGED CYCLES")
    t_first, t_last = times[logged[0]], times[logged[-1]]
    print(f"    time at first logged cycle    {t_first} s")
    print(f"    time at last logged cycle     {t_last} s")
    print(f"    if cycle 1 ended at t={int(CYCLE)}, cycle {logged[0]} would end at "
          f"{int(logged[0]*CYCLE)} s")
    print(f"    time remaining after the last {RUN_END - t_last} s "
          f"({(RUN_END - t_last)/CYCLE:.2f} cycles)")

    # ---- 3. interval between cycles --------------------------------------
    ts = [times[c] for c in logged]
    deltas = np.diff(ts)
    print("\n3.  INTERVAL BETWEEN CONSECUTIVE LOGGED CYCLES")
    vals, counts_ = np.unique(deltas, return_counts=True)
    for v, n in zip(vals, counts_):
        print(f"    {int(v)} s   x{n}")
    print(f"    mean {deltas.mean():.2f} s   against a nominal {int(CYCLE)} s")

    # ---- 4. vehicles in the absent cycles ---------------------------------
    print("\n4.  VEHICLES RELEASED IN CYCLES ABSENT FROM THE LOG")
    absent_idx = sorted(set(range(N_CYCLES)) - {c - 1 for c in logged})
    n_absent = sum(released.get(i, 0) for i in absent_idx)
    print(f"    generator cycle indices absent  {absent_idx if len(absent_idx) < 12 else str(absent_idx[:12]) + ' ...'}")
    print(f"    vehicles released in them       {n_absent}")
    print(f"    total vehicles released         {total_vehicles}")
    print(f"    share of the run                {100*n_absent/total_vehicles:.2f} %")

    # ---- interpretation ---------------------------------------------------
    print("\n" + "=" * 74)
    print("WHAT EACH OUTCOME WOULD INDICATE")
    print("=" * 74)
    print("  If the absent indices are the LAST one or two and the time remaining")
    print("  after the final logged cycle is under one cycle, those cycles simply")
    print("  did not complete before the run ended. The log is correct and the")
    print("  comparison window should be restricted to the cycles that completed.")
    print("  If the absent index is the FIRST, the run began part-way through the")
    print("  programme and the first return to phase one was not a full cycle.")
    print("  If intervals other than the nominal cycle appear, the programme is not")
    print("  holding its cycle length and that must be explained before any timing")
    print("  claim is made.")
    print("  If absent indices fall in the middle of the sequence, cycle starts are")
    print("  being missed during the run and the cause is not established here.")


if __name__ == "__main__":
    main()
