"""
Quantify the divergence between detector-observed counts and true arrivals,
separately for each evaluation scenario and each controller.

This addresses the requirement to report the measurement error by scenario
rather than as a single overall correlation, and to expose the endogeneity in
the observation process: the counts a loop registers depend on the signal
timings, which the controller itself decides, so the divergence is expected to
differ between controllers as well as between scenarios.

For each scenario and controller the script runs one simulation, retains the
per-cycle controller log, reconstructs the true arrivals from the scenario's
recorded seed, and compares the two.

True arrivals are recoverable without rerunning the demand generation because
every scenario is determined by its seed: build_scenarios produces the same
vehicles for the same seed every time.

    python3 scripts/detector_divergence.py --jobs 4

Writes results/detector_divergence.json and prints a table suitable for the
report.
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

CONTROLLERS = ["fixed", "reactive", "predictive"]
LABEL = {"fixed": "Fixed-time", "reactive": "Reactive", "predictive": "Prediction-informed"}


def true_arrivals(scenario_name, seed, vary_turning=True):
    """Reconstruct the vehicles released onto each movement in each cycle."""
    scn = SCENARIOS[scenario_name]
    turns = turning_ratios(seed, vary_turning)
    rows, _ = build_rows(scn, seed, turns)
    vehicles = sample_departures(rows, seed, turns)
    arr = np.zeros((N_CYCLES, len(MOVEMENTS)), dtype=float)
    idx = {m: i for i, m in enumerate(MOVEMENTS)}
    for v in vehicles:
        c = int(v["depart"] // CYCLE)
        if 0 <= c < N_CYCLES and v["movement"] in idx:
            arr[c, idx[v["movement"]]] += 1
    return arr


def run_one(controller, scenario, seed, route, model_dir):
    """Run one simulation keeping its per-cycle log, and return the detector counts."""
    workdir = tempfile.mkdtemp(prefix="div_")
    try:
        label = f"{controller}_{scenario}_{seed}"
        cmd = [sys.executable, os.path.join(HERE, "run_controller.py"),
               "--controller", controller, "--route", route,
               "--label", label, "--outdir", workdir]
        if controller == "predictive" and model_dir:
            cmd += ["--model-dir", model_dir]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            return None, proc.stderr[-600:]
        path = os.path.join(workdir, f"{label}_cycles.json")
        if not os.path.exists(path):
            return None, "no cycle log produced"
        cycles = json.load(open(path))
        det = np.zeros((N_CYCLES, len(MOVEMENTS)), dtype=float)
        idx = {m: i for i, m in enumerate(MOVEMENTS)}
        for rec in cycles:
            c = rec.get("cycle")
            if c is None or not (1 <= c <= N_CYCLES):
                continue
            for m, v in rec["counts"].items():
                if m in idx:
                    det[c - 1, idx[m]] = v
        return det, None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def compare(det, true):
    """Compare two (cycles, movements) arrays over the cycles actually logged."""
    used = det.sum(axis=1) > 0
    if used.sum() < 5:
        return None
    d, t = det[used], true[used]
    err = d - t
    mask = t > 0
    return dict(
        cycles=int(used.sum()),
        detector_total=int(d.sum()),
        true_total=int(t.sum()),
        ratio=round(float(d.sum() / t.sum()), 4) if t.sum() else None,
        mae=round(float(np.abs(err).mean()), 3),
        mean_signed_error=round(float(err.mean()), 3),
        correlation=round(float(np.corrcoef(d.ravel(), t.ravel())[0, 1]), 4)
        if t.std() > 0 else None,
        mape_nonzero=round(float(np.abs(err[mask] / t[mask]).mean() * 100), 2)
        if mask.any() else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "models_broad"))
    ap.add_argument("--scenarios-dir", default=os.path.join(ROOT, "demand", "scenarios"))
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="seeds per scenario to average over (default: 3)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "detector_divergence.json"))
    args = ap.parse_args()

    man_path = os.path.join(args.scenarios_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"missing {man_path} - run demand/build_scenarios.py first")
    manifest = json.load(open(man_path))

    results = {}
    for scenario, spec in manifest["scenarios"].items():
        runs = spec["runs"][:max(1, args.n_seeds)]
        seeds = [r["seed"] for r in runs]
        results[scenario] = dict(seeds=seeds, controllers={})
        print(f"\n{scenario}  (seeds {seeds})")

        for c in CONTROLLERS:
            per_seed = []
            for r in runs:
                seed = r["seed"]
                route = os.path.join(args.scenarios_dir, r["route_file"])
                det, err = run_one(c, scenario, seed, route, args.model_dir)
                if det is None:
                    print(f"   {LABEL[c]:<20} seed {seed} FAILED: {err}")
                    continue
                st = compare(det, true_arrivals(scenario, seed))
                if st is not None:
                    per_seed.append(st)
            if not per_seed:
                print(f"   {LABEL[c]:<20} no usable runs")
                continue
            agg = {}
            for k in ("ratio", "correlation", "mae", "mean_signed_error", "mape_nonzero"):
                vals = [p[k] for p in per_seed if p.get(k) is not None]
                agg[k] = round(float(np.mean(vals)), 4) if vals else None
                agg[k + "_sd"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0
            agg["n_seeds"] = len(per_seed)
            agg["per_seed"] = per_seed
            results[scenario]["controllers"][c] = agg
            print(f"   {LABEL[c]:<20} ratio {agg['ratio']:.3f}   "
                  f"corr {agg['correlation']:.3f}   MAE {agg['mae']:.2f}   "
                  f"bias {agg['mean_signed_error']:+.2f}   (n={agg['n_seeds']})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # report table
    print("\n" + "=" * 86)
    print("DETECTOR-OBSERVED COUNTS AGAINST TRUE ARRIVALS, BY SCENARIO AND CONTROLLER")
    print("=" * 86)
    print(f"{'Scenario':<16}{'Controller':<22}{'Ratio':>9}{'Corr':>9}{'MAE':>9}{'Bias':>10}")
    print("-" * 86)
    for scenario, data in results.items():
        for c, st in data["controllers"].items():
            print(f"{scenario:<16}{LABEL[c]:<22}{st['ratio']:>9.3f}{st['correlation']:>9.3f}"
                  f"{st['mae']:>9.2f}{st['mean_signed_error']:>10.2f}")
    print("=" * 86)
    print("  Ratio below 1 means the loops registered fewer vehicles than were released.")
    print("  Bias is the mean signed error, detector count minus true arrivals.")
    print("  Figures are means over the seeds used; per-seed values are in the JSON.")
    print("  Part of the divergence is travel time: a vehicle released at the boundary")
    print("  reaches the loop about fourteen seconds later, so a release near the end of")
    print("  one cycle is registered in the next. This is separate from queue spillback.")
    print(f"\nwrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
