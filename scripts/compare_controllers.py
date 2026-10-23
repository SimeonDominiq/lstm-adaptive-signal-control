"""
Compare how responsive each controller's decisions actually are.

Accuracy and usefulness are not the same thing. A forecast can have a lower
average error than a raw measurement while still being less useful for control,
if it achieves that accuracy by smoothing away the peaks that the controller
needs to respond to.

A model trained on mean-squared error learns to predict the conditional average
of the next cycle. That is the right thing to minimise error, but it produces
estimates with less spread than reality. Feeding a flattened demand estimate
into a proportional control law produces flattened green times, and a controller
whose green times barely move behaves like a fixed-time plan.

This script measures that directly. For each controller it reports:

  spread of the demand estimate   compared with the spread of what actually
                                  happened. A ratio well below 1 means the
                                  estimate is smoother than reality.
  spread of the allocated greens  how much the timings actually move from cycle
                                  to cycle. A controller that does not move is
                                  not adapting.
  clamping                        how often a phase hit the 10 s or 60 s bound.

Usage:
    python3 scripts/compare_controllers.py \\
        --runs results/A_fixed_cycles.json results/B_reactive_cycles.json \\
               results/C_predictive_cycles.json
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOVEMENTS = ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]
PHASE_MOVEMENTS = [["NB_TR", "SB_TR"], ["NB_L", "SB_L"],
                   ["EB_TR", "WB_TR"], ["EB_L", "WB_L"]]
G_MIN, G_MAX = 10, 60


def analyse(path):
    cycles = json.load(open(path))
    label = os.path.basename(path).replace("_cycles.json", "")

    greens = np.array([c["greens"] for c in cycles if c.get("greens")], dtype=float)
    queues = np.array([c["queue_vehicles"] for c in cycles], dtype=float)

    est, act = [], []
    for a, b in zip(cycles, cycles[1:]):
        if not a.get("estimate"):
            continue
        est.append([a["estimate"][m] for m in MOVEMENTS])
        act.append([b["counts"][m] for m in MOVEMENTS])
    est, act = np.array(est, dtype=float), np.array(act, dtype=float)

    out = dict(label=label, cycles=len(cycles),
               mean_queue=float(queues.mean()))

    if greens.size:
        # how far the timings move: spread across phases within a cycle,
        # and movement of each phase from one cycle to the next
        out["green_mean"] = greens.mean(axis=0).round(1).tolist()
        out["green_spread_within_cycle"] = float(greens.std(axis=1).mean())
        out["green_change_between_cycles"] = float(np.abs(np.diff(greens, axis=0)).mean())
        out["pct_at_min"] = float((greens <= G_MIN).mean() * 100)
        out["pct_at_max"] = float((greens >= G_MAX).mean() * 100)

    if est.size:
        # critical movement demand per phase - what the control law actually uses
        def critical(arr):
            idx = [[MOVEMENTS.index(m) for m in movs] for movs in PHASE_MOVEMENTS]
            return np.stack([arr[:, i].max(axis=1) for i in idx], axis=1)

        ce, ca = critical(est), critical(act)
        out["estimate_sd"] = float(ce.std(axis=0).mean())
        out["actual_sd"] = float(ca.std(axis=0).mean())
        out["spread_ratio"] = float(out["estimate_sd"] / out["actual_sd"]) \
            if out["actual_sd"] > 0 else float("nan")
        out["estimate_mae"] = float(np.abs(ce - ca).mean())

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    rows = [analyse(p) for p in args.runs]

    print("\nControl responsiveness")
    print("=" * 78)
    print(f"{'controller':<16}{'mean queue':>12}{'green spread':>14}"
          f"{'green change':>14}{'at min %':>10}{'at max %':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<16}{r['mean_queue']:>12.2f}"
              f"{r.get('green_spread_within_cycle', float('nan')):>14.2f}"
              f"{r.get('green_change_between_cycles', float('nan')):>14.2f}"
              f"{r.get('pct_at_min', float('nan')):>10.1f}"
              f"{r.get('pct_at_max', float('nan')):>10.1f}")
    print("=" * 78)
    print("  green spread  = how unevenly green is split across the four phases")
    print("  green change  = how much each phase's green moves between cycles")

    demand_rows = [r for r in rows if "spread_ratio" in r]
    if demand_rows:
        print("\nDemand estimate versus what actually happened")
        print("=" * 78)
        print(f"{'controller':<16}{'estimate sd':>14}{'actual sd':>12}"
              f"{'ratio':>10}{'MAE':>10}")
        print("-" * 78)
        for r in demand_rows:
            print(f"{r['label']:<16}{r['estimate_sd']:>14.2f}{r['actual_sd']:>12.2f}"
                  f"{r['spread_ratio']:>10.3f}{r['estimate_mae']:>10.2f}")
        print("=" * 78)
        print("  ratio < 1 means the estimate is smoother than reality, so the")
        print("  controller under-reacts to peaks even if its average error is low.")

        for r in demand_rows:
            if r["spread_ratio"] < 0.8:
                print(f"\n  {r['label']}: estimate spread is only "
                      f"{r['spread_ratio']:.0%} of actual. This controller is")
                print("  systematically flattening demand, which will make its green")
                print("  times move less than they should.")

    for r in rows:
        if "green_mean" in r:
            print(f"\n  {r['label']} mean greens: {r['green_mean']}")


if __name__ == "__main__":
    main()
