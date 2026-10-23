"""
Diagnostic - how accurate is the forecast when it is actually driving the signal?

The model was trained and tested on detector counts recorded under fixed-time
control. It is then used under adaptive control, where the green times change
every cycle and so the pattern the detectors see is different. If the forecast
is much less accurate in use than it was on the test partition, that difference
is the distribution shift, and it would explain a predictive controller failing
to beat a reactive one.

This compares, cycle by cycle, the demand the controller estimated for the cycle
about to begin against the counts actually measured during that cycle.

Usage:
    python3 scripts/diagnose_forecast.py --run results/C_predictive_cycles.json
    python3 scripts/diagnose_forecast.py --run results/B_reactive_cycles.json
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOVEMENTS = ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]


def load(path):
    with open(path) as f:
        return json.load(f)


def paired(cycles):
    """
    Pair the estimate made at the end of cycle t (which is a forecast for the
    cycle about to begin) with the counts actually measured in cycle t+1.
    """
    est, act = [], []
    for a, b in zip(cycles, cycles[1:]):
        if not a.get("estimate"):
            continue
        est.append([a["estimate"][m] for m in MOVEMENTS])
        act.append([b["counts"][m] for m in MOVEMENTS])
    return np.array(est, dtype=float), np.array(act, dtype=float)


def metrics(est, act):
    err = est - act
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    mask = act > 0
    mape = float(np.mean(np.abs(err[mask] / act[mask])) * 100.0) if mask.any() else float("nan")
    return mae, rmse, bias, mape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a *_cycles.json file")
    ap.add_argument("--report", default=os.path.join(ROOT, "models", "model_report.json"))
    args = ap.parse_args()

    cycles = load(args.run)
    est, act = paired(cycles)
    if len(est) == 0:
        raise SystemExit("no estimates logged in that run - re-run the controller "
                         "with the updated script")

    mae, rmse, bias, mape = metrics(est, act)

    print(f"\nLive forecast accuracy under adaptive control")
    print(f"  file            {os.path.relpath(args.run, ROOT)}")
    print(f"  cycles compared {len(est)}")
    print(f"  MAE             {mae:.3f}")
    print(f"  RMSE            {rmse:.3f}")
    print(f"  MAPE            {mape:.2f} %")
    print(f"  bias            {bias:+.3f}  "
          f"({'over' if bias > 0 else 'under'}-estimating on average)")

    if os.path.exists(args.report):
        rep = json.load(open(args.report))
        off = rep.get("test_metrics", {}).get("LSTM")
        if off:
            print(f"\nOffline accuracy on the held-out test partition (fixed-time data)")
            print(f"  MAE             {off['MAE']:.3f}")
            print(f"  RMSE            {off['RMSE']:.3f}")
            degrade = (mae - off["MAE"]) / off["MAE"] * 100.0
            print(f"\n  MAE is {degrade:+.1f}% in use compared with offline")
            if degrade > 25:
                print("\n  The forecast is substantially less accurate when it is driving")
                print("  the signal than it was offline. This is distribution shift: the")
                print("  model was trained on traffic shaped by a fixed plan, and is now")
                print("  seeing traffic shaped by its own decisions. It is a real")
                print("  limitation and should be reported, not hidden.")
            elif degrade > 10:
                print("\n  Some degradation in use, but moderate.")
            else:
                print("\n  Accuracy holds up in use, so distribution shift is not the")
                print("  main explanation for the controller's behaviour.")

    print(f"\nPer-movement MAE in use")
    for i, m in enumerate(MOVEMENTS):
        e = np.abs(est[:, i] - act[:, i]).mean()
        print(f"  {m:<6} {e:.3f}   (mean actual {act[:, i].mean():.1f})")


if __name__ == "__main__":
    main()
