"""
Re-analyse the experiment, correcting two things the first summary got wrong.

1. Completion bias.
   The delay metrics are averages over vehicles that finished their trip. A
   controller that admits fewer vehicles has fewer completions, and the ones
   that do finish look faster. Comparing average delay between controllers that
   completed different numbers of trips is therefore not like for like. This
   script reports the completion count alongside every delay figure and flags
   any comparison where the two controllers differ materially in how much
   traffic they served.

2. Practical significance.
   With twenty paired seeds and small variance, differences well under a second
   reach very small p values and very large standardised effect sizes. A
   statistically reliable difference of half a second in waiting time is not a
   traffic engineering result. Each difference is therefore also judged against
   a practical threshold, and reported separately from the statistical test.

Thresholds, stated in advance rather than chosen after seeing the numbers:

    waiting_time   5 s per vehicle
    time_loss      5 s per vehicle
    duration       5 s per vehicle
    throughput    50 veh/h

    python3 scripts/analyse_experiment.py
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

METRICS = ["waiting_time", "time_loss", "duration", "throughput"]
LOWER_IS_BETTER = {"waiting_time": True, "time_loss": True,
                   "duration": True, "throughput": False}
PRACTICAL = {"waiting_time": 5.0, "time_loss": 5.0, "duration": 5.0,
             "throughput": 50.0}
UNITS = {"waiting_time": "s", "time_loss": "s", "duration": "s",
         "throughput": "veh/h"}
SHORT = {"fixed": "A", "reactive": "B", "predictive": "C"}
# a completion difference above this makes the delay comparison unsafe
COMPLETION_TOLERANCE = 0.03


def load(outdir):
    raw = json.load(open(os.path.join(outdir, "raw_results.json")))
    stats_ = json.load(open(os.path.join(outdir, "statistics.json")))
    return raw, stats_


def completion_check(raw, scenario, x, y):
    rx = {r["seed"]: r for r in raw if r["controller"] == x and r["scenario"] == scenario}
    ry = {r["seed"]: r for r in raw if r["controller"] == y and r["scenario"] == scenario}
    seeds = sorted(set(rx) & set(ry))
    cx = np.array([rx[s]["completed"] for s in seeds], float)
    cy = np.array([ry[s]["completed"] for s in seeds], float)
    mx, my = cx.mean(), cy.mean()
    rel = abs(mx - my) / max(mx, my) if max(mx, my) > 0 else 0.0
    return dict(completed_x=round(float(mx), 1), completed_y=round(float(my), 1),
                relative_gap=round(float(rel), 4),
                comparable=bool(rel <= COMPLETION_TOLERANCE))


def verdict(scenario_stats, comp):
    """
    Decide a scenario for one comparison, using only differences that are both
    statistically reliable and practically meaningful. Delay metrics are set
    aside when the two controllers served materially different amounts of
    traffic.
    """
    wins = {"x": [], "y": [], "ignored": []}
    xs, ys = scenario_stats["comparison"].split(" vs ")
    for metric in METRICS:
        m = scenario_stats["metrics"][metric]
        if not m["significant_after_holm"]:
            continue
        diff = abs(m["mean_difference"])
        if diff < PRACTICAL[metric]:
            wins["ignored"].append((metric, "below practical threshold", diff))
            continue
        if metric != "throughput" and not comp["comparable"]:
            wins["ignored"].append((metric, "completion gap", diff))
            continue
        if m["better"] == xs:
            wins["x"].append((metric, diff))
        elif m["better"] == ys:
            wins["y"].append((metric, diff))
    return wins, xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "experiment"))
    args = ap.parse_args()

    raw, stats_ = load(args.outdir)
    scenarios = sorted({s["scenario"] for s in stats_},
                       key=lambda s: ["low", "moderate", "peak", "oversaturated",
                                      "surge", "imbalanced"].index(s)
                       if s in ["low", "moderate", "peak", "oversaturated",
                                "surge", "imbalanced"] else 99)

    print("=" * 92)
    print("TRAFFIC SERVED  (delay averages are over completed trips only)")
    print("=" * 92)
    print(f"{'scenario':<16}{'A completed':>14}{'B completed':>14}{'C completed':>14}")
    for scn in scenarios:
        row = [scn]
        for c in ["fixed", "reactive", "predictive"]:
            v = [r["completed"] for r in raw
                 if r["controller"] == c and r["scenario"] == scn]
            row.append(f"{np.mean(v):.0f}" if v else "-")
        print(f"{row[0]:<16}{row[1]:>14}{row[2]:>14}{row[3]:>14}")
    print("\n  Where these differ materially, average delay is not comparable:")
    print("  a controller serving fewer vehicles will tend to show lower delay")
    print("  simply because the vehicles it failed to serve are not counted.")

    summary = {}
    for comparison in ["A vs B", "B vs C", "A vs C"]:
        print("\n" + "=" * 92)
        print(f"{comparison}   statistically reliable AND practically meaningful differences")
        print("=" * 92)
        tally = {}
        for scn in scenarios:
            s = next((z for z in stats_ if z["scenario"] == scn
                      and z["comparison"] == comparison), None)
            if not s:
                continue
            x, y = s["controllers"]
            comp = completion_check(raw, scn, x, y)
            wins, xs, ys = verdict(s, comp)

            if wins["x"] and not wins["y"]:
                v = f"{xs} better"
            elif wins["y"] and not wins["x"]:
                v = f"{ys} better"
            elif wins["x"] and wins["y"]:
                v = "mixed"
            else:
                v = "no meaningful difference"
            tally[scn] = v

            flag = "" if comp["comparable"] else \
                f"   [completion gap {comp['relative_gap']:.1%} - delay not comparable]"
            print(f"\n  {scn:<16}{v}{flag}")
            for metric, d in wins["x"]:
                print(f"      {xs} better on {metric:<13} by {d:.1f} {UNITS[metric]}")
            for metric, d in wins["y"]:
                print(f"      {ys} better on {metric:<13} by {d:.1f} {UNITS[metric]}")
            for metric, why, d in wins["ignored"]:
                print(f"      set aside: {metric:<13} ({why}, {d:.1f} {UNITS[metric]})")
        summary[comparison] = tally

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(f"{'scenario':<16}{'A vs B':<26}{'B vs C':<26}{'A vs C':<26}")
    print("-" * 92)
    for scn in scenarios:
        print(f"{scn:<16}"
              f"{summary['A vs B'].get(scn, '-'):<26}"
              f"{summary['B vs C'].get(scn, '-'):<26}"
              f"{summary['A vs C'].get(scn, '-'):<26}")
    print("=" * 92)

    bc = summary["B vs C"]
    c_wins = sum(1 for v in bc.values() if v.startswith("C"))
    b_wins = sum(1 for v in bc.values() if v.startswith("B"))
    none = sum(1 for v in bc.values() if v.startswith("no"))
    mixed = sum(1 for v in bc.values() if v == "mixed")
    print("\nDoes prediction add value beyond reacting?")
    print(f"  scenarios where prediction is better : {c_wins}")
    print(f"  scenarios where reaction is better   : {b_wins}")
    print(f"  scenarios with mixed evidence        : {mixed}")
    print(f"  scenarios with no meaningful difference: {none}")

    with open(os.path.join(args.outdir, "honest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.relpath(args.outdir, ROOT)}/honest_summary.json")


if __name__ == "__main__":
    main()
