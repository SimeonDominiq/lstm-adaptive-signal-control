"""
Stage 9a - build the six evaluation scenarios.

Chapter Three specifies six demand conditions for the controller comparison.
Each is generated with the same set of seeds, and every controller is run on
exactly the same route files, so the comparison is paired: any difference
between controllers cannot come from differences in the traffic.

    low            light demand throughout
    moderate       comfortable demand
    peak           heavy demand near the calibrated design point
    oversaturated  demand above what the intersection can serve
    surge          moderate demand with a sharp directional surge
    imbalanced     sustained imbalance between the two road axes

Scenario demand is expressed as a fraction of the intersection capacity of
about 3,086 veh/h derived in Chapter Three.

These are evaluation conditions only. No model is fitted to them.

    python3 demand/build_scenarios.py --seeds 20
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_demand import (  # noqa: E402
    CAPACITY, CYCLE, N_CYCLES, APPROACHES, ROUTE_MAP, VEHICLE_TYPES,
    sample_departures, write_routes, turning_ratios,
)

SCENARIOS = {
    "low":           dict(level=0.35, shape="flat"),
    "moderate":      dict(level=0.60, shape="flat"),
    "peak":          dict(level=0.85, shape="flat"),
    "oversaturated": dict(level=1.05, shape="flat"),
    "surge":         dict(level=0.60, shape="surge", surge_level=1.00,
                          surge_start_min=90, surge_end_min=130),
    "imbalanced":    dict(level=0.65, shape="imbalanced", major_share=0.72),
}


def level_at(scn, t_min):
    if scn["shape"] == "surge":
        if scn["surge_start_min"] <= t_min < scn["surge_end_min"]:
            return scn["surge_level"]
        return scn["level"]
    return scn["level"]


def splits(scn, dominant, t_min):
    """Share of total demand on each approach."""
    if scn["shape"] == "imbalanced":
        # sustained imbalance between the north-south and east-west axes
        major = scn["major_share"]
        if dominant in ("NB", "SB"):
            axis = {"NB": major / 2, "SB": major / 2,
                    "EB": (1 - major) / 2, "WB": (1 - major) / 2}
        else:
            axis = {"EB": major / 2, "WB": major / 2,
                    "NB": (1 - major) / 2, "SB": (1 - major) / 2}
        return axis

    if scn["shape"] == "surge" and \
            scn["surge_start_min"] <= t_min < scn["surge_end_min"]:
        s = {a: 0.18 for a in APPROACHES}
        s[dominant] = 0.46
        total = sum(s.values())
        return {a: v / total for a, v in s.items()}

    return {a: 0.25 for a in APPROACHES}


def build_rows(scn, seed, turns, noise_sd=0.02):
    rng = np.random.default_rng(seed)
    dominant = APPROACHES[int(np.random.default_rng(seed + 31).integers(0, 4))]
    rows = []
    for c in range(N_CYCLES):
        t_s = c * CYCLE
        t_min = t_s / 60.0
        lvl = level_at(scn, t_min)
        rate = CAPACITY * lvl * float(rng.normal(1.0, noise_sd))
        rate = max(rate, 0.0)
        sh = splits(scn, dominant, t_min)
        row = dict(cycle=c, t_start=t_s, regime=scn["shape"],
                   level=lvl, total_rate=rate)
        for a in APPROACHES:
            ar = rate * sh[a]
            row[f"{a}_L"] = ar * turns["left"]
            row[f"{a}_TR"] = ar * (turns["through"] + turns["right"])
        rows.append(row)
    return rows, dominant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20,
                    help="paired seeds per scenario (Chapter 3 requires at least 20)")
    ap.add_argument("--seed-base", type=int, default=50000)
    ap.add_argument("--vary-turning", action="store_true", default=True,
                    help="vary turning ratios across seeds, as in the held-out data")
    ap.add_argument("--outdir", default=os.path.join(HERE, "scenarios"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    manifest = dict(capacity_veh_per_h=CAPACITY, cycles_per_run=N_CYCLES,
                    seeds_per_scenario=args.seeds, scenarios={})

    total_runs = 0
    for name, scn in SCENARIOS.items():
        sdir = os.path.join(args.outdir, name)
        os.makedirs(sdir, exist_ok=True)
        runs = []
        for k in range(args.seeds):
            seed = args.seed_base + 1000 * list(SCENARIOS).index(name) + k
            turns = turning_ratios(seed, args.vary_turning)
            rows, dominant = build_rows(scn, seed, turns)
            vehicles = sample_departures(rows, seed, turns)
            fname = f"{name}_s{seed}.rou.xml"
            write_routes(vehicles, os.path.join(sdir, fname))
            runs.append(dict(route_file=os.path.join(name, fname), seed=seed,
                             dominant=dominant, turning=turns,
                             n_vehicles=len(vehicles),
                             level=scn["level"], shape=scn["shape"]))
            total_runs += 1
        mean_veh = float(np.mean([r["n_vehicles"] for r in runs]))
        manifest["scenarios"][name] = dict(spec=scn, n_runs=len(runs),
                                           mean_vehicles=round(mean_veh, 1),
                                           runs=runs)
        print(f"{name:<14} {len(runs):>3} seeds   mean {mean_veh:>7.0f} vehicles   "
              f"level {scn['level']:.2f} of capacity")

    mpath = os.path.join(args.outdir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 66)
    print(f"scenarios          {len(SCENARIOS)}")
    print(f"seeds per scenario {args.seeds}")
    print(f"route files        {total_runs}")
    print(f"controller runs    {total_runs * 3}  (three controllers, paired seeds)")
    print(f"cycle records      {total_runs * 3 * N_CYCLES}")
    print(f"manifest           {os.path.relpath(mpath, os.path.dirname(HERE))}")
    print("=" * 66)


if __name__ == "__main__":
    main()
