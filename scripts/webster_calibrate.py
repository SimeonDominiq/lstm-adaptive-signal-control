"""
Webster calibration of the fixed-time baseline.

Chapter Three specifies that the fixed-time controller is calibrated using
Webster's method, so that the baseline is a competently timed plan rather than
a straw man. This script derives that plan from the demand profiles and writes
it into the network.

Method (Webster, 1958):

    y_i = q_i / s          flow ratio of the critical movement in phase i
    Y   = sum of y_i       total critical flow ratio
    L   = total lost time  = n_phases x (yellow + all-red)
    C   = (1.5 L + 5) / (1 - Y)      optimum cycle length
    g_i = (y_i / Y) x (C - L)        effective green per phase

Two departures from textbook Webster are made deliberately, and both are
recorded in Chapter Four:

  1. The cycle length is FIXED at 140 s rather than set to Webster's optimum.
     Chapter Three commits every controller to the same 140 s cycle so that
     the three controllers are compared on identical terms; letting the
     baseline choose a different cycle would confound the comparison. Webster's
     optimum is still computed and reported, so the size of this constraint is
     visible rather than hidden.

  2. Green times are clamped to the 10-60 s bounds that apply to every
     controller, and any residual is redistributed proportionally.

The critical flow ratio for a phase is taken as the maximum over the movements
that phase serves, since the phase must clear its busiest movement.
"""

import argparse
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "demand"))
sys.path.insert(0, HERE)

from generate_demand import build_profile, CAPACITY  # noqa: E402
from build_partitions import PARTITIONS              # noqa: E402
import make_tls                                       # noqa: E402

SATURATION_FLOW = 1800.0      # veh/h per lane
LANES_PER_MOVEMENT = 1        # each movement has its own lane


def critical_flow_ratios(demand_by_movement, phase_movements):
    """
    y_i for each phase = max over its movements of (movement flow / saturation flow).
    Each movement is served by one lane, so the saturation flow is per lane.
    """
    ys = []
    for movs in phase_movements:
        y = max(demand_by_movement[m] / (SATURATION_FLOW * LANES_PER_MOVEMENT)
                for m in movs)
        ys.append(y)
    return ys


def webster_cycle(ys, lost_time):
    """Webster's optimum cycle length. Returns None if the intersection is oversaturated."""
    Y = sum(ys)
    if Y >= 1.0:
        return None, Y
    return (1.5 * lost_time + 5.0) / (1.0 - Y), Y


def allocate_green(ys, green_available, g_min, g_max):
    """
    Split the available green in proportion to the critical flow ratios,
    then clamp to bounds and redistribute any residual proportionally.
    """
    ys = np.asarray(ys, dtype=float)
    if ys.sum() <= 0:
        g = np.full(len(ys), green_available / len(ys))
    else:
        g = green_available * ys / ys.sum()

    for _ in range(50):
        clamped = np.clip(g, g_min, g_max)
        residual = green_available - clamped.sum()
        if abs(residual) < 1e-6:
            g = clamped
            break
        free = (clamped > g_min) & (clamped < g_max)
        if residual > 0:
            free = clamped < g_max
        else:
            free = clamped > g_min
        if not free.any():
            g = clamped
            break
        w = ys[free] if ys[free].sum() > 0 else np.ones(free.sum())
        clamped[free] = clamped[free] + residual * w / w.sum()
        g = clamped

    g = np.clip(g, g_min, g_max)
    # integer seconds, preserving the total
    gi = np.floor(g).astype(int)
    while gi.sum() < int(round(green_available)):
        gi[np.argmax(g - gi)] += 1
    while gi.sum() > int(round(green_available)):
        gi[np.argmin(g - gi)] -= 1
    return gi.tolist()


def peak_demand_by_movement(partition="train"):
    """
    Design demand for calibration: the mean of the per-movement demand during
    the saturated regime, averaged across the training profiles.

    Only training profiles are used. Calibrating on validation or test demand
    would leak information from the held-out partitions into the baseline,
    which would undermine the separation guarantees.
    """
    spec = PARTITIONS[partition]
    totals = {}
    n = 0
    for pid, seed in zip(spec["profiles"], spec["seeds"]):
        rows, meta = build_profile(pid, seed, vary_turning=spec["vary_turning"])
        sat = [r for r in rows if r["regime"] == "saturated"]
        if not sat:
            continue
        for r in sat:
            for a in ("NB", "SB", "EB", "WB"):
                for suf in ("L", "TR"):
                    k = f"{a}_{suf}"
                    totals[k] = totals.get(k, 0.0) + r[k]
            n += 1
    return {k: v / n for k, v in totals.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="train",
                    help="which partition's demand to calibrate on (default: train)")
    ap.add_argument("--write", action="store_true",
                    help="write the plan into the network and the additional file")
    args = ap.parse_args()

    net_path = os.path.join(ROOT, "network", "intersection.net.xml")
    movement_links, all_idx, unmapped = make_tls.read_network(net_path)
    n_links = max(all_idx) + 1
    if unmapped:
        raise SystemExit("unmapped connections - run scripts/make_tls.py first")

    demand, n_cycles = peak_demand_by_movement(args.partition)

    print(f"Design demand (mean over the saturated regime, {args.partition} profiles, "
          f"{n_cycles} cycles)")
    for m in make_tls.MOVEMENTS:
        print(f"   {m:<6} {demand[m]:7.1f} veh/h")
    print(f"   {'total':<6} {sum(demand.values()):7.1f} veh/h "
          f"({sum(demand.values())/CAPACITY:.2f} of capacity)")

    ys = critical_flow_ratios(demand, make_tls.PHASE_MOVEMENTS)
    lost_time = 4 * (make_tls.YELLOW + make_tls.ALLRED)
    C_opt, Y = webster_cycle(ys, lost_time)

    print(f"\nWebster calculation")
    for i, (name, y) in enumerate(zip(make_tls.PHASE_NAMES, ys)):
        crit = max(make_tls.PHASE_MOVEMENTS[i],
                   key=lambda m: demand[m])
        print(f"   phase {i+1} {name:<18} critical movement {crit:<6} y = {y:.4f}")
    print(f"   Y (sum of critical flow ratios) = {Y:.4f}")
    print(f"   L (lost time)                   = {lost_time} s")
    if C_opt is None:
        print("   Webster optimum cycle           = undefined (Y >= 1, oversaturated)")
    else:
        print(f"   Webster optimum cycle C         = (1.5 x {lost_time} + 5) / (1 - {Y:.4f})"
              f" = {C_opt:.1f} s")

    green_available = make_tls.CYCLE - lost_time
    print(f"\n   Cycle constrained to {make_tls.CYCLE} s (Chapter 3), so available green "
          f"= {make_tls.CYCLE} - {lost_time} = {green_available} s")

    greens = allocate_green(ys, green_available, make_tls.G_MIN, make_tls.G_MAX)
    print(f"\nCalibrated green times")
    for i, (name, g) in enumerate(zip(make_tls.PHASE_NAMES, greens)):
        share = ys[i] / sum(ys) if sum(ys) > 0 else 0.25
        print(f"   phase {i+1} {name:<18} {g:>3} s   (proportional share {share:.3f})")
    print(f"   total green {sum(greens)} s + {lost_time} s lost = "
          f"{sum(greens) + lost_time} s cycle")

    # capacity check per phase
    print(f"\nCapacity check (per phase, against the critical movement)")
    shortfalls = []
    for i, (name, g) in enumerate(zip(make_tls.PHASE_NAMES, greens)):
        cap = SATURATION_FLOW * (g / make_tls.CYCLE)
        crit = max(make_tls.PHASE_MOVEMENTS[i], key=lambda m: demand[m])
        dem = demand[crit]
        ratio = dem / cap if cap > 0 else float("inf")
        flag = "OK " if ratio <= 1.0 else "OVER"
        if ratio > 1.0:
            shortfalls.append((name, ratio))
        print(f"   {flag} phase {i+1} {name:<18} capacity {cap:6.1f} veh/h  "
              f"demand {dem:6.1f} veh/h  v/c {ratio:.2f}")

    if shortfalls:
        print("\n   NOTE: one or more phases remain oversaturated at the design demand.")
        print("   The 140 s cycle constraint limits what any fixed plan can achieve here;")
        print("   this is a genuine property of the scenario, not a calibration error.")

    if args.write:
        phases = make_tls.build_phases(greens, movement_links, n_links)
        total = sum(p[0] for p in phases)
        if total != make_tls.CYCLE:
            raise SystemExit(f"cycle is {total}s, expected {make_tls.CYCLE}s")
        add_out = os.path.join(ROOT, "network", "tls_webster.add.xml")
        make_tls.write_tls(phases, add_out, program_id="webster")
        make_tls.patch_net(net_path, phases, program_id="0")
        print(f"\n   wrote {add_out}  (programID 'webster')")
        print(f"   patched {net_path}  (programID '0' now carries the Webster plan)")

        meta = dict(method="Webster (1958)", partition=args.partition,
                    design_demand_veh_per_h=demand,
                    critical_flow_ratios=dict(zip(make_tls.PHASE_NAMES, ys)),
                    Y=Y, lost_time_s=lost_time,
                    webster_optimum_cycle_s=C_opt,
                    cycle_used_s=make_tls.CYCLE,
                    green_times_s=dict(zip(make_tls.PHASE_NAMES, greens)),
                    note="Cycle constrained to 140 s for comparability across controllers.")
        mpath = os.path.join(ROOT, "network", "webster_plan.json")
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"   wrote {mpath}")
    else:
        print("\n   (dry run - pass --write to apply this plan)")


if __name__ == "__main__":
    main()
