"""
Determine whether the controller extends the signal or merely mis-records it.

The signal programme sums to 140 seconds and, with no controller attached, the
simulator executes it in exactly 140 seconds. Under the controller the logged
interval between cycles is a uniform 144 seconds. The four seconds therefore
enters somewhere in the controller, and there are two possibilities with very
different consequences.

  Mis-recording. The signal still runs at 140 seconds and the controller records
  the wrong time against each cycle. The timings applied are correct, the
  results stand, and the fault is confined to the log.

  Extension. The controller genuinely lengthens each phase, so every run has
  executed a longer cycle than specified, the demand grid and the signal drift
  apart, and the per-cycle counts are affected.

This script distinguishes them by instrumenting a run directly. It drives the
simulation itself, using the same read-phase-then-step sequence the controller
uses, and records for every phase transition:

  - the simulation time the simulator reports, taken from the simulator rather
    than from a counter maintained alongside it,
  - the loop counter the controller would hold at that moment,
  - the phase index before and after the transition.

If the simulator's own clock shows 140 seconds between cycle starts while the
counter shows 144, the fault is in the counter and nothing else is affected. If
the simulator's clock itself shows 144, the signal is being extended.

The run is short by default, since the question is settled within a few cycles.

    python3 scripts/check_controller_timing.py
    python3 scripts/check_controller_timing.py --controller reactive
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402

TLS_ID = "C"
MOVEMENTS = ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]
GREEN_PHASES = [0, 3, 6, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(ROOT, "network", "intersection.net.xml"))
    ap.add_argument("--detectors", default=os.path.join(ROOT, "detectors", "detectors.add.xml"))
    ap.add_argument("--route", default=None)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--controller", default="fixed", choices=["fixed", "reactive"],
                    help="reactive applies setPhaseDuration; fixed applies nothing")
    args = ap.parse_args()

    route = args.route
    if route is None:
        cand = os.path.join(ROOT, "demand", "routes.rou.xml")
        if os.path.exists(cand):
            route = cand
        else:
            man = os.path.join(ROOT, "demand", "scenarios", "manifest.json")
            if os.path.exists(man):
                m = json.load(open(man))
                first = list(m["scenarios"].values())[0]["runs"][0]
                route = os.path.join(ROOT, "demand", "scenarios", first["route_file"])
    if not route or not os.path.exists(route):
        raise SystemExit("no route file found; pass --route explicitly")

    cmd = ["sumo", "-n", os.path.abspath(args.net), "-r", os.path.abspath(route),
           "-a", os.path.abspath(args.detectors),
           "--time-to-teleport", "-1", "--no-step-log", "true",
           "--no-warnings", "true", "--end", str(args.steps)]
    traci.start(cmd)

    # The loop below mirrors run_controller.py: read the phase, act on a cycle
    # boundary, then step, then increment the counter.
    transitions = []       # (sim_time, counter, from_phase, to_phase)
    cycle_marks = []       # (sim_time, counter) each time the phase returns to 0
    counter = 0
    prev_phase = None

    while counter < args.steps:
        sim_t = traci.simulation.getTime()
        phase = traci.trafficlight.getPhase(TLS_ID)

        if prev_phase is not None and phase != prev_phase:
            transitions.append((sim_t, counter, prev_phase, phase))
            if phase == 0:
                cycle_marks.append((sim_t, counter))

        # the reactive variant applies a duration, as the adaptive controllers do
        if args.controller == "reactive" and prev_phase is not None \
                and phase != prev_phase and phase in GREEN_PHASES:
            k = GREEN_PHASES.index(phase)
            traci.trafficlight.setPhaseDuration(TLS_ID, [53, 10, 47, 10][k])

        prev_phase = phase
        traci.simulationStep()
        counter += 1

    traci.close()

    print(f"controller variant: {args.controller}\n")

    if len(cycle_marks) < 2:
        print("too few cycles observed; increase --steps")
        return

    sim_times = [m[0] for m in cycle_marks]
    counters = [m[1] for m in cycle_marks]
    sim_iv = np.diff(sim_times)
    cnt_iv = np.diff(counters)

    print("1.  CYCLE STARTS")
    print(f"    {'simulator time':>16}{'loop counter':>15}{'difference':>13}")
    for st, ct in cycle_marks[:8]:
        print(f"    {st:>16.1f}{ct:>15}{ct - st:>+13.1f}")
    if len(cycle_marks) > 8:
        print(f"    ... {len(cycle_marks) - 8} more")

    print("\n2.  INTERVAL BETWEEN CYCLE STARTS")
    print(f"    by the simulator's clock   {sorted(set(sim_iv.tolist()))}   "
          f"mean {sim_iv.mean():.2f} s")
    print(f"    by the loop counter        {sorted(set(cnt_iv.tolist()))}   "
          f"mean {cnt_iv.mean():.2f}")

    print("\n3.  PHASE DURATIONS BY THE SIMULATOR'S CLOCK, FIRST FULL CYCLE")
    if len(cycle_marks) >= 2:
        a, b = cycle_marks[0][0], cycle_marks[1][0]
        seg = [t for t in transitions if a <= t[0] <= b]
        prev_t = a
        total = 0.0
        for st, ct, frm, to in seg[1:] if seg and seg[0][0] == a else seg:
            dur = st - prev_t
            if dur > 0:
                print(f"    phase {frm:>2} ran {dur:>6.1f} s")
                total += dur
            prev_t = st
        print(f"    {'':>9}total {total:>6.1f} s")

    print("\n" + "=" * 74)
    print("WHAT THIS SHOWS")
    print("=" * 74)
    sim_mean, cnt_mean = sim_iv.mean(), cnt_iv.mean()
    if abs(sim_mean - cnt_mean) < 0.5:
        if abs(sim_mean - 140) < 0.5:
            print("    Both the simulator and the counter report the nominal cycle. The")
            print("    144 s in the stored logs did not arise here, and the difference")
            print("    between this run and those logs must be accounted for before any")
            print("    conclusion is drawn.")
        else:
            print(f"    Both report {sim_mean:.1f} s. The signal is genuinely running at")
            print("    that length under the controller, so the cycle is being extended")
            print("    and the effect on the demand alignment must be assessed.")
    else:
        print(f"    The simulator reports {sim_mean:.1f} s between cycle starts while the")
        print(f"    loop counter reports {cnt_mean:.1f}. The signal is running at the")
        print("    length the simulator reports; the counter is drifting against it, so")
        print("    the timings applied are correct and the fault is confined to the")
        print("    times recorded in the log. Results are unaffected; the log and any")
        print("    statement derived from it require correction.")


if __name__ == "__main__":
    main()
