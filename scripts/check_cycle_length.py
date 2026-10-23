"""
Locate the source of the 144 second cycle.

The controller's log shows a uniform 144 second interval between cycles, with no
variance across 99 intervals, against a nominal 140 seconds. This is not a
sampling artefact. Either the signal programme written into the network sums to
144 seconds, or the programme sums to 140 and something in execution adds four
seconds per cycle.

This script separates those two possibilities without assuming either. It does
three independent things:

  A.  Reads the phase durations out of the built network file and sums them.
      This is what the simulator is asked to execute, and it involves no
      simulation at all.

  B.  Runs the simulator with no controller attached and records the signal
      state at every step, measuring the cycle length the simulator actually
      produces. If A and B agree, the programme is the whole story.

  C.  Reports the phase-by-phase durations from both sources side by side, so
      that any discrepancy can be attributed to a particular phase rather than
      to the cycle as a whole.

The distinction matters because the two causes have different consequences. A
programme that sums to 144 means the generator writing it is wrong and its
verification is passing on something it does not write. A programme that sums to
140 but executes in 144 means the discrepancy is introduced during the run and
the signal is being extended by something other than its own definition.

    python3 scripts/check_cycle_length.py
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

TLS_ID = "C"


# ---------------------------------------------------------------------------
# A.  what the network file says
# ---------------------------------------------------------------------------
def read_programme(net_path):
    if not os.path.exists(net_path):
        raise SystemExit(f"network not found: {net_path}")
    root = ET.parse(net_path).getroot()
    programmes = []
    for tl in root.findall("tlLogic"):
        if tl.get("id") != TLS_ID:
            continue
        phases = []
        for ph in tl.findall("phase"):
            phases.append(dict(duration=float(ph.get("duration")),
                               state=ph.get("state"),
                               name=ph.get("name") or ""))
        programmes.append(dict(programID=tl.get("programID"),
                               type=tl.get("type"), phases=phases))
    return programmes


# ---------------------------------------------------------------------------
# B.  what the simulator executes, with no controller attached
# ---------------------------------------------------------------------------
def observe_cycle(net_path, route_path, detectors_path, steps=2000):
    import traci
    workdir = tempfile.mkdtemp(prefix="cyc_")
    try:
        cmd = ["sumo", "-n", os.path.abspath(net_path),
               "-r", os.path.abspath(route_path),
               "-a", os.path.abspath(detectors_path),
               "--time-to-teleport", "-1", "--no-step-log", "true",
               "--no-warnings", "true", "--end", str(steps)]
        traci.start(cmd)
        phase_at = []
        for t in range(steps):
            phase_at.append(traci.trafficlight.getPhase(TLS_ID))
            traci.simulationStep()
        traci.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # durations of each observed run of a constant phase index
    runs = []
    cur, start = phase_at[0], 0
    for t in range(1, len(phase_at)):
        if phase_at[t] != cur:
            runs.append((cur, t - start))
            cur, start = phase_at[t], t
    # cycle starts: each time the index returns to 0
    starts = [t for t in range(1, len(phase_at))
              if phase_at[t] == 0 and phase_at[t - 1] != 0]
    intervals = [b - a for a, b in zip(starts, starts[1:])]
    return runs, starts, intervals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(ROOT, "network", "intersection.net.xml"))
    ap.add_argument("--route", default=None)
    ap.add_argument("--detectors", default=os.path.join(ROOT, "detectors", "detectors.add.xml"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--skip-sim", action="store_true",
                    help="do part A only, without running the simulator")
    args = ap.parse_args()

    route = args.route
    if route is None:
        cand = os.path.join(ROOT, "demand", "routes.rou.xml")
        if os.path.exists(cand):
            route = cand
        else:
            import json
            man = os.path.join(ROOT, "demand", "scenarios", "manifest.json")
            if os.path.exists(man):
                m = json.load(open(man))
                first = list(m["scenarios"].values())[0]["runs"][0]
                route = os.path.join(ROOT, "demand", "scenarios", first["route_file"])

    # ---- A ----------------------------------------------------------------
    print("A.  PHASE DURATIONS AS WRITTEN IN THE NETWORK FILE")
    programmes = read_programme(args.net)
    if not programmes:
        raise SystemExit(f"no tlLogic with id {TLS_ID} found in {args.net}")
    for prog in programmes:
        total = sum(p["duration"] for p in prog["phases"])
        print(f"    programID {prog['programID']!r}   type {prog['type']!r}   "
              f"{len(prog['phases'])} phases")
        for i, p in enumerate(prog["phases"]):
            print(f"      {i:>2}  {p['duration']:>6.1f} s   {p['state']}   {p['name']}")
        print(f"      {'':>2}  {total:>6.1f} s   TOTAL")
        greens = sum(p["duration"] for p in prog["phases"] if "G" in p["state"])
        inter = total - greens
        print(f"      green {greens:.1f} s   intergreen {inter:.1f} s   cycle {total:.1f} s")
        print()

    active = programmes[0]
    written_cycle = sum(p["duration"] for p in active["phases"])

    if args.skip_sim:
        print("--skip-sim given; parts B and C not run")
        return

    if route is None or not os.path.exists(route):
        raise SystemExit("no route file found; pass --route explicitly")

    # ---- B ----------------------------------------------------------------
    print("B.  CYCLE LENGTH AS EXECUTED, WITH NO CONTROLLER ATTACHED")
    runs, starts, intervals = observe_cycle(args.net, route, args.detectors, args.steps)
    if not intervals:
        print("    too few cycles observed; increase --steps")
        return
    uniq = sorted(set(intervals))
    print(f"    cycle starts observed at steps  {starts[:8]}{' ...' if len(starts) > 8 else ''}")
    print(f"    intervals between starts        {uniq}  "
          f"(n={len(intervals)})")
    observed_cycle = sum(intervals) / len(intervals)
    print(f"    mean observed cycle             {observed_cycle:.2f} s")

    # ---- C ----------------------------------------------------------------
    print("\nC.  PHASE DURATIONS, WRITTEN AGAINST OBSERVED")
    first_cycle = []
    if len(starts) >= 2:
        seen = set()
        for idx, dur in runs:
            if idx in seen:
                break
            seen.add(idx)
            first_cycle.append((idx, dur))
    print(f"    {'phase':>6}{'written':>10}{'observed':>11}{'difference':>13}")
    for i, p in enumerate(active["phases"]):
        obs = next((d for (idx, d) in first_cycle if idx == i), None)
        if obs is None:
            print(f"    {i:>6}{p['duration']:>10.1f}{'not seen':>11}{'':>13}")
        else:
            print(f"    {i:>6}{p['duration']:>10.1f}{obs:>11.1f}{obs - p['duration']:>+13.1f}")

    # ---- conclusion -------------------------------------------------------
    print("\n" + "=" * 74)
    print("WHAT THIS SHOWS")
    print("=" * 74)
    print(f"    written cycle   {written_cycle:.1f} s")
    print(f"    observed cycle  {observed_cycle:.2f} s")
    if abs(written_cycle - observed_cycle) < 0.5:
        if abs(written_cycle - 140) < 0.5:
            print("    The programme sums to the nominal cycle and executes at that")
            print("    length. The 144 s seen in the controller log therefore arises in")
            print("    the controller, not in the signal, and the stepping loop is where")
            print("    to look next.")
        else:
            print("    The programme does not sum to the nominal cycle, and the simulator")
            print("    executes what the programme says. The generator that writes the")
            print("    programme is the fault, and its verification is passing on")
            print("    something it does not write.")
    else:
        print("    The programme and its execution disagree. The per-phase comparison")
        print("    in part C shows where. This is not explained by either candidate")
        print("    cause and requires further work.")


if __name__ == "__main__":
    main()
