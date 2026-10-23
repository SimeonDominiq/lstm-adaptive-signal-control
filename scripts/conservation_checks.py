"""
Conservation checks for the simulation and control apparatus.

Every fault found in this project produced plausible output. Nothing crashed,
every run completed, every figure had a sensible shape. Each was eventually
caught not by reading code but by checking a quantity that must be true by
definition and finding that it was not.

This module collects those checks in one place so they can be run after any
change and before any result is used. Each check is arithmetic that cannot be
argued with: either the identity holds or something is wrong.

    CHECK                        WOULD HAVE CAUGHT
    detector counts vs released  the occupancy fault and the boundary
                                 double-count, both of which made the loops
                                 register more vehicles than existed
    realised vs specified cycle  the phase extension, which gave the adaptive
                                 controllers a longer cycle than the baseline
    equal cycles across          the same fault, by showing the controllers
    controllers                  were not comparable
    green times sum and bounds   any allocation that silently violated the
                                 specification
    specification against code   the divergence between the summed phase demand
                                 in Chapter Three and the maximum used in code

Run everything:
    python3 scripts/conservation_checks.py

Run one group:
    python3 scripts/conservation_checks.py --only control-law
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
sys.path.insert(0, HERE)

RESULTS = []


def record(group, name, passed, detail):
    RESULTS.append(dict(group=group, name=name, passed=passed, detail=detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}")
    if detail:
        for line in str(detail).split("\n"):
            print(f"         {line}")


# ---------------------------------------------------------------------------
# 1.  The control law
# ---------------------------------------------------------------------------
def check_control_law():
    print("\ncontrol law")
    src_path = os.path.join(HERE, "run_controller.py")
    if not os.path.exists(src_path):
        record("control-law", "run_controller.py present", False, src_path)
        return
    src = open(src_path).read()

    # extract allocate_green without importing traci
    start = src.index("def allocate_green")
    end = src.index("# ---------------------------------------------------------------------------\n# demand estimators")
    ns = {"np": np}
    exec(src[start:end], ns)
    allocate = ns["allocate_green"]

    MOV = ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]
    PM = [["NB_TR", "SB_TR"], ["NB_L", "SB_L"], ["EB_TR", "WB_TR"], ["EB_L", "WB_L"]]
    G, GMIN, GMAX = 120, 10, 60

    cases = {
        "balanced": {m: 20 for m in MOV},
        "through-heavy": {"NB_L": 4, "NB_TR": 24, "SB_L": 4, "SB_TR": 26,
                          "EB_L": 4, "EB_TR": 22, "WB_L": 4, "WB_TR": 21},
        "one phase dominant": {"NB_L": 1, "NB_TR": 200, "SB_L": 1, "SB_TR": 200,
                               "EB_L": 1, "EB_TR": 2, "WB_L": 1, "WB_TR": 2},
        "no demand": {m: 0 for m in MOV},
        "left turns only": {"NB_L": 10, "NB_TR": 0, "SB_L": 12, "SB_TR": 0,
                            "EB_L": 9, "EB_TR": 0, "WB_L": 8, "WB_TR": 0},
        "single movement": {m: (50 if m == "SB_TR" else 0) for m in MOV},
    }
    bad = []
    for name, d in cases.items():
        g = allocate(d, PM, G, GMIN, GMAX)
        if sum(g) != G:
            bad.append(f"{name}: sums to {sum(g)}, expected {G}")
        if not all(GMIN <= x <= GMAX for x in g):
            bad.append(f"{name}: {g} violates the {GMIN}-{GMAX} bounds")
    record("control-law", f"green times sum to {G}s and respect bounds "
           f"across {len(cases)} demand patterns",
           not bad, "\n".join(bad))

    # the aggregation rule actually implemented
    probe = {"NB_L": 0, "NB_TR": 18, "SB_L": 0, "SB_TR": 22,
             "EB_L": 0, "EB_TR": 0, "WB_L": 0, "WB_TR": 0}
    g = allocate(probe, PM, G, GMIN, GMAX)
    # under MAX phase 1 demand is 22; under SUM it is 40. With all other phases
    # at zero both give phase 1 the maximum, so probe a case that separates them.
    probe2 = {"NB_L": 0, "NB_TR": 18, "SB_L": 0, "SB_TR": 22,
              "EB_L": 0, "EB_TR": 40, "WB_L": 0, "WB_TR": 0}
    g2 = allocate(probe2, PM, G, GMIN, GMAX)
    # phase1 demand: MAX -> 22, SUM -> 40 ; phase3 demand: MAX -> 40, SUM -> 40
    rule = "MAX" if g2[0] < g2[2] else ("SUM" if g2[0] == g2[2] else "unknown")
    record("control-law", f"implemented aggregation rule is {rule}",
           rule == "MAX",
           "Chapter Three must state the same rule. Phase demands 18,22 against "
           "40,0 give unequal green under MAX and equal green under SUM.\n"
           f"observed greens {g2}")


# ---------------------------------------------------------------------------
# 2.  Timing
# ---------------------------------------------------------------------------
def check_timing():
    print("\ntiming")
    mpath = os.path.join(ROOT, "network", "movement_map.json")
    npath = os.path.join(ROOT, "network", "intersection.net.xml")
    if not os.path.exists(mpath) or not os.path.exists(npath):
        record("timing", "network and movement map present", False,
               "build the network and run make_tls.py first")
        return
    m = json.load(open(mpath))
    cycle, green_total = int(m["cycle"]), int(m["green_total"])
    yellow, allred = int(m["yellow"]), int(m["allred"])
    n_phases = len(m["phase_names"])

    expected = green_total + n_phases * (yellow + allred)
    record("timing", f"green plus intergreen equals the specified cycle",
           expected == cycle,
           f"{green_total} + {n_phases}x({yellow}+{allred}) = {expected}, "
           f"specified {cycle}")

    import xml.etree.ElementTree as ET
    root = ET.parse(npath).getroot()
    for tl in root.findall("tlLogic"):
        if tl.get("id") != m["tls_id"]:
            continue
        total = sum(float(p.get("duration")) for p in tl.findall("phase"))
        record("timing", f"programme in the network sums to the specified cycle",
               abs(total - cycle) < 0.01,
               f"programme totals {total}s, specified {cycle}s")
        break


# ---------------------------------------------------------------------------
# 3.  A live run: conservation and equal treatment
# ---------------------------------------------------------------------------
def run_once(controller, route, model_dir, end):
    workdir = tempfile.mkdtemp(prefix="cons_")
    try:
        label = f"cons_{controller}"
        cmd = [sys.executable, os.path.join(HERE, "run_controller.py"),
               "--controller", controller, "--route", route,
               "--label", label, "--outdir", workdir, "--end", str(end)]
        if controller == "predictive" and model_dir and os.path.isdir(model_dir):
            cmd += ["--model-dir", model_dir]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            return None, proc.stderr[-700:]
        p = os.path.join(workdir, f"{label}_cycles.json")
        return (json.load(open(p)) if os.path.exists(p) else None), None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def check_live(end):
    print("\nlive run")
    from generate_demand import CYCLE, MOVEMENTS, turning_ratios, sample_departures
    from build_scenarios import SCENARIOS, build_rows

    man = os.path.join(ROOT, "demand", "scenarios", "manifest.json")
    if not os.path.exists(man):
        record("live", "scenario manifest present", False, man)
        return
    manifest = json.load(open(man))
    scen = "low" if "low" in manifest["scenarios"] else list(manifest["scenarios"])[0]
    spec = manifest["scenarios"][scen]["runs"][0]
    seed = spec["seed"]
    route = os.path.join(ROOT, "demand", "scenarios", spec["route_file"])
    model_dir = os.path.join(ROOT, "models_broad")

    # vehicles released within the window examined
    turns = turning_ratios(seed, True)
    rows, _ = build_rows(SCENARIOS[scen], seed, turns)
    vehicles = sample_departures(rows, seed, turns)

    logs, intervals = {}, {}
    for c in ["fixed", "reactive", "predictive"]:
        cy, err = run_once(c, route, model_dir, end)
        if cy is None:
            record("live", f"{c} controller completes a run", False, err)
            continue
        logs[c] = cy
        ts = [r["t"] for r in sorted(cy, key=lambda r: r["cycle"])]
        intervals[c] = np.diff(ts) if len(ts) > 1 else np.array([])

    if not logs:
        return

    # 3a. detector counts cannot exceed vehicles released in the same window
    for c, cy in logs.items():
        last_t = max(r["t"] for r in cy)
        released = sum(1 for v in vehicles if v["depart"] < last_t)
        counted = sum(sum(r["counts"].values()) for r in cy)
        record("live", f"{c}: detector counts do not exceed vehicles released",
               counted <= released,
               f"counted {counted}, released before t={last_t}: {released}")

    # 3b. realised cycle equals the specified cycle
    m = json.load(open(os.path.join(ROOT, "network", "movement_map.json")))
    cycle = int(m["cycle"])
    for c, iv in intervals.items():
        if len(iv) == 0:
            continue
        ok = bool(np.all(np.abs(iv - cycle) <= 1))
        record("live", f"{c}: realised cycle equals the specified {cycle}s",
               ok, f"observed intervals {sorted(set(iv.tolist()))}")

    # 3c. all controllers run the same cycle as each other
    means = {c: float(iv.mean()) for c, iv in intervals.items() if len(iv)}
    if len(means) > 1:
        spread = max(means.values()) - min(means.values())
        record("live", "all controllers run the same cycle length",
               spread <= 1.0,
               "  ".join(f"{c} {v:.1f}s" for c, v in means.items()) +
               f"   spread {spread:.1f}s")

    # 3d. applied green times obey the specification in every logged cycle
    gmin, gmax = int(m["g_min"]), int(m["g_max"])
    gtot = int(m["green_total"])
    for c, cy in logs.items():
        bad = 0
        for r in cy:
            g = r.get("greens")
            if not g:
                continue
            if sum(g) != gtot or not all(gmin <= x <= gmax for x in g):
                bad += 1
        record("live", f"{c}: every applied green plan sums to {gtot}s within bounds",
               bad == 0, f"{bad} cycles violated the specification")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["control-law", "timing", "live"], default=None)
    ap.add_argument("--end", type=int, default=1400,
                    help="simulated seconds for the live checks (default: ten cycles)")
    args = ap.parse_args()

    print("CONSERVATION CHECKS")
    print("=" * 62)

    if args.only in (None, "control-law"):
        check_control_law()
    if args.only in (None, "timing"):
        check_timing()
    if args.only in (None, "live"):
        check_live(args.end)

    passed = sum(1 for r in RESULTS if r["passed"])
    failed = [r for r in RESULTS if not r["passed"]]
    print("\n" + "=" * 62)
    print(f"{passed} passed, {len(failed)} failed")
    if failed:
        print("\nfailures:")
        for r in failed:
            print(f"  {r['group']}: {r['name']}")
        print("\nNo result should be used until these are resolved.")
    print("=" * 62)

    out = os.path.join(ROOT, "results", "conservation_checks.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"wrote {os.path.relpath(out, ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
