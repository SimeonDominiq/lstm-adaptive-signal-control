"""
Stage 6 - build the datasets the prediction model learns from.

For every run in the demand partitions this script:

  1. runs the simulation under the calibrated fixed-time (Webster) plan,
  2. reads the induction-loop counts for each of the eight movements in each
     140 s cycle,
  3. records, alongside them, the true number of vehicles that were released
     onto that movement in that cycle,
  4. writes one dataset file per partition.

Two quantities are recorded per movement per cycle, and the distinction matters.

  detector count - what the loops actually measured. This is what a real
      installation would have, and it is what the model is trained on. It is
      an imperfect view of demand: when a queue reaches back over a loop, the
      loop counts vehicles leaving on green rather than vehicles arriving.

  true arrivals  - how many vehicles were actually released onto that movement
      in that cycle. This is only knowable inside a simulation. It is recorded
      as a reference so the gap between what the sensors see and what is really
      there can be measured and reported, rather than assumed away.

The model is trained on detector counts, because that is what a deployed system
would receive. The true arrivals are used to quantify how good that view is.

Run after the network and the Webster plan are in place:
    python3 scripts/extract_dataset.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "demand"))

from generate_demand import (  # noqa: E402
    build_profile, sample_departures, CYCLE, N_CYCLES, MOVEMENTS,
)

DETECTOR_PREFIX = "det_"
PARTITION_DIR = [os.path.join(ROOT, "demand", "partitions")]


def detector_additional(path, out_name="detector_out.xml"):
    """Write a detector definition whose output goes to a run-specific file."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<additional>"]
    lane_for = {
        "NB_L": "S2C_1", "NB_TR": "S2C_0",
        "SB_L": "N2C_1", "SB_TR": "N2C_0",
        "EB_L": "W2C_1", "EB_TR": "W2C_0",
        "WB_L": "E2C_1", "WB_TR": "E2C_0",
    }
    for mv in MOVEMENTS:
        lines.append(
            f'    <inductionLoop id="{DETECTOR_PREFIX}{mv}" lane="{lane_for[mv]}" '
            f'pos="200" period="{int(CYCLE)}" file="{out_name}"/>')
    lines += ["</additional>", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_simulation(net, route, workdir, end=None):
    """Run one simulation in its own directory and return the detector output path."""
    det_add = os.path.join(workdir, "det.add.xml")
    detector_additional(det_add)
    cmd = [
        "sumo",
        "-n", os.path.abspath(net),
        "-r", os.path.abspath(route),
        "-a", os.path.abspath(det_add),
        "--time-to-teleport", "-1",
        "--no-step-log", "true",
        "--no-warnings", "true",
    ]
    if end is not None:
        cmd += ["--end", str(end)]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sumo failed for {route}\n{proc.stderr[:2000]}")
    out = os.path.join(workdir, "detector_out.xml")
    if not os.path.exists(out):
        raise RuntimeError(f"no detector output produced for {route}")
    return out


def parse_detectors(path):
    """detector counts -> {movement: {cycle_index: count}}"""
    counts = {m: {} for m in MOVEMENTS}
    for iv in ET.parse(path).getroot():
        if iv.tag != "interval":
            continue
        det_id = iv.get("id", "")
        if not det_id.startswith(DETECTOR_PREFIX):
            continue
        mv = det_id[len(DETECTOR_PREFIX):]
        if mv not in counts:
            continue
        begin = float(iv.get("begin"))
        idx = int(round(begin / CYCLE))
        n = iv.get("nVehContrib")
        counts[mv][idx] = int(float(n)) if n is not None else 0
    return counts


def true_arrivals(profile_id, seed, vary_turning):
    """How many vehicles were actually released onto each movement in each cycle."""
    rows, meta = build_profile(profile_id, seed, vary_turning=vary_turning)
    vehicles = sample_departures(rows, seed, meta["turning"])
    arr = {m: np.zeros(N_CYCLES, dtype=int) for m in MOVEMENTS}
    for v in vehicles:
        idx = int(v["depart"] // CYCLE)
        if 0 <= idx < N_CYCLES and v["movement"] in arr:
            arr[v["movement"]][idx] += 1
    regimes = [r["regime"] for r in rows]
    return arr, regimes


def build_partition(part, runs, net, keep_going=False):
    records = []
    for i, run in enumerate(runs, 1):
        route = os.path.join(PARTITION_DIR[0], run["route_file"])
        if not os.path.exists(route):
            raise SystemExit(
                f"route file missing: {route}\nRun: python3 demand/build_partitions.py")

        workdir = tempfile.mkdtemp(prefix="sumorun_")
        try:
            det_path = run_simulation(net, route, workdir)
            det = parse_detectors(det_path)
        except Exception as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            if keep_going:
                print(f"    run {i} FAILED: {exc}")
                continue
            raise
        shutil.rmtree(workdir, ignore_errors=True)

        arr, regimes = true_arrivals(run["profile_id"], run["seed"],
                                     vary_turning=(part != "train"))

        for c in range(N_CYCLES):
            rec = dict(partition=part, profile_id=run["profile_id"],
                       seed=run["seed"], cycle=c,
                       t_start=c * CYCLE,
                       regime=regimes[c] if c < len(regimes) else "")
            for m in MOVEMENTS:
                rec[f"det_{m}"] = det[m].get(c, 0)
                rec[f"true_{m}"] = int(arr[m][c])
            records.append(rec)

        print(f"    run {i}/{len(runs)}  profile {run['profile_id']:>3} "
              f"seed {run['seed']}  cycles {N_CYCLES}")
    return records


def save(records, part, outdir):
    os.makedirs(outdir, exist_ok=True)
    det_cols = [f"det_{m}" for m in MOVEMENTS]
    true_cols = [f"true_{m}" for m in MOVEMENTS]

    # CSV, readable and auditable
    csv_path = os.path.join(outdir, f"{part}.csv")
    header = ["partition", "profile_id", "seed", "cycle", "t_start", "regime"] + det_cols + true_cols
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in records:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    # arrays, grouped by run so sequences never cross a run boundary
    runs = sorted({(r["profile_id"], r["seed"]) for r in records})
    det_stack, true_stack, run_ids = [], [], []
    for pid, seed in runs:
        rows = [r for r in records if r["profile_id"] == pid and r["seed"] == seed]
        rows.sort(key=lambda r: r["cycle"])
        det_stack.append(np.array([[r[c] for c in det_cols] for r in rows], dtype=np.float32))
        true_stack.append(np.array([[r[c] for c in true_cols] for r in rows], dtype=np.float32))
        run_ids.append((pid, seed))

    npz_path = os.path.join(outdir, f"{part}.npz")
    np.savez_compressed(
        npz_path,
        detector=np.stack(det_stack),          # (runs, cycles, 8)
        true=np.stack(true_stack),             # (runs, cycles, 8)
        run_ids=np.array(run_ids),
        movements=np.array(MOVEMENTS),
    )
    return csv_path, npz_path, np.stack(det_stack), np.stack(true_stack)


def report(part, det, true):
    d = det.reshape(-1, len(MOVEMENTS))
    t = true.reshape(-1, len(MOVEMENTS))
    total_d, total_t = d.sum(), t.sum()
    ratio = total_d / total_t if total_t else float("nan")

    flat_d, flat_t = d.ravel(), t.ravel()
    corr = float(np.corrcoef(flat_d, flat_t)[0, 1]) if flat_t.std() > 0 else float("nan")

    print(f"    cycles {det.shape[0] * det.shape[1]}  "
          f"records {det.shape[0] * det.shape[1] * len(MOVEMENTS)}")
    print(f"    detector total {int(total_d)}   true total {int(total_t)}   "
          f"ratio {ratio:.3f}")
    print(f"    correlation between detector count and true arrivals: {corr:.3f}")
    return dict(ratio=round(float(ratio), 4), correlation=round(corr, 4),
                records=int(det.shape[0] * det.shape[1] * len(MOVEMENTS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--partitions", default=os.path.join(ROOT, "demand", "partitions"),
                    help="partition directory (use demand/partitions_broad for the "
                         "broadened training set)")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    net = os.path.join(ROOT, "network", "intersection.net.xml")
    if not os.path.exists(net):
        raise SystemExit("network not built. Run: bash network/build_network.sh "
                         "then python3 scripts/make_tls.py")
    plan = os.path.join(ROOT, "network", "webster_plan.json")
    if not os.path.exists(plan):
        print("WARNING: network/webster_plan.json not found.")
        print("The dataset should be generated under the calibrated plan.")
        print("Run: python3 scripts/webster_calibrate.py --write")
        print()

    manifest_path = os.path.join(args.partitions, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("run: python3 demand/build_partitions.py")
    PARTITION_DIR[0] = args.partitions
    manifest = json.load(open(manifest_path))

    summary = {}
    for part, spec in manifest["partitions"].items():
        print(f"\n{part}: {spec['n_runs']} runs")
        records = build_partition(part, spec["runs"], net, keep_going=args.keep_going)
        csv_path, npz_path, det, true = save(records, part, args.outdir)
        summary[part] = report(part, det, true)
        summary[part]["csv"] = os.path.relpath(csv_path, ROOT)
        summary[part]["npz"] = os.path.relpath(npz_path, ROOT)

    total = sum(v["records"] for v in summary.values())
    print("\n" + "=" * 62)
    print(f"TOTAL movement records : {total}")
    print("Datasets written to    :", os.path.relpath(args.outdir, ROOT))
    print("=" * 62)

    spath = os.path.join(args.outdir, "dataset_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary written to {os.path.relpath(spath, ROOT)}")


if __name__ == "__main__":
    main()
