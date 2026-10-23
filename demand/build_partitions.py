"""
Build the training / validation / test demand partitions.

This implements the separation guarantees stated in Chapter Three (slide 17):

  1. Test profiles and seeds never appear in training or validation.
  2. The input scaler is fitted on training data only (enforced later, in the
     model pipeline - see scripts/train_lstm.py).
  3. Chronological order is preserved within every run.
  4. Identical seeded traffic streams are replayed across all three controllers
     (enforced by reusing these route files unchanged for every controller).

Training profiles use the base turning ratios. Validation and test profiles
vary the turning ratios, so the model is selected and finally judged on
demand structures it has not been fitted to.

Outputs one route file per run, plus a manifest recording exactly which
profile and seed produced each file.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_demand import (  # noqa: E402
    build_profile, sample_departures, write_routes,
    specified_phf, peak_hour_factor, N_CYCLES, MOVEMENTS, CAPACITY,
)

# ----------------------------------------------------------------------------
# Partition definition - profile ids and seeds are disjoint by construction
# ----------------------------------------------------------------------------
PARTITIONS = {
    "train": dict(profiles=list(range(1, 13)),   seeds=list(range(1000, 1012)), vary_turning=False),
    "val":   dict(profiles=list(range(20, 26)),  seeds=list(range(2000, 2006)), vary_turning=True),
    "test":  dict(profiles=list(range(40, 48)),  seeds=list(range(3000, 3008)), vary_turning=True),
}

# ---------------------------------------------------------------------------
# Broadened partitions.
#
# The first experiment showed that the forecast did not generalise to sustained
# directional imbalance, a structure absent from the original training profiles.
# These partitions add profiles that span a range of axis imbalance so the model
# sees persistently lopsided junctions during training.
#
# The additional profiles use new profile ids and new seeds. They are generated
# independently of the six evaluation scenarios, which are not touched. Training
# on demand structures of a similar KIND to those evaluated is generalisation
# coverage; training on the evaluation runs themselves would be leakage, and is
# not done.
# ---------------------------------------------------------------------------
IMBALANCE_LEVELS = [0.06, 0.12, 0.18, 0.24, 0.30, 0.36]

PARTITIONS_BROAD = {
    "train": dict(profiles=list(range(1, 13)) + list(range(60, 72)),
                  seeds=list(range(1000, 1012)) + list(range(6000, 6012)),
                  vary_turning=False),
    "val":   dict(profiles=list(range(20, 26)) + list(range(80, 86)),
                  seeds=list(range(2000, 2006)) + list(range(7000, 7006)),
                  vary_turning=True),
    "test":  dict(profiles=list(range(40, 48)) + list(range(90, 96)),
                  seeds=list(range(3000, 3008)) + list(range(8000, 8006)),
                  vary_turning=True),
}


def imbalance_for(profile_id):
    """Profiles numbered 60 and above carry a sustained axis imbalance."""
    if profile_id < 60:
        return 0.0
    return IMBALANCE_LEVELS[(profile_id - 60) % len(IMBALANCE_LEVELS)]


def check_disjoint(partitions=None):
    partitions = partitions or PARTITIONS
    """Assert that no profile id or seed is shared between partitions."""
    ok = True
    names = list(partitions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = set(partitions[a]["profiles"]), set(partitions[b]["profiles"])
            sa, sb = set(partitions[a]["seeds"]), set(partitions[b]["seeds"])
            if pa & pb:
                print(f"  FAIL: {a} and {b} share profile ids {sorted(pa & pb)}")
                ok = False
            if sa & sb:
                print(f"  FAIL: {a} and {b} share seeds {sorted(sa & sb)}")
                ok = False
    if ok:
        print("  PASS: profile ids and seeds are disjoint across all partitions")
    return ok


def build(out_root=None, crest=0.12, noise_sd=0.02, broad=False):
    here = os.path.dirname(os.path.abspath(__file__))
    partitions = PARTITIONS_BROAD if broad else PARTITIONS
    out_root = out_root or os.path.join(
        here, "partitions_broad" if broad else "partitions")
    os.makedirs(out_root, exist_ok=True)

    if broad:
        print("BROADENED partitions: training now spans sustained axis imbalance")
    print("Checking partition separation")
    if not check_disjoint(partitions):
        raise SystemExit("partition separation violated - aborting")

    manifest = {"capacity_veh_per_h": CAPACITY, "cycles_per_run": N_CYCLES,
                "movements": MOVEMENTS, "crest": crest, "noise_sd": noise_sd,
                "partitions": {}}

    grand_total_vehicles = 0
    grand_total_records = 0

    for part, spec in partitions.items():
        part_dir = os.path.join(out_root, part)
        os.makedirs(part_dir, exist_ok=True)
        runs = []
        spec_phfs, real_phfs, sat_means = [], [], []

        for pid, seed in zip(spec["profiles"], spec["seeds"]):
            imb = imbalance_for(pid) if broad else 0.0
            rows, meta = build_profile(pid, seed,
                                       vary_turning=spec["vary_turning"],
                                       crest=crest, noise_sd=noise_sd,
                                       axis_imbalance=imb)
            vehicles = sample_departures(rows, seed, meta["turning"])
            fname = f"{part}_p{pid}_s{seed}.rou.xml"
            write_routes(vehicles, os.path.join(part_dir, fname))

            sphf = specified_phf(meta["bounds"], crest)
            rphf, _ = peak_hour_factor(vehicles)
            sat = [r["total_rate"] / CAPACITY for r in rows if r["regime"] == "saturated"]

            spec_phfs.append(sphf)
            real_phfs.append(rphf)
            sat_means.append(float(np.mean(sat)) if sat else float("nan"))
            grand_total_vehicles += len(vehicles)
            grand_total_records += N_CYCLES * len(MOVEMENTS)

            runs.append(dict(route_file=os.path.join(part, fname),
                             profile_id=pid, seed=seed,
                             dominant=meta["dominant"],
                             turning=meta["turning"],
                             axis_imbalance=imb,
                             n_vehicles=len(vehicles),
                             phf_specified=round(sphf, 4),
                             phf_realised=round(rphf, 4)))

        manifest["partitions"][part] = dict(
            vary_turning=spec["vary_turning"],
            n_runs=len(runs),
            phf_specified_mean=round(float(np.mean(spec_phfs)), 4),
            phf_specified_range=[round(float(min(spec_phfs)), 4),
                                 round(float(max(spec_phfs)), 4)],
            saturated_mean_of_capacity=round(float(np.mean(sat_means)), 4),
            runs=runs)

        print(f"\n{part}: {len(runs)} runs")
        print(f"  specified PHF   mean {np.mean(spec_phfs):.3f}  "
              f"range {min(spec_phfs):.3f}-{max(spec_phfs):.3f}")
        print(f"  realised PHF    mean {np.mean(real_phfs):.3f}")
        print(f"  saturated level mean {np.mean(sat_means):.3f} of capacity")
        print(f"  vehicles        {sum(r['n_vehicles'] for r in runs)}")

    manifest["totals"] = dict(
        runs=sum(len(p["runs"]) for p in manifest["partitions"].values()),
        vehicles=grand_total_vehicles,
        movement_records=grand_total_records)

    mpath = os.path.join(out_root, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    all_spec = [r["phf_specified"] for p in manifest["partitions"].values() for r in p["runs"]]
    print("\n" + "=" * 62)
    print(f"TOTAL runs             : {manifest['totals']['runs']}")
    print(f"TOTAL vehicles         : {manifest['totals']['vehicles']}")
    print(f"TOTAL movement records : {manifest['totals']['movement_records']}"
          f"   (Chapter 3 states approximately 20,000)")
    print(f"Specified PHF overall  : mean {np.mean(all_spec):.3f}, "
          f"range {min(all_spec):.3f}-{max(all_spec):.3f}")
    print(f"Chukwurah et al. band  : 0.79-0.93  -> "
          f"{100*np.mean([(0.79 <= p <= 0.93) for p in all_spec]):.0f}% of profiles inside")
    print(f"manifest written to    : {mpath}")
    print("=" * 62)
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--broad", action="store_true",
                    help="include profiles with sustained axis imbalance")
    a = ap.parse_args()
    build(broad=a.broad)
