"""
Online prediction error, by controller and scenario.

The accuracy reported in Section 5.3.2 was measured offline, on data generated
under fixed-time control. That does not describe the model's accuracy while it
is being used, when the counts it receives are produced under whichever
controller is running. This script measures that.

WHY NO CHANGE TO THE CONTROLLER IS NEEDED
The forecast is a pure function of the history of detector counts, and every
controller already logs those counts each cycle. Feeding a run's logged counts
through the model afterwards therefore produces exactly the forecast a
'shadow' model would have produced had it run alongside the controller without
influencing it. run_controller.py, which produced the reported results, is not
modified.

THE CONSISTENCY CHECK
Under the prediction-informed controller the in-loop forecast is already
logged, as 'estimate'. The forecast recomputed here must match it. If it does,
the offline recomputation is shown to reproduce the online forecast exactly,
and the figures for the other two controllers can be trusted on the same basis.

WHAT IS MEASURED
For each cycle k, the forecast made from counts up to k is compared with the
count actually registered in cycle k+1, the detector-observed count defined in
Section 3.5.2.3. Persistence, which forecasts cycle k+1 as a repeat of cycle k,
is computed from the same logs for comparison, so it can be seen whether the
model's advantage over the naive forecast survives in operation.

Usage, from the project root:
    python3 scripts/online_prediction_error.py            # run, then analyse
    python3 scripts/online_prediction_error.py --phase analyse
Resumable: runs whose logs already exist are skipped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "results", "online_error")
SUMMARY = os.path.join(ROOT, "results", "online_error_summary.json")
MODEL_DIR = os.path.join(ROOT, "models_broad")
MANIFEST = os.path.join(ROOT, "demand", "scenarios", "manifest.json")
CONTROLLERS = ["fixed", "reactive", "predictive"]
SCENARIOS = ["low", "moderate", "peak", "oversaturated", "surge", "imbalanced"]
OFFLINE_MAE = 1.778


# --------------------------------------------------------------------------
# phase 1: runs
# --------------------------------------------------------------------------
def job_list(n_seeds):
    m = json.load(open(MANIFEST))["scenarios"]
    jobs = []
    for s in SCENARIOS:
        runs = m[s]["runs"] if isinstance(m[s], dict) and "runs" in m[s] else m[s]
        for r in sorted(runs, key=lambda r: r["seed"])[:n_seeds]:
            for c in CONTROLLERS:
                jobs.append((c, s, r["seed"],
                             os.path.join(ROOT, "demand", "scenarios", r["route_file"])))
    return jobs


def run_one(c, s, seed, route):
    label = f"{c}_{s}_s{seed}"
    if os.path.exists(os.path.join(OUTDIR, f"{label}_cycles.json")):
        return label, "skipped"
    work = tempfile.mkdtemp(prefix="onl_")   # own directory: detector output cannot collide
    try:
        cmd = [sys.executable, os.path.join(HERE, "run_controller.py"),
               "--controller", c, "--route", route, "--label", label,
               "--outdir", OUTDIR, "--model-dir", MODEL_DIR]
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if p.returncode != 0:
            return label, "FAILED: " + p.stderr.strip()[-300:]
        trip = os.path.join(OUTDIR, f"{label}_tripinfo.xml")
        if os.path.exists(trip):
            os.remove(trip)                  # not needed here; saves ~3 MB per run
        return label, "ok"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def phase_run(n_seeds, jobs_n):
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = job_list(n_seeds)
    print(f"{len(jobs)} runs: {len(CONTROLLERS)} controllers x {len(SCENARIOS)} scenarios "
          f"x {n_seeds} seeds, {jobs_n} at a time")
    failed = []
    with ThreadPoolExecutor(max_workers=jobs_n) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            label, status = f.result()
            if status.startswith("FAILED"):
                failed.append((label, status))
            if i % 18 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} complete", flush=True)
    if failed:
        print("\nFAILED RUNS:")
        for l, s in failed:
            print(" ", l, s)
        sys.exit("some runs failed; resolve before analysing")


# --------------------------------------------------------------------------
# phase 2: analysis
# --------------------------------------------------------------------------
def load_estimator():
    """Bind to the PredictiveEstimator class in run_controller.py rather than a
    copy of it, without importing traci."""
    src = open(os.path.join(HERE, "run_controller.py")).read()
    a = src.index("class PredictiveEstimator")
    b = src.index("# ---------------------------------------------------------------------------\n"
                  "# simulation loop")
    ns = {"np": np, "json": json, "os": os,
          "MOVEMENTS": ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]}
    exec(src[a:b], ns)
    return ns["PredictiveEstimator"](MODEL_DIR), ns["MOVEMENTS"]


def batch_forecast(est, H):
    """Forecast for every cycle k >= window-1, using the estimator's own model,
    scaler and post-processing. Returns F where F[k] forecasts cycle k+1."""
    w = est.window
    idx = list(range(w - 1, len(H)))
    X = np.stack([H[k - w + 1:k + 1] for k in idx]).astype(np.float32)
    ys = est.model.predict((X - est.mean) / est.std, verbose=0)
    Y = np.clip(ys * est.std + est.mean, 0.0, None)
    F = np.full((len(H), H.shape[1]), np.nan)
    F[idx] = Y
    return F


def phase_analyse():
    est, MOV = load_estimator()
    w = est.window
    print(f"model: {MODEL_DIR}   window: {w} cycles\n")

    agg = {}              # (controller, scenario) -> dict of per-seed MAEs
    max_dev = 0.0         # largest gap between recomputed and logged forecast
    checked = 0
    first_verified = False

    for c in CONTROLLERS:
        for s in SCENARIOS:
            files = sorted(f for f in os.listdir(OUTDIR)
                           if f.startswith(f"{c}_{s}_s") and f.endswith("_cycles.json"))
            lstm_seed, pers_seed = [], []
            for fn in files:
                log = json.load(open(os.path.join(OUTDIR, fn)))
                H = np.array([[r["counts"][m] for m in MOV] for r in log], dtype=float)
                F = batch_forecast(est, H)

                # the batch path must equal the estimator's own method
                if not first_verified:
                    for k in range(w - 1, min(w + 4, len(H))):
                        e = est.estimate(list(H[:k + 1]))
                        d = max(abs(e[m] - F[k][i]) for i, m in enumerate(MOV))
                        assert d < 1e-4, f"batch forecast departs from estimate() by {d}"
                    first_verified = True

                # under the predictive controller, compare with the logged forecast
                if c == "predictive":
                    for k, r in enumerate(log):
                        if r.get("estimate") and not np.isnan(F[k][0]):
                            d = max(abs(r["estimate"][m] - F[k][i]) for i, m in enumerate(MOV))
                            max_dev = max(max_dev, d)
                            checked += 1

                ks = range(w - 1, len(H) - 1)
                le = np.abs(np.array([F[k] for k in ks]) - np.array([H[k + 1] for k in ks]))
                pe = np.abs(np.array([H[k] for k in ks]) - np.array([H[k + 1] for k in ks]))
                lstm_seed.append(float(le.mean()))
                pers_seed.append(float(pe.mean()))
            if files:
                agg[(c, s)] = dict(n_seeds=int(len(files)),
                                   lstm_mae=float(np.mean(lstm_seed)),
                                   lstm_sd=float(np.std(lstm_seed, ddof=1)) if len(files) > 1 else 0.0,
                                   persistence_mae=float(np.mean(pers_seed)))

    print("CONSISTENCY CHECK")
    print(f"  forecasts compared with those logged by the prediction-informed controller: {checked}")
    print(f"  largest difference: {max_dev:.6f} vehicles "
          f"(logged values are rounded to 0.001)")
    max_dev = float(max_dev)
    ok = bool(checked > 0 and max_dev < 0.002)
    print(f"  {'PASS' if ok else 'FAIL'}: the recomputed forecast "
          f"{'reproduces' if ok else 'does NOT reproduce'} the online forecast\n")

    print(f"{'scenario':<15}{'controller':<12}{'LSTM MAE':>10}{'(sd)':>8}"
          f"{'persist.':>10}{'LSTM better':>13}")
    for s in SCENARIOS:
        for c in CONTROLLERS:
            a = agg.get((c, s))
            if not a:
                continue
            gain = (a["persistence_mae"] - a["lstm_mae"]) / a["persistence_mae"] * 100
            a["improvement_over_persistence_pct"] = round(gain, 2)
            print(f"{s:<15}{c:<12}{a['lstm_mae']:>10.3f}{a['lstm_sd']:>8.3f}"
                  f"{a['persistence_mae']:>10.3f}{gain:>12.1f}%")
    print(f"\noffline reference (held-out partition, fixed-time data): {OFFLINE_MAE}")

    out = dict(model_dir=os.path.relpath(MODEL_DIR, ROOT), window=w,
               offline_mae=OFFLINE_MAE, consistency_check=dict(
                   forecasts_compared=checked, max_difference=round(max_dev, 6), passed=ok),
               results={f"{c}|{s}": v for (c, s), v in agg.items()})
    json.dump(out, open(SUMMARY, "w"), indent=2)
    print(f"wrote {os.path.relpath(SUMMARY, ROOT)}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["run", "analyse", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.phase in ("run", "all"):
        phase_run(a.seeds, a.jobs)
    if a.phase in ("analyse", "all"):
        phase_analyse()
