"""
Stage 9b - run the controlled experiment and apply the statistical protocol.

Every controller is run on exactly the same route files, so each scenario gives
a set of paired observations differing only in the controller. The comparisons
reported are:

    A vs B   does adapting at all help, against the calibrated fixed plan
    B vs C   does the forecast add anything beyond reacting to measurements
    A vs C   the overall effect of prediction-informed control

Protocol (Chapter Three, slide 28):

    replication   at least twenty paired seeds per controller per scenario
    descriptives  mean, standard deviation, 95% confidence interval
    normality     Shapiro-Wilk on the paired differences
    inference     paired t-test where the differences are normal,
                  Wilcoxon signed-rank where they are not
    effect size   Cohen's d for the t-test, matched-pairs rank-biserial
                  correlation for Wilcoxon
    multiplicity  Holm-Bonferroni across the four metrics within a comparison

Metrics, all per vehicle except throughput:

    waiting_time    time spent stopped
    time_loss       delay relative to free-flow travel
    duration        total trip time
    throughput      vehicles completing their trip per hour

Runs are independent, so they are executed in parallel, each in its own working
directory so the detector output files cannot collide.

    python3 scripts/run_experiment.py --jobs 8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CONTROLLERS = ["fixed", "reactive", "predictive"]
SHORT = {"fixed": "A", "reactive": "B", "predictive": "C"}
METRICS = ["waiting_time", "time_loss", "duration", "throughput"]
LOWER_IS_BETTER = {"waiting_time": True, "time_loss": True,
                   "duration": True, "throughput": False}


# ---------------------------------------------------------------------------
def parse_tripinfo(path):
    wait, loss, dur = [], [], []
    last_arrival = 0.0
    for ev in ET.parse(path).getroot():
        if ev.tag != "tripinfo":
            continue
        wait.append(float(ev.get("waitingTime")))
        loss.append(float(ev.get("timeLoss")))
        dur.append(float(ev.get("duration")))
        last_arrival = max(last_arrival, float(ev.get("arrival")))
    n = len(wait)
    if n == 0:
        return None
    hours = last_arrival / 3600.0 if last_arrival > 0 else 1.0
    return dict(waiting_time=float(np.mean(wait)),
                time_loss=float(np.mean(loss)),
                duration=float(np.mean(dur)),
                throughput=n / hours,
                completed=n)


def one_run(task):
    controller, route, label, scenario, seed, model_dir = task
    workdir = tempfile.mkdtemp(prefix="exp_")
    try:
        cmd = [sys.executable, os.path.join(HERE, "run_controller.py"),
               "--controller", controller,
               "--route", os.path.join(ROOT, "demand", "scenarios", route),
               "--label", label, "--outdir", workdir]
        if model_dir:
            cmd += ["--model-dir", model_dir]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            return dict(ok=False, controller=controller, scenario=scenario,
                        seed=seed, error=proc.stderr[-1500:])
        m = parse_tripinfo(os.path.join(workdir, f"{label}_tripinfo.xml"))
        if m is None:
            return dict(ok=False, controller=controller, scenario=scenario,
                        seed=seed, error="no completed trips")
        run_json = os.path.join(workdir, f"{label}_run.json")
        queue, latency = None, None
        if os.path.exists(run_json):
            rj = json.load(open(run_json))
            queue = rj.get("mean_queue_vehicles")
            latency = rj.get("decision_latency_s", {}).get("max")
        m.update(ok=True, controller=controller, scenario=scenario, seed=seed,
                 mean_queue=queue, max_latency_s=latency)
        return m
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
def describe(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    else:
        half = 0.0
    return dict(n=n, mean=round(mean, 4), sd=round(sd, 4),
                ci95=[round(mean - half, 4), round(mean + half, 4)])


def paired_test(a, b, alpha=0.05):
    """
    Compare paired samples. Shapiro-Wilk decides which test applies, and the
    matching effect size is reported alongside.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    out = dict(n=n, mean_difference=round(float(d.mean()), 4))

    if np.allclose(d, 0):
        out.update(normality_p=None, test="none", statistic=None, p=1.0,
                   effect_size=0.0, effect_name="none")
        return out

    sh_stat, sh_p = stats.shapiro(d)
    out["normality_p"] = round(float(sh_p), 6)

    if sh_p > alpha:
        t, p = stats.ttest_rel(a, b)
        sd = d.std(ddof=1)
        out.update(test="paired t-test", statistic=round(float(t), 4),
                   p=float(p), effect_name="Cohen's d",
                   effect_size=round(float(d.mean() / sd), 4) if sd > 0 else 0.0)
    else:
        try:
            w, p = stats.wilcoxon(a, b)
        except ValueError:
            w, p = float("nan"), 1.0
        # matched-pairs rank-biserial correlation
        nz = d[d != 0]
        if len(nz) > 0:
            ranks = stats.rankdata(np.abs(nz))
            rpos = ranks[nz > 0].sum()
            rneg = ranks[nz < 0].sum()
            total = rpos + rneg
            rb = float((rpos - rneg) / total) if total > 0 else 0.0
        else:
            rb = 0.0
        out.update(test="Wilcoxon signed-rank",
                   statistic=round(float(w), 4) if w == w else None,
                   p=float(p), effect_name="rank-biserial", effect_size=round(rb, 4))
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def compare(results, scenario, x, y, alpha=0.05):
    """Paired comparison of controllers x and y within a scenario."""
    rows_x = {r["seed"]: r for r in results
              if r["controller"] == x and r["scenario"] == scenario}
    rows_y = {r["seed"]: r for r in results
              if r["controller"] == y and r["scenario"] == scenario}
    seeds = sorted(set(rows_x) & set(rows_y))
    if len(seeds) < 3:
        return None

    out = dict(scenario=scenario, comparison=f"{SHORT[x]} vs {SHORT[y]}",
               controllers=[x, y], n_pairs=len(seeds), metrics={})
    raw_p = []
    for metric in METRICS:
        a = [rows_x[s][metric] for s in seeds]
        b = [rows_y[s][metric] for s in seeds]
        t = paired_test(a, b, alpha=alpha)
        t["describe_" + SHORT[x]] = describe(a)
        t["describe_" + SHORT[y]] = describe(b)
        better = None
        if t["p"] < alpha:
            if LOWER_IS_BETTER[metric]:
                better = SHORT[x] if t["mean_difference"] < 0 else SHORT[y]
            else:
                better = SHORT[x] if t["mean_difference"] > 0 else SHORT[y]
        t["better"] = better
        out["metrics"][metric] = t
        raw_p.append(t["p"])

    adj = holm(raw_p)
    for metric, p_adj in zip(METRICS, adj):
        m = out["metrics"][metric]
        m["p_holm"] = float(p_adj)
        m["significant_after_holm"] = bool(p_adj < alpha)
        if not m["significant_after_holm"]:
            m["better"] = None
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--controllers", nargs="*", default=CONTROLLERS)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "experiment"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--model-dir", default=None,
                    help="model directory for the predictive controller")
    args = ap.parse_args()

    man_path = os.path.join(ROOT, "demand", "scenarios", "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit("run: python3 demand/build_scenarios.py --seeds 20")
    manifest = json.load(open(man_path))
    os.makedirs(args.outdir, exist_ok=True)

    scenarios = args.scenarios or list(manifest["scenarios"])
    tasks = []
    for scn in scenarios:
        for run in manifest["scenarios"][scn]["runs"]:
            for c in args.controllers:
                label = f"{SHORT[c]}_{scn}_{run['seed']}"
                tasks.append((c, run["route_file"], label, scn, run["seed"],
                              args.model_dir))

    print(f"{len(tasks)} runs across {len(scenarios)} scenarios, "
          f"{len(args.controllers)} controllers, {args.jobs} parallel jobs")
    print("the predictive controller is slower - expect this to take a while\n")

    results, failures = [], []
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(one_run, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r.get("ok"):
                results.append(r)
            else:
                failures.append(r)
                print(f"  FAILED {r['controller']} {r['scenario']} "
                      f"seed {r['seed']}: {str(r.get('error'))[:200]}")
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} complete")

    with open(os.path.join(args.outdir, "raw_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if failures:
        print(f"\n{len(failures)} runs failed - see failures.json")
        with open(os.path.join(args.outdir, "failures.json"), "w") as f:
            json.dump(failures, f, indent=2)

    # ---- statistics --------------------------------------------------------
    pairs = [("fixed", "reactive"), ("reactive", "predictive"), ("fixed", "predictive")]
    all_stats = []
    for scn in scenarios:
        for x, y in pairs:
            if x in args.controllers and y in args.controllers:
                c = compare(results, scn, x, y, alpha=args.alpha)
                if c:
                    all_stats.append(c)

    with open(os.path.join(args.outdir, "statistics.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 96)
    print("PAIRED COMPARISONS  (Holm-Bonferroni adjusted across the four metrics)")
    print("=" * 96)
    for scn in scenarios:
        print(f"\n{scn.upper()}")
        for c in [s for s in all_stats if s["scenario"] == scn]:
            print(f"  {c['comparison']}   n = {c['n_pairs']} paired seeds")
            print(f"    {'metric':<14}{'mean A':>10}{'mean B':>10}{'diff':>10}"
                  f"{'test':>22}{'p_holm':>10}{'effect':>10}  better")
            for metric in METRICS:
                m = c["metrics"][metric]
                dx = m[f"describe_{c['comparison'][0]}"]["mean"]
                dy = m[f"describe_{c['comparison'][-1]}"]["mean"]
                star = "*" if m["significant_after_holm"] else " "
                print(f"    {metric:<14}{dx:>10.2f}{dy:>10.2f}"
                      f"{m['mean_difference']:>10.2f}{m['test']:>22}"
                      f"{m['p_holm']:>10.4f}{m['effect_size']:>10.3f}  "
                      f"{m['better'] or '-'}{star}")

    # ---- summary of the question that matters ------------------------------
    print("\n" + "=" * 96)
    print("DOES PREDICTION ADD VALUE BEYOND REACTING?  (B vs C)")
    print("=" * 96)
    wins = {"B": 0, "C": 0, "no difference": 0}
    for c in [s for s in all_stats if s["comparison"] == "B vs C"]:
        verdicts = [c["metrics"][m]["better"] for m in METRICS]
        cw = sum(1 for v in verdicts if v == "C")
        bw = sum(1 for v in verdicts if v == "B")
        if cw > bw:
            verdict = "C better"
            wins["C"] += 1
        elif bw > cw:
            verdict = "B better"
            wins["B"] += 1
        else:
            verdict = "no clear difference"
            wins["no difference"] += 1
        print(f"  {c['scenario']:<16} {verdict:<22} "
              f"(C better on {cw} of {len(METRICS)} metrics, B on {bw})")
    print("-" * 96)
    print(f"  scenarios favouring prediction (C): {wins['C']}")
    print(f"  scenarios favouring reaction   (B): {wins['B']}")
    print(f"  scenarios with no clear difference: {wins['no difference']}")
    print("=" * 96)
    print(f"\nwrote {os.path.relpath(args.outdir, ROOT)}/raw_results.json")
    print(f"wrote {os.path.relpath(args.outdir, ROOT)}/statistics.json")


if __name__ == "__main__":
    main()
