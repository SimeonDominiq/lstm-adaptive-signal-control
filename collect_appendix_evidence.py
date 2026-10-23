#!/usr/bin/env python3
"""
Collect the evidence needed for Appendices B and C of the dissertation.

Run it from the ROOT of the project folder (the folder that contains scripts/
and demand/), with the project's virtual environment ACTIVATED:

    cd /path/to/your/project
    source venv/bin/activate          # or whatever your environment is called
    python3 collect_appendix_evidence.py

What it does
  * Records the software environment: Python, macOS, processor, the installed
    package versions (pip freeze) and the SUMO version.
  * Computes a SHA-256 checksum of every code, configuration, model, data and
    result file in the project.
  * Summarises every per-cycle controller log (cycle lengths, green plans,
    decision sources, latency) as evidence for test cases TC2, TC3 and TC7.
  * Finds repeated runs of the same controller on the same route file and
    checks that their outputs are identical (test case TC6).
  * Runs the detector-counting test and any environment-verification script,
    and saves their output.
  * Copies the small JSON reports (model, dataset, conservation checks,
    divergence, Webster plan, manifests) into the evidence folder.

What it does NOT do
  * It never modifies, moves or deletes any project file. It only reads them.
    Its only output is a new folder, appendix_evidence/, and a zip of it.
  * It uses no network connection and installs nothing.
  * It records no personal information: paths are recorded relative to the
    project folder, and your home-folder path is replaced by "~".

When it finishes, upload the file  appendix_evidence.zip  that it creates.
Only the Python standard library is used.
"""
import hashlib, json, os, platform, re, shutil, subprocess, sys, time, zipfile
from collections import Counter, defaultdict
from importlib import metadata

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
OUT = os.path.join(ROOT, "appendix_evidence")
HOME = os.path.expanduser("~")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", "appendix_evidence", ".idea", ".vscode"}
KEY_PACKAGES = ["tensorflow", "tensorflow-macos", "tensorflow-metal", "keras", "numpy", "pandas", "scipy",
                "scikit-learn", "matplotlib", "sumolib", "traci", "eclipse-sumo", "libsumo", "h5py"]
CATEGORY = [  # (category, test on the lower-case file name)
    ("code",          lambda n: n.endswith((".py", ".sh")) or ".py." in n),
    ("model",         lambda n: n.endswith((".keras", ".h5")) or n.startswith("model_report")),
    ("network/config",lambda n: n.endswith((".net.xml", ".nod.xml", ".edg.xml", ".con.xml", ".add.xml", ".sumocfg", ".netccfg", ".typ.xml"))),
    ("routes",        lambda n: n.endswith(".rou.xml")),
    ("controller logs", lambda n: n.endswith(("_cycles.json", "_run.json"))),
    ("trip output",   lambda n: n.endswith(("tripinfo.xml", "summary.xml", "detector_out.xml"))),
    ("datasets",      lambda n: n.endswith((".npz", ".csv"))),
    ("results/reports", lambda n: n.endswith(".json")),
    ("environment",   lambda n: n.startswith("requirements") or n in ("environment.yml", "environment.yaml", "pyproject.toml")),
    ("logs/figures",  lambda n: n.endswith((".log", ".png", ".txt"))),
]
COPY_NAMES = re.compile(r"^(model_report.*|dataset_summary.*|conservation_checks.*|detector_divergence.*|webster_plan.*|"
                        r"movement_map.*|failures.*|.*manifest.*|honest_summary.*|statistics.*|.*_summary)\.json$", re.I)


def clean(s):
    """Replace the home-folder path so no user name is recorded."""
    return str(s).replace(HOME, "~")


def rel(p):
    return os.path.relpath(p, ROOT)


def is_venv(d):
    return os.path.isfile(os.path.join(d, "pyvenv.cfg")) or os.path.basename(d) in ("site-packages", "venv", ".venv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, timeout=300, cwd=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd or ROOT)
        return r.returncode, clean(r.stdout), clean(r.stderr)
    except FileNotFoundError:
        return None, "", "not found"
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout} s"


def main():
    if not os.path.isfile(os.path.join(ROOT, "scripts", "run_controller.py")):
        print(f"WARNING: {clean(ROOT)} does not contain scripts/run_controller.py.\n"
              "         Run this from the project's root folder (the one holding scripts/ and demand/).")
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            sys.exit(1)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "reports"))
    started = time.time()
    report = {"collected_at": time.strftime("%Y-%m-%d %H:%M:%S"), "project_folder": clean(ROOT)}

    # ---------------- 1. environment ----------------
    print("1/6 Recording the software environment ...")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    env = dict(python=sys.version.split()[0], python_executable=clean(sys.executable), virtual_environment_active=in_venv,
               macos=platform.mac_ver()[0] or None, platform=platform.platform(), machine=platform.machine(),
               processor=platform.processor() or None)
    pkgs = {}
    for p in KEY_PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    env["key_packages"] = pkgs
    rc, out, err = run([sys.executable, "-m", "pip", "freeze"], timeout=120)
    open(os.path.join(OUT, "pip_freeze.txt"), "w").write(out or err)
    env["pip_freeze_lines"] = len(out.splitlines()) if out else 0
    for tool in ("sumo", "netconvert"):
        rc, out, err = run([tool, "--version"], timeout=60)
        env[f"{tool}_version"] = (out.strip().splitlines() or [err.strip()])[0] if (out or err) else None
    env["SUMO_HOME_set"] = bool(os.environ.get("SUMO_HOME"))
    if not in_venv:
        print("   NOTE: no virtual environment appears to be active, so the package list may not be the project's.")
    report["environment"] = env

    # ---------------- 2. checksums ----------------
    print("2/6 Computing SHA-256 checksums (this can take a minute for large folders) ...")
    files = []
    for d, dirs, names in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not is_venv(os.path.join(d, x))]
        for n in names:
            if n.startswith(".") or n == os.path.basename(__file__):
                continue
            p = os.path.join(d, n)
            cat = next((c for c, test in CATEGORY if test(n.lower())), None)
            if cat is None:
                continue
            try:
                files.append(dict(path=rel(p), category=cat, bytes=os.path.getsize(p), sha256=sha256(p),
                                  modified=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))))
            except OSError as e:
                files.append(dict(path=rel(p), category=cat, error=str(e)))
    files.sort(key=lambda f: (f["category"], f["path"]))
    report["file_counts_by_category"] = dict(Counter(f["category"] for f in files))
    with open(os.path.join(OUT, "checksums.json"), "w") as f:
        json.dump(files, f, indent=1)
    with open(os.path.join(OUT, "checksums_sha256.txt"), "w") as f:  # standard 'shasum -a 256' format
        for x in files:
            if "sha256" in x:
                f.write(f"{x['sha256']}  {x['path']}\n")
    report["models"] = [x for x in files if x["category"] == "model" and x["path"].endswith((".keras", ".h5"))]
    report["training_history_files"] = [x["path"] for x in files if re.search(r"histor|curve|training_log|epoch", x["path"], re.I)]

    # ---------------- 3. small reports ----------------
    print("3/6 Copying the small JSON reports ...")
    copied = []
    for x in files:
        n = os.path.basename(x["path"])
        if x.get("bytes", 0) < 3_000_000 and COPY_NAMES.match(n) and not n.endswith(("_cycles.json", "_run.json")):
            dest = os.path.join(OUT, "reports", x["path"].replace(os.sep, "__"))
            shutil.copyfile(os.path.join(ROOT, x["path"]), dest)
            copied.append(x["path"])
    report["reports_copied"] = copied

    # ---------------- 4. controller logs: TC2, TC3, TC7 ----------------
    print("4/6 Checking every per-cycle controller log ...")
    runs = {}
    for x in files:
        if x["path"].endswith("_run.json"):
            try:
                runs[x["path"][: -len("_run.json")]] = json.load(open(os.path.join(ROOT, x["path"])))
            except Exception:
                pass
    by_dir = defaultdict(lambda: dict(logs=0, cycles=0, decisions_by_source=Counter(), realised_cycle_lengths=Counter(),
                                      green_sum_values=Counter(), greens_outside_10_60=0, max_latency_s=0.0, controllers=Counter()))
    sample = None
    for x in files:
        if not x["path"].endswith("_cycles.json"):
            continue
        try:
            log = json.load(open(os.path.join(ROOT, x["path"])))
        except Exception:
            continue
        stem = x["path"][: -len("_cycles.json")]
        ctrl = runs.get(stem, {}).get("controller", "unknown")
        g = by_dir[os.path.dirname(x["path"]) or "."]
        g["logs"] += 1; g["cycles"] += len(log); g["controllers"][ctrl] += 1
        ts = [c.get("t") for c in log if isinstance(c, dict) and c.get("t") is not None]
        for a, b in zip(ts, ts[1:]):
            g["realised_cycle_lengths"][b - a] += 1
        for c in log:
            g["decisions_by_source"][str(c.get("source"))] += 1
            gr = c.get("greens")
            if gr:
                g["green_sum_values"][sum(gr)] += 1
                g["greens_outside_10_60"] += sum(1 for v in gr if v < 10 or v > 60)
            g["max_latency_s"] = max(g["max_latency_s"], float(c.get("latency_s") or 0))
        if sample is None and ctrl == "predictive" and len(log) >= 12:
            sample = dict(log=x["path"], first_cycles=log[:12])
    report["controller_logs"] = {d: dict(v, decisions_by_source=dict(v["decisions_by_source"]),
                                         realised_cycle_lengths=dict(v["realised_cycle_lengths"]),
                                         green_sum_values=dict(v["green_sum_values"]), controllers=dict(v["controllers"]))
                                 for d, v in sorted(by_dir.items())}
    if sample:
        json.dump(sample, open(os.path.join(OUT, "sample_prediction_informed_cycles.json"), "w"), indent=1)
    latency = defaultdict(list)
    for stem, r in runs.items():
        if isinstance(r.get("decision_latency_s"), dict):
            latency[r.get("controller", "unknown")].append(r["decision_latency_s"])
    report["decision_latency_by_controller"] = {
        c: dict(runs=len(v), mean_of_means_s=round(sum(x["mean"] for x in v) / len(v), 6),
                max_s=max(x["max"] for x in v), max_pct_of_cycle=max(x.get("max_as_pct_of_cycle", 0) for x in v))
        for c, v in latency.items()}

    # ---------------- 5. repeated runs: TC6 ----------------
    print("5/6 Looking for repeated runs (TC6) ...")
    hashes = {x["path"]: x.get("sha256") for x in files}
    groups = defaultdict(list)
    for stem, r in runs.items():
        groups[(r.get("controller"), r.get("route"))].append((stem, r))
    repeats = []
    for (ctrl, route), members in groups.items():
        if len(members) < 2:
            continue
        trip = [hashes.get(os.path.normpath(r.get("tripinfo", ""))) for _, r in members]
        repeats.append(dict(controller=ctrl, route=route, runs=[s for s, _ in members],
                            tripinfo_identical=len(set(trip)) == 1 and trip[0] is not None))
    report["repeated_runs"] = repeats[:40]
    report["repeated_runs_total"] = len(repeats)
    report["repeated_runs_identical"] = sum(r["tripinfo_identical"] for r in repeats)
    report["repeated_runs_by_controller"] = dict(Counter(r["controller"] for r in repeats))

    # ---------------- 6. tests ----------------
    print("6/6 Running the detector-counting test and any environment-verification script ...")
    tests = []
    candidates = [x["path"] for x in files if x["category"] == "code" and x["path"].endswith(".py") and
                  re.search(r"(test_detector_counting|verify|check_env|environment_check|env_check)", os.path.basename(x["path"]), re.I)]
    for path in candidates:
        rc, out, err = run([sys.executable, path], timeout=300)
        name = os.path.basename(path).replace(".py", "")
        open(os.path.join(OUT, f"test_output_{name}.txt"), "w").write(f"$ python {path}\n(exit code {rc})\n\n{out}\n{err}")
        tests.append(dict(script=path, exit_code=rc, output_lines=len(out.splitlines())))
    report["tests_run"] = tests
    report["git"] = os.path.isdir(os.path.join(ROOT, ".git"))
    if report["git"]:
        rc, out, _ = run(["git", "rev-parse", "HEAD"], timeout=30)
        report["git_commit"] = out.strip() if rc == 0 else None

    # ---------------- write and zip ----------------
    report["seconds_taken"] = round(time.time() - started, 1)
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    with open(os.path.join(OUT, "README.txt"), "w") as f:
        f.write("Evidence for Appendices B and C, collected by collect_appendix_evidence.py.\n"
                "report.json            environment, models, controller-log checks, repeated runs, tests\n"
                "checksums.json         every file with its size, SHA-256 and date\n"
                "checksums_sha256.txt   the same checksums in standard 'shasum -a 256' format\n"
                "pip_freeze.txt         the installed package list\n"
                "reports/               copies of the small JSON reports\n"
                "test_output_*.txt      output of the tests that were run\n")
    zpath = os.path.join(ROOT, "appendix_evidence.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, names in os.walk(OUT):
            for n in names:
                z.write(os.path.join(d, n), os.path.relpath(os.path.join(d, n), ROOT))

    # ---------------- summary on screen ----------------
    print("\nDone in", report["seconds_taken"], "seconds.\n")
    print(f"  Python {env['python']} on macOS {env['macos']} ({env['machine']}); virtual environment active: {in_venv}")
    print(f"  Key packages: " + ", ".join(f"{k} {v}" for k, v in pkgs.items()))
    print(f"  SUMO: {env.get('sumo_version')}")
    print(f"  Files checksummed: {len(files)}  " + str(report['file_counts_by_category']))
    for m in report["models"]:
        print(f"  Model: {m['path']}  {m['bytes']:,} bytes  sha256 {m['sha256'][:16]}...")
    print(f"  Controller logs checked: {sum(v['logs'] for v in report['controller_logs'].values())} in {len(report['controller_logs'])} folders")
    print(f"  Repeated runs found: {len(repeats)}; with identical outputs: {sum(r['tripinfo_identical'] for r in repeats)}")
    print(f"  Tests run: {len(tests)}  " + ", ".join(f"{os.path.basename(t['script'])} (exit {t['exit_code']})" for t in tests))
    print(f"\nUpload this file:\n  {clean(zpath)}")


if __name__ == "__main__":
    main()
