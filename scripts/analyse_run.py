"""
Read the SUMO output files and produce the summary table and figures.

This is how results are inspected. The guide is explicit that Chapter Four must
not be a gallery of screenshots, so the deliverables here are numbered figures
generated from data, plus a table of the four performance measures used
throughout the study.

Usage, from the project root:
    python3 scripts/analyse_run.py
    python3 scripts/analyse_run.py --label webster --results-dir results

Reads:
    results/tripinfo.xml    per-vehicle outcomes
    results/summary.xml     per-second network state
Writes:
    results/<label>_summary.json
    results/<label>_fig1_demand_and_throughput.png
    results/<label>_fig2_delay_over_time.png
    results/<label>_fig3_metric_distributions.png
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CYCLE = 140.0


def read_tripinfo(path):
    """Per-vehicle records. Only vehicles that completed their trip appear here."""
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}\nRun the simulation with tripinfo output enabled.")
    rows = []
    for ev in ET.parse(path).getroot():
        if ev.tag != "tripinfo":
            continue
        rows.append(dict(
            vid=ev.get("id"),
            depart=float(ev.get("depart")),
            departDelay=float(ev.get("departDelay")),
            arrival=float(ev.get("arrival")),
            duration=float(ev.get("duration")),
            waitingTime=float(ev.get("waitingTime")),
            timeLoss=float(ev.get("timeLoss")),
            routeLength=float(ev.get("routeLength")),
        ))
    return rows


def read_summary(path):
    """Per-second network state: how many are running, waiting, arrived."""
    if not os.path.exists(path):
        return []
    rows = []
    for st in ET.parse(path).getroot():
        if st.tag != "step":
            continue
        rows.append(dict(
            time=float(st.get("time")),
            running=int(st.get("running")),
            waiting=int(st.get("waiting")),
            ended=int(st.get("ended")),
            meanWaitingTime=float(st.get("meanWaitingTime") or 0.0),
            meanTravelTime=float(st.get("meanTravelTime") or 0.0),
        ))
    return rows


def summarise(trips, steps, label):
    n = len(trips)
    if n == 0:
        raise SystemExit("no completed trips in tripinfo output")

    wait = np.array([t["waitingTime"] for t in trips])
    loss = np.array([t["timeLoss"] for t in trips])
    dur = np.array([t["duration"] for t in trips])
    ddelay = np.array([t["departDelay"] for t in trips])

    duration_s = max((s["time"] for s in steps), default=max(t["arrival"] for t in trips))
    throughput = n / (duration_s / 3600.0)

    max_queue = max((s["waiting"] for s in steps), default=0)
    mean_queue = float(np.mean([s["waiting"] for s in steps])) if steps else float("nan")

    out = {
        "label": label,
        "completed_trips": n,
        "simulated_seconds": duration_s,
        "throughput_veh_per_h": round(throughput, 1),
        "avg_waiting_time_s": round(float(wait.mean()), 2),
        "avg_time_loss_s": round(float(loss.mean()), 2),
        "avg_trip_duration_s": round(float(dur.mean()), 2),
        "avg_depart_delay_s": round(float(ddelay.mean()), 2),
        "mean_vehicles_waiting": round(mean_queue, 2),
        "max_vehicles_waiting": max_queue,
        "waiting_time_p50": round(float(np.percentile(wait, 50)), 2),
        "waiting_time_p95": round(float(np.percentile(wait, 95)), 2),
    }
    return out


def fig_demand_throughput(trips, steps, path, label):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bin_s = CYCLE
    if trips:
        dep = np.array([t["depart"] for t in trips])
        arr = np.array([t["arrival"] for t in trips])
        tmax = max(dep.max(), arr.max())
        bins = np.arange(0, tmax + bin_s, bin_s)
        d, _ = np.histogram(dep, bins=bins)
        a, _ = np.histogram(arr, bins=bins)
        centres = (bins[:-1] + bin_s / 2) / 60.0
        ax.plot(centres, d * 3600.0 / bin_s, label="departures (entered)", linewidth=1.4)
        ax.plot(centres, a * 3600.0 / bin_s, label="arrivals (completed)", linewidth=1.4)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("rate (veh/h)")
    ax.set_title(f"Entry and completion rate per {int(CYCLE)} s cycle — {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_delay_over_time(steps, path, label):
    if not steps:
        return False
    t = np.array([s["time"] for s in steps]) / 60.0
    waiting = np.array([s["waiting"] for s in steps])
    running = np.array([s["running"] for s in steps])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, running, label="vehicles in network", linewidth=1.0)
    ax.plot(t, waiting, label="vehicles waiting", linewidth=1.0)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("vehicles")
    ax.set_title(f"Network occupancy and queueing over time — {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_distributions(trips, path, label):
    wait = np.array([t["waitingTime"] for t in trips])
    loss = np.array([t["timeLoss"] for t in trips])
    dur = np.array([t["duration"] for t in trips])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, data, name in zip(axes, [wait, loss, dur],
                              ["waiting time (s)", "time loss (s)", "trip duration (s)"]):
        ax.hist(data, bins=40)
        ax.axvline(data.mean(), linestyle="--", linewidth=1.2,
                   label=f"mean {data.mean():.1f}")
        ax.set_xlabel(name)
        ax.set_ylabel("vehicles")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Distribution of per-vehicle measures — {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rdir = args.results_dir or os.path.join(root, "results")

    trips = read_tripinfo(os.path.join(rdir, "tripinfo.xml"))
    steps = read_summary(os.path.join(rdir, "summary.xml"))

    s = summarise(trips, steps, args.label)

    print(f"\nResults — {args.label}")
    print("-" * 52)
    for k, v in s.items():
        if k == "label":
            continue
        print(f"  {k:<26} {v}")
    print("-" * 52)

    f1 = os.path.join(rdir, f"{args.label}_fig1_demand_and_throughput.png")
    f2 = os.path.join(rdir, f"{args.label}_fig2_delay_over_time.png")
    f3 = os.path.join(rdir, f"{args.label}_fig3_metric_distributions.png")
    fig_demand_throughput(trips, steps, f1, args.label)
    have2 = fig_delay_over_time(steps, f2, args.label)
    fig_distributions(trips, f3, args.label)

    jpath = os.path.join(rdir, f"{args.label}_summary.json")
    with open(jpath, "w") as f:
        json.dump(s, f, indent=2)

    print(f"\n  wrote {jpath}")
    print(f"  wrote {f1}")
    if have2:
        print(f"  wrote {f2}")
    else:
        print("  (no summary.xml - skipped the occupancy figure)")
    print(f"  wrote {f3}")


if __name__ == "__main__":
    main()
