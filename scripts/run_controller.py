"""
Stage 8 - the three controllers.

    fixed        the Webster-calibrated fixed-time plan. Never changes.
    reactive     adapts each cycle using the demand it has just measured.
    predictive   adapts each cycle using the LSTM's forecast of the coming cycle.

The reactive and predictive controllers share exactly one control law, in
`allocate_green` below. They differ in a single line: which demand vector is
handed to it. That is what makes the B-versus-C comparison meaningful - any
difference in performance between them is attributable to measured versus
predicted demand, and to nothing else.

Control law (Chapter Three, slide 24):

    g_k = (D_k / D_total) x G_available

where D_k is the demand of the critical movement in phase k, and G_available is
120 s. Green times are clamped to 10-60 s and any residual is redistributed
proportionally, so the cycle always sums to 140 s.

Both adaptive controllers use the fixed Webster plan for the first `window`
cycles, until enough history exists for a forecast. They warm up identically,
so neither gains an advantage from the start of the run.

Decision latency is measured every cycle, as committed in Chapter Three
(test case TC7): the time to read the detectors, produce the demand estimate
and compute the timings must be far shorter than the 140 s cycle.

Usage:
    python3 scripts/run_controller.py --controller fixed      --route <file>
    python3 scripts/run_controller.py --controller reactive   --route <file>
    python3 scripts/run_controller.py --controller predictive --route <file>
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402

MOVEMENTS = ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]
LANE_FOR = {
    "NB_L": "S2C_1", "NB_TR": "S2C_0",
    "SB_L": "N2C_1", "SB_TR": "N2C_0",
    "EB_L": "W2C_1", "EB_TR": "W2C_0",
    "WB_L": "E2C_1", "WB_TR": "E2C_0",
}
APPROACH_LANES = sorted(set(LANE_FOR.values()))


# ---------------------------------------------------------------------------
# the control law - shared by reactive and predictive
# ---------------------------------------------------------------------------
def allocate_green(demand_by_movement, phase_movements, green_available,
                   g_min, g_max):
    """
    Split the available green between the phases in proportion to the demand of
    each phase's critical movement, then clamp and redistribute.

    This function is deliberately the only place green times are decided. Both
    adaptive controllers call it with identical arguments except for the demand
    vector, so the comparison between them isolates the demand estimate.
    """
    d = np.array([max(demand_by_movement[m] for m in movs)
                  for movs in phase_movements], dtype=float)

    if d.sum() <= 0:
        g = np.full(len(d), green_available / len(d), dtype=float)
    else:
        g = green_available * d / d.sum()

    for _ in range(50):
        clamped = np.clip(g, g_min, g_max)
        residual = green_available - clamped.sum()
        if abs(residual) < 1e-9:
            g = clamped
            break
        free = clamped < g_max if residual > 0 else clamped > g_min
        if not free.any():
            g = clamped
            break
        w = d[free] if d[free].sum() > 0 else np.ones(int(free.sum()))
        clamped[free] = clamped[free] + residual * w / w.sum()
        g = clamped

    g = np.clip(g, g_min, g_max)
    gi = np.floor(g).astype(int)
    target = int(round(green_available))
    while gi.sum() < target:
        gi[int(np.argmax(g - gi))] += 1
    while gi.sum() > target:
        gi[int(np.argmin(g - gi))] -= 1
    return gi.tolist()


# ---------------------------------------------------------------------------
# demand estimators - the single point of difference
# ---------------------------------------------------------------------------
class ReactiveEstimator:
    """Uses the demand measured during the cycle that has just finished."""

    name = "reactive"
    needs_history = 1

    def estimate(self, history):
        return {m: float(history[-1][i]) for i, m in enumerate(MOVEMENTS)}


class PredictiveEstimator:
    """
    Uses the LSTM's forecast of the cycle about to begin.

    Optional variance correction (`gain` > 1) is a diagnostic, not part of the
    method specified in Chapter Three. A model fitted on mean-squared error
    predicts the conditional mean, so its outputs vary less than reality. Where
    the control law responds to the RATIOS between phases, damped variation
    means the timings barely move. The correction rescales each movement's
    deviation from its training mean:

        corrected = mean + gain x (forecast - mean)

    It changes the spread of the forecast without changing its central
    tendency. If a controller with gain > 1 performs better, the limitation is
    under-dispersion rather than inaccuracy. Any use of this in the final method
    must be declared in Chapter Four.
    """

    name = "predictive"

    def __init__(self, model_dir, gain=1.0, bias_correct=False):
        import tensorflow as tf
        report = json.load(open(os.path.join(model_dir, "model_report.json")))
        self.window = int(report["window_selected"])
        self.mean = np.array(report["scaler"]["mean"], dtype=np.float32)
        self.std = np.array(report["scaler"]["std"], dtype=np.float32)
        self.model = tf.keras.models.load_model(
            os.path.join(model_dir, "lstm_model.keras"))
        self.needs_history = self.window
        # the range the model was fitted on, used as a sanity guard below
        self.train_mean = float(self.mean.mean())
        self.warned = False
        self.gain = float(gain)
        # Bias measured on the validation partition, never on test.
        self.bias = np.array(report.get("val_bias") or [0.0] * len(MOVEMENTS),
                             dtype=np.float32)
        self.bias_correct = bool(bias_correct)
        bits = []
        if self.gain != 1.0:
            bits.append(f"gain{self.gain:g}")
        if self.bias_correct:
            bits.append("bias")
        if bits:
            self.name = "predictive_" + "_".join(bits)

    def estimate(self, history):
        # Guard: if the live counts are far outside the range the model was
        # trained on, its output is meaningless. That situation indicates a
        # measurement fault rather than unusual traffic, so say so loudly
        # instead of quietly producing nonsense.
        recent = float(np.mean(history[-self.window:]))
        if not self.warned and recent > 5.0 * self.train_mean:
            print(f"  WARNING: live detector counts average {recent:.1f} per movement "
                  f"per cycle, against {self.train_mean:.1f} in training. "
                  f"The forecast cannot be trusted at this input scale - "
                  f"check how the detectors are being read.")
            self.warned = True

        x = np.array(history[-self.window:], dtype=np.float32)[None, :, :]
        xs = (x - self.mean) / self.std
        ys = self.model.predict(xs, verbose=0)[0]
        y = ys * self.std + self.mean
        if self.bias_correct:
            y = y - self.bias
        if self.gain != 1.0:
            y = self.mean + self.gain * (y - self.mean)
        y = np.clip(y, 0.0, None)
        return {m: float(y[i]) for i, m in enumerate(MOVEMENTS)}


# ---------------------------------------------------------------------------
# simulation loop
# ---------------------------------------------------------------------------
def run(controller, route, net, mapping, outdir, label, gui=False, end=14400,
        gain=1.0, bias_correct=False, model_dir=None):
    cycle = int(mapping["cycle"])
    green_available = int(mapping["green_total"])
    g_min, g_max = int(mapping["g_min"]), int(mapping["g_max"])
    phase_movements = mapping["phase_movements"]
    tls = mapping["tls_id"]

    # green phases sit at indices 0, 3, 6, 9 in the twelve-phase programme
    green_phase_idx = [0, 3, 6, 9]

    webster_plan = None
    wp = os.path.join(ROOT, "network", "webster_plan.json")
    if os.path.exists(wp):
        meta = json.load(open(wp))
        webster_plan = [int(v) for v in meta["green_times_s"].values()]

    estimator = None
    if controller == "reactive":
        estimator = ReactiveEstimator()
    elif controller == "predictive":
        estimator = PredictiveEstimator(model_dir or os.path.join(ROOT, "models"),
                                        gain=gain, bias_correct=bias_correct)
    elif controller != "fixed":
        raise SystemExit(f"unknown controller: {controller}")

    os.makedirs(outdir, exist_ok=True)
    tripinfo = os.path.join(outdir, f"{label}_tripinfo.xml")

    detectors = os.path.join(ROOT, "detectors", "detectors.add.xml")
    if not os.path.exists(detectors):
        raise SystemExit(f"missing {detectors}")

    binary = "sumo-gui" if gui else "sumo"
    cmd = [binary, "-n", os.path.abspath(net), "-r", os.path.abspath(route),
           "-a", os.path.abspath(detectors),
           "--time-to-teleport", "-1", "--no-step-log", "true",
           "--no-warnings", "true", "--end", str(end),
           "--tripinfo-output", os.path.abspath(tripinfo)]

    traci.start(cmd)

    history = []            # per-cycle detector counts, one row of 8 per cycle
    cycle_log = []
    # Per-movement running count for the cycle in progress, and the set of
    # identities occupying each loop in the previous step. Counting only
    # newly appeared identities avoids double-counting a vehicle that is
    # still on the loop when a cycle boundary passes.
    counts_acc = np.zeros(len(MOVEMENTS), dtype=float)
    # Cycle-length assertion state. The tolerance allows the one-step
    # resolution at which a cycle start is detected, and nothing more.
    last_cycle_start = None
    cycle_tolerance = 1
    prev_on_loop = [set() for _ in MOVEMENTS]
    latencies = []
    applied = None
    t = 0
    prev_phase = None

    while t < end:
        # Count a vehicle when it first appears on the loop.
        #
        # Accumulating identities and clearing them at each cycle boundary
        # counts a vehicle twice when it is still on the loop as the boundary
        # passes: once in the cycle that ends and once in the cycle that
        # begins. Comparing against the previous step's occupants instead
        # counts each vehicle once per continuous occupancy, wherever the
        # boundary falls.
        for i, m in enumerate(MOVEMENTS):
            now = set(traci.inductionloop.getLastStepVehicleIDs(f"det_{m}"))
            counts_acc[i] += len(now - prev_on_loop[i])
            prev_on_loop[i] = now

        phase = traci.trafficlight.getPhase(tls)

        # a new cycle begins when we return to the first green phase
        if prev_phase is not None and phase == 0 and prev_phase != 0:
            # Assert the realised cycle length against the specification. A
            # controller that silently lengthens the cycle would otherwise
            # produce entirely plausible results while comparing controllers
            # on unequal terms, which is how an earlier fault went unnoticed.
            if last_cycle_start is not None:
                realised = t - last_cycle_start
                if abs(realised - cycle) > cycle_tolerance:
                    traci.close()
                    raise SystemExit(
                        f"cycle length {realised}s differs from the specified "
                        f"{cycle}s by more than {cycle_tolerance}s at t={t}. "
                        f"The controller is altering the cycle; the run has been "
                        f"stopped rather than produce results on unequal terms.")
            last_cycle_start = t
            counts = counts_acc.copy()
            history.append(counts.copy())
            queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in APPROACH_LANES)

            greens, source, latency = webster_plan, "webster", 0.0
            demand = None
            if estimator is not None and len(history) >= estimator.needs_history:
                t0 = time.perf_counter()
                demand = estimator.estimate(history)
                greens = allocate_green(demand, phase_movements,
                                        green_available, g_min, g_max)
                latency = time.perf_counter() - t0
                source = estimator.name
                latencies.append(latency)

            applied = greens
            cycle_log.append(dict(
                cycle=len(history), t=t,
                counts={m: int(counts[i]) for i, m in enumerate(MOVEMENTS)},
                greens=list(greens) if greens else None,
                source=source, latency_s=round(latency, 6),
                queue_vehicles=int(queue),
                estimate={k: round(v, 3) for k, v in demand.items()} if demand else None))
            counts_acc[:] = 0.0

        # Apply the decided duration as each green phase starts.
        #
        # setPhaseDuration sets the REMAINING duration of the phase now
        # running. The transition is detected one step after the phase began,
        # so one second has already elapsed by the time this call is made.
        # Requesting the full duration would therefore extend the phase by that
        # second and lengthen the cycle by four seconds across the four green
        # phases. Requesting one second less makes the
        # realised duration equal the intended one.
        if applied is not None and phase != prev_phase and phase in green_phase_idx:
            k = green_phase_idx.index(phase)
            traci.trafficlight.setPhaseDuration(tls, max(1, applied[k] - 1))

        prev_phase = phase
        traci.simulationStep()
        t += 1

    traci.close()

    lat = np.array(latencies) if latencies else np.array([0.0])
    summary = dict(
        controller=controller, label=label,
        route=os.path.relpath(route, ROOT),
        cycles_completed=len(history),
        decision_latency_s=dict(
            mean=round(float(lat.mean()), 6),
            max=round(float(lat.max()), 6),
            p95=round(float(np.percentile(lat, 95)), 6),
            cycle_length_s=cycle,
            max_as_pct_of_cycle=round(float(lat.max()) / cycle * 100.0, 4)),
        mean_queue_vehicles=round(float(np.mean([c["queue_vehicles"] for c in cycle_log])), 2)
        if cycle_log else None,
        tripinfo=os.path.relpath(tripinfo, ROOT),
    )

    with open(os.path.join(outdir, f"{label}_cycles.json"), "w") as f:
        json.dump(cycle_log, f, indent=2)
    with open(os.path.join(outdir, f"{label}_run.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True,
                    choices=["fixed", "reactive", "predictive"])
    ap.add_argument("--route", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--end", type=int, default=14400)
    ap.add_argument("--model-dir", default=None,
                    help="directory holding lstm_model.keras and model_report.json")
    ap.add_argument("--bias-correct", action="store_true",
                    help="subtract the per-movement bias measured on validation "
                         "(diagnostic; must be declared if adopted)")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="variance correction for the predictive controller "
                         "(diagnostic; 1.0 = the method as specified)")
    args = ap.parse_args()

    net = os.path.join(ROOT, "network", "intersection.net.xml")
    mpath = os.path.join(ROOT, "network", "movement_map.json")
    for p in (net, mpath):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - build the network and run make_tls.py first")
    mapping = json.load(open(mpath))

    label = args.label or f"{args.controller}_{os.path.basename(args.route).split('.')[0]}"
    s = run(args.controller, args.route, net, mapping, args.outdir, label,
            gui=args.gui, end=args.end, gain=args.gain,
            bias_correct=args.bias_correct, model_dir=args.model_dir)

    print(f"\n{args.controller} - {label}")
    print(f"  cycles completed     {s['cycles_completed']}")
    dl = s["decision_latency_s"]
    print(f"  decision latency     mean {dl['mean']*1000:.2f} ms   "
          f"max {dl['max']*1000:.2f} ms   "
          f"({dl['max_as_pct_of_cycle']:.4f}% of the {dl['cycle_length_s']} s cycle)")
    print(f"  mean queue           {s['mean_queue_vehicles']} vehicles")
    print(f"  wrote {s['tripinfo']}")


if __name__ == "__main__":
    main()
