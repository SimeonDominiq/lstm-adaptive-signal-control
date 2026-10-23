"""
Demand generator for the modelled intersection.

Produces a four-hour demand profile across the four regimes specified in
Chapter Three, converts it to seeded stochastic vehicle departures, and
writes a SUMO route file.

Capacity basis (Chapter 3 / slide 16):
    1800 veh/h per lane of green x 2 lanes per phase x (120 s green / 140 s cycle)
    = 3086 veh/h  (reported as approximately 3,090)

Regimes:
    off-peak        ~0.30 of capacity
    building peak   ramp 0.30 -> 0.85
    saturated       ~0.85 plateau
    recovery        ramp 0.85 -> 0.35

The generator is deterministic given a seed. Different profiles are obtained
by varying the profile_id, which changes regime boundaries, the dominant
direction, turning ratios and vehicle mix. This supports the strict
train / validation / test separation described on slide 17.
"""

import argparse
import json
import os
import numpy as np

# ----------------------------------------------------------------------------
# Fixed structural constants (must match Chapter Three)
# ----------------------------------------------------------------------------
SATURATION_FLOW = 1800.0      # veh/h per lane of green
LANES_PER_PHASE = 2
GREEN_TOTAL = 120.0           # s
CYCLE = 140.0                 # s
CAPACITY = SATURATION_FLOW * LANES_PER_PHASE * (GREEN_TOTAL / CYCLE)   # ~3086 veh/h

SIM_DURATION = 14400          # 4 hours in seconds
N_CYCLES = int(SIM_DURATION // CYCLE)   # 102 complete cycles

APPROACHES = ["NB", "SB", "EB", "WB"]
MOVEMENTS = ["NB_L", "NB_TR", "SB_L", "SB_TR", "EB_L", "EB_TR", "WB_L", "WB_TR"]

# approach -> (incoming edge, {left: edge, through: edge, right: edge})
ROUTE_MAP = {
    "NB": ("S2C", {"left": "C2W", "through": "C2N", "right": "C2E"}),
    "SB": ("N2C", {"left": "C2E", "through": "C2S", "right": "C2W"}),
    "EB": ("W2C", {"left": "C2N", "through": "C2E", "right": "C2S"}),
    "WB": ("E2C", {"left": "C2S", "through": "C2W", "right": "C2N"}),
}

# ----------------------------------------------------------------------------
# Vehicle mix (share, length m, accel, decel, sigma, minGap)
# Mixed urban traffic: passenger cars dominant, plus minibuses and heavy vehicles.
# ----------------------------------------------------------------------------
VEHICLE_TYPES = {
    "car":     dict(share=0.70, length=4.5,  accel=2.6, decel=4.5, sigma=0.5, minGap=2.5, maxSpeed=13.89),
    "minibus": dict(share=0.22, length=7.0,  accel=2.0, decel=4.0, sigma=0.5, minGap=3.0, maxSpeed=13.89),
    "heavy":   dict(share=0.08, length=10.0, accel=1.3, decel=3.5, sigma=0.5, minGap=3.5, maxSpeed=12.50),
}


# ----------------------------------------------------------------------------
# Profile definition
# ----------------------------------------------------------------------------
def regime_schedule(profile_id):
    """
    Return regime boundaries in minutes and the dominant approach.
    Varying profile_id shifts the peak and changes the dominant direction, so
    that training, validation and test profiles are genuinely different.
    """
    rng = np.random.default_rng(10_000 + profile_id)

    # peak onset varies by +/- 10 minutes around 45 min
    onset = 45 + int(rng.integers(-10, 11))
    build = 60 + int(rng.integers(-8, 9))
    plateau = 60 + int(rng.integers(-8, 9))
    recover = 60 + int(rng.integers(-8, 9))

    bounds = [
        ("offpeak", 0, onset),
        ("building", onset, onset + build),
        ("saturated", onset + build, onset + build + plateau),
        ("recovery", onset + build + plateau, onset + build + plateau + recover),
        ("offpeak", onset + build + plateau + recover, 240),
    ]
    dominant = APPROACHES[int(rng.integers(0, 4))]
    return bounds, dominant


def regime_level(t_min, bounds, crest=0.0):
    """
    Demand as a fraction of capacity at time t (minutes), plus regime label.

    The saturated regime is not perfectly flat: a cosine crest is applied across
    the plateau so that the busiest hour has genuine internal structure. Its mean
    over the plateau remains at the specified 0.85 level, but the crest is what
    produces a realistic peak-hour factor rather than one drifting towards 1.0.
    """
    for label, a, b in bounds:
        if a <= t_min < b:
            if label == "offpeak":
                return 0.30, "offpeak"
            if label == "building":
                f = (t_min - a) / max(b - a, 1e-9)
                return 0.30 + f * (0.85 - 0.30), "building"
            if label == "saturated":
                span = max(b - a, 1e-9)
                # cosine crest: +crest at the midpoint, -crest at the edges
                phase = 2 * np.pi * (t_min - a) / span
                return 0.85 * (1.0 + crest * (-np.cos(phase))), "saturated"
            if label == "recovery":
                f = (t_min - a) / max(b - a, 1e-9)
                return 0.85 - f * (0.85 - 0.35), "recovery"
    return 0.30, "offpeak"


def directional_split(regime, dominant, base_shift, axis_imbalance=0.0):
    """
    Share of total demand on each approach.

    Off-peak is close to balanced; during the peak the dominant direction grows.

    axis_imbalance (0 to about 0.5) additionally loads one road axis relative to
    the other and holds it there for the whole run. Training profiles that vary
    this parameter teach the model that a junction can be persistently lopsided,
    which profiles with a single dominant approach do not.
    """
    if axis_imbalance > 0.0:
        major = 0.5 + axis_imbalance
        if dominant in ("NB", "SB"):
            shares = {"NB": major / 2, "SB": major / 2,
                      "EB": (1 - major) / 2, "WB": (1 - major) / 2}
        else:
            shares = {"EB": major / 2, "WB": major / 2,
                      "NB": (1 - major) / 2, "SB": (1 - major) / 2}
        # keep a mild within-axis peak shift on top of the imbalance
        if regime in ("building", "saturated", "recovery"):
            shares[dominant] += base_shift * 0.4
        total = sum(shares.values())
        return {a: v / total for a, v in shares.items()}

    shares = {a: 0.25 for a in APPROACHES}
    if regime in ("building", "saturated", "recovery"):
        shift = base_shift if regime == "saturated" else base_shift * 0.6
        opposite = {"NB": "SB", "SB": "NB", "EB": "WB", "WB": "EB"}[dominant]
        shares[dominant] = 0.25 + shift
        shares[opposite] = 0.25 - shift * 0.45
        for a in APPROACHES:
            if a not in (dominant, opposite):
                shares[a] = 0.25 - shift * 0.275
    total = sum(shares.values())
    return {a: s / total for a, s in shares.items()}


def turning_ratios(profile_id, vary):
    """Left / through / right split. Varied for validation and test profiles."""
    if not vary:
        return dict(left=0.15, through=0.70, right=0.15)
    rng = np.random.default_rng(20_000 + profile_id)
    left = float(rng.uniform(0.10, 0.20))
    right = float(rng.uniform(0.10, 0.20))
    return dict(left=left, through=1.0 - left - right, right=right)


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def build_profile(profile_id, seed, vary_turning=False, crest=0.12, noise_sd=0.02,
                  axis_imbalance=0.0):
    """
    Return per-cycle demand (veh/h) for each of the eight movements,
    plus metadata.

    crest    controls the shape of the saturated plateau and therefore the
             peak-hour factor (larger crest -> more peaked -> lower PHF).
    noise_sd cycle-to-cycle stochastic variation.
    """
    rng = np.random.default_rng(seed)
    bounds, dominant = regime_schedule(profile_id)
    turns = turning_ratios(profile_id, vary_turning)
    base_shift = 0.10

    rows = []
    for c in range(N_CYCLES):
        t_s = c * CYCLE
        t_min = t_s / 60.0
        level, regime = regime_level(t_min, bounds, crest=crest)

        noise = float(rng.normal(1.0, noise_sd))
        total_rate = CAPACITY * level * noise
        total_rate = max(total_rate, 0.0)

        shares = directional_split(regime, dominant, base_shift,
                                   axis_imbalance=axis_imbalance)
        row = dict(cycle=c, t_start=t_s, regime=regime, level=level,
                   total_rate=total_rate)
        for a in APPROACHES:
            ar = total_rate * shares[a]
            row[f"{a}_L"] = ar * turns["left"]
            row[f"{a}_TR"] = ar * (turns["through"] + turns["right"])
        rows.append(row)

    meta = dict(profile_id=profile_id, seed=seed, dominant=dominant,
                turning=turns, bounds=bounds, crest=crest, noise_sd=noise_sd,
                axis_imbalance=axis_imbalance,
                capacity=CAPACITY)
    return rows, meta


def sample_departures(rows, seed, turns):
    """Convert per-cycle rates into individual seeded vehicle departures."""
    rng = np.random.default_rng(seed + 555)
    type_names = list(VEHICLE_TYPES.keys())
    type_probs = [VEHICLE_TYPES[t]["share"] for t in type_names]

    vehicles = []
    for row in rows:
        for a in APPROACHES:
            in_edge, outs = ROUTE_MAP[a]
            for mv_suffix in ("L", "TR"):
                rate = row[f"{a}_{mv_suffix}"]              # veh/h
                expected = rate * CYCLE / 3600.0
                n = int(rng.poisson(expected))
                for _ in range(n):
                    depart = row["t_start"] + float(rng.uniform(0, CYCLE))
                    vtype = str(rng.choice(type_names, p=type_probs))
                    if mv_suffix == "L":
                        to_edge = outs["left"]
                        lane = "1"
                    else:
                        # split the shared movement into through and right
                        p_through = turns["through"] / (turns["through"] + turns["right"])
                        to_edge = outs["through"] if rng.random() < p_through else outs["right"]
                        lane = "0"
                    vehicles.append(dict(depart=depart, vtype=vtype,
                                         frm=in_edge, to=to_edge, lane=lane,
                                         movement=f"{a}_{mv_suffix}"))
    vehicles.sort(key=lambda v: v["depart"])
    return vehicles


def write_routes(vehicles, path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             "<!-- Generated by demand/generate_demand.py - do not edit by hand -->",
             '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
             '        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">']
    for name, p in VEHICLE_TYPES.items():
        lines.append(
            f'    <vType id="{name}" length="{p["length"]}" accel="{p["accel"]}" '
            f'decel="{p["decel"]}" sigma="{p["sigma"]}" minGap="{p["minGap"]}" '
            f'maxSpeed="{p["maxSpeed"]}" carFollowModel="Krauss"/>')
    lines.append("")
    for i, v in enumerate(vehicles):
        lines.append(
            f'    <vehicle id="{v["movement"]}_{i}" type="{v["vtype"]}" '
            f'depart="{v["depart"]:.2f}" departLane="{v["lane"]}" departSpeed="max">')
        lines.append(f'        <route edges="{v["frm"]} {v["to"]}"/>')
        lines.append("    </vehicle>")
    lines += ["</routes>", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def specified_phf(bounds, crest):
    """
    Peak-hour factor of the SPECIFIED demand profile (noise-free).

    This is the defensible figure to quote: the profile is a specification,
    not measured data, so its PHF is a deterministic property of the design
    rather than of any one stochastic realisation. Computed on 15-minute
    aggregates of the specified arrival rate.
    """
    step_min = 1.0                      # evaluate the specification each minute
    minutes = np.arange(0, 240, step_min)
    rates = np.array([regime_level(m, bounds, crest=crest)[0] for m in minutes])
    # volume per 15-minute bin (rate is a fraction of capacity; scale is irrelevant to PHF)
    per_bin = rates.reshape(-1, 15).sum(axis=1)
    best_vol, best_start = -1.0, 0
    for s in range(0, len(per_bin) - 3):
        vol = per_bin[s:s + 4].sum()
        if vol > best_vol:
            best_vol, best_start = vol, s
    peak15 = per_bin[best_start:best_start + 4].max()
    return float(best_vol / (4.0 * peak15))


def peak_hour_factor(vehicles):
    """PHF = peak-hour volume / (4 x peak 15-minute volume within that hour)."""
    times = np.array([v["depart"] for v in vehicles])
    if len(times) == 0:
        return float("nan"), 0
    # 15-minute bins
    bin_s = 900
    nbins = int(np.ceil(SIM_DURATION / bin_s))
    counts = np.zeros(nbins)
    idx = (times // bin_s).astype(int)
    for i in idx:
        if 0 <= i < nbins:
            counts[i] += 1
    # rolling hour = 4 consecutive 15-min bins
    best_vol, best_start = -1, 0
    for s in range(0, nbins - 3):
        vol = counts[s:s + 4].sum()
        if vol > best_vol:
            best_vol, best_start = vol, s
    peak15 = counts[best_start:best_start + 4].max()
    phf = best_vol / (4.0 * peak15) if peak15 > 0 else float("nan")
    return phf, int(best_vol)


def summarise(rows, vehicles, meta):
    phf, peak_vol = peak_hour_factor(vehicles)
    spec_phf = specified_phf(meta["bounds"], meta["crest"])
    rates = np.array([r["total_rate"] for r in rows])
    print(f"  profile {meta['profile_id']}  seed {meta['seed']}  dominant {meta['dominant']}")
    print(f"  turning ratios: L {meta['turning']['left']:.3f} "
          f"T {meta['turning']['through']:.3f} R {meta['turning']['right']:.3f}")
    print(f"  capacity            : {meta['capacity']:.0f} veh/h")
    print(f"  demand rate min/max : {rates.min():.0f} / {rates.max():.0f} veh/h "
          f"({rates.min()/meta['capacity']:.2f} / {rates.max()/meta['capacity']:.2f} of capacity)")
    print(f"  vehicles generated  : {len(vehicles)}")
    print(f"  peak hour volume    : {peak_vol} veh")
    sat = [r["total_rate"] / meta["capacity"] for r in rows if r["regime"] == "saturated"]
    if sat:
        print(f"  saturated plateau   : mean {np.mean(sat):.3f} of capacity (specified 0.85)")
    print(f"  PHF (specified)     : {spec_phf:.3f}   <- quote this figure")
    print(f"  PHF (this seed)     : {phf:.3f}")
    return spec_phf, phf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vary-turning", action="store_true")
    ap.add_argument("--crest", type=float, default=0.12)
    ap.add_argument("--noise-sd", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta-out", default=None)
    args = ap.parse_args()

    rows, meta = build_profile(args.profile, args.seed,
                               vary_turning=args.vary_turning,
                               crest=args.crest, noise_sd=args.noise_sd)
    vehicles = sample_departures(rows, args.seed, meta["turning"])

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "routes.rou.xml")
    write_routes(vehicles, out)
    spec_phf, phf = summarise(rows, vehicles, meta)
    meta["phf_specified"] = spec_phf
    meta["phf_realised"] = phf
    meta["n_vehicles"] = len(vehicles)
    if args.meta_out:
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2, default=str)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
