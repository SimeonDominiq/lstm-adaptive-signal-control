"""
Fix the phase extension introduced by setPhaseDuration, and assert the cycle
length at runtime so the fault cannot recur silently.

THE FAULT
setPhaseDuration sets the REMAINING duration of the phase currently running.
The controller detects that a phase has begun one step after it began, so one
second has already elapsed when the call is made. Requesting the full duration
therefore produces a phase one second longer than intended. Across four green
phases this lengthens the cycle by four seconds.

Measured directly: with no controller attached the signal runs at 140 s, every
phase exactly as written. Under a controller that calls setPhaseDuration the
four green phases run 54, 11, 48 and 11 seconds against written values of 53,
10, 47 and 10, giving a 144 s cycle. The yellow and all-red phases, which the
controller never sets, are unaffected.

CONSEQUENCE
The adaptive controllers ran a 144 s cycle while the fixed-time baseline ran
140 s, so the three controllers were not compared on equal terms. The adaptive
controllers received about 2.9 per cent more green per cycle than the baseline.
The experiment matrix must be rerun after this fix.

THE FIX
Request one second less than the intended duration, so that the realised
duration equals the intended one. A runtime assertion is added at each cycle
boundary comparing the observed cycle length against the specified one, so that
any future divergence stops the run instead of producing plausible numbers.

    python3 scripts/patch_phase_duration.py            # apply
    python3 scripts/patch_phase_duration.py --check    # report only
"""

import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "run_controller.py")

MARKER = "realised duration equal the intended one"

OLD_APPLY = re.compile(
    r"        # apply the decided duration as each green phase starts\n"
    r"        if applied is not None and phase != prev_phase and phase in green_phase_idx:\n"
    r"            k = green_phase_idx\.index\(phase\)\n"
    r"            traci\.trafficlight\.setPhaseDuration\(tls, applied\[k\]\)\n",
    re.M)

NEW_APPLY = (
"        # Apply the decided duration as each green phase starts.\n"
"        #\n"
"        # setPhaseDuration sets the REMAINING duration of the phase now\n"
"        # running. The transition is detected one step after the phase began,\n"
"        # so one second has already elapsed by the time this call is made.\n"
"        # Requesting the full duration would therefore extend the phase by that\n"
"        # second and lengthen the cycle by four seconds across the four green\n"
"        # phases. Requesting one second less makes the\n"
"        # realised duration equal the intended one.\n"
"        if applied is not None and phase != prev_phase and phase in green_phase_idx:\n"
"            k = green_phase_idx.index(phase)\n"
"            traci.trafficlight.setPhaseDuration(tls, max(1, applied[k] - 1))\n")

OLD_CYCLE = re.compile(
    r"        # a new cycle begins when we return to the first green phase\n"
    r"        if prev_phase is not None and phase == 0 and prev_phase != 0:\n",
    re.M)

NEW_CYCLE = (
"        # a new cycle begins when we return to the first green phase\n"
"        if prev_phase is not None and phase == 0 and prev_phase != 0:\n"
"            # Assert the realised cycle length against the specification. A\n"
"            # controller that silently lengthens the cycle would otherwise\n"
"            # produce entirely plausible results while comparing controllers\n"
"            # on unequal terms, which is how an earlier fault went unnoticed.\n"
"            if last_cycle_start is not None:\n"
"                realised = t - last_cycle_start\n"
"                if abs(realised - cycle) > cycle_tolerance:\n"
"                    traci.close()\n"
"                    raise SystemExit(\n"
"                        f\"cycle length {realised}s differs from the specified \"\n"
"                        f\"{cycle}s by more than {cycle_tolerance}s at t={t}. \"\n"
"                        f\"The controller is altering the cycle; the run has been \"\n"
"                        f\"stopped rather than produce results on unequal terms.\")\n"
"            last_cycle_start = t\n")

OLD_INIT = re.compile(
    r"    counts_acc = np\.zeros\(len\(MOVEMENTS\), dtype=float\)\n",
    re.M)

NEW_INIT = (
"    counts_acc = np.zeros(len(MOVEMENTS), dtype=float)\n"
"    # Cycle-length assertion state. The tolerance allows the one-step\n"
"    # resolution at which a cycle start is detected, and nothing more.\n"
"    last_cycle_start = None\n"
"    cycle_tolerance = 1\n")


def apply_patch(text):
    if MARKER in text:
        return text, ["already patched - no change made"]

    changes = []

    new, n = OLD_APPLY.subn(NEW_APPLY, text)
    if n != 1:
        raise SystemExit(
            f"could not locate the setPhaseDuration call (found {n}). "
            "Has run_controller.py been edited since the detector patch?")
    changes.append("setPhaseDuration now requests one second less, so the realised "
                   "duration matches the intended one")
    text = new

    text, n = OLD_INIT.subn(NEW_INIT, text)
    if n != 1:
        raise SystemExit(f"could not locate the accumulator declaration (found {n})")
    changes.append("added cycle-length assertion state")

    text, n = OLD_CYCLE.subn(NEW_CYCLE, text)
    if n != 1:
        raise SystemExit(f"could not locate the cycle-boundary test (found {n})")
    changes.append("added a runtime assertion that stops the run if the realised "
                   "cycle departs from the specified one")

    return text, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()

    if not os.path.exists(args.target):
        raise SystemExit(f"not found: {args.target}")

    original = open(args.target).read()
    patched, changes = apply_patch(original)

    print("Changes:")
    for c in changes:
        print(f"  - {c}")

    if patched == original:
        print("\nnothing to write")
        return
    if args.check:
        print("\n--check given; no file written")
        return

    backup = args.target + ".prephase"
    shutil.copy(args.target, backup)
    with open(args.target, "w") as f:
        f.write(patched)

    import ast
    try:
        ast.parse(patched)
    except SyntaxError as e:
        shutil.copy(backup, args.target)
        raise SystemExit(f"patched file does not parse ({e}); original restored")

    print(f"\nwrote   {args.target}")
    print(f"backup  {backup}")
    print("""
Next, in this order:

  1.  python3 scripts/check_controller_timing.py --controller reactive
      Both clocks should now report 140 s and the green phases should run
      53, 10, 47 and 10 seconds. If they do not, stop and report the output.

  2.  python3 scripts/check_cycle_length.py
      Confirms the signal programme is unchanged.

  3.  Rerun the experiment matrix. The stored results were produced with the
      adaptive controllers on a 144 s cycle and the fixed-time baseline on
      140 s, so they compare controllers on unequal terms and cannot stand.

        python3 scripts/run_experiment.py --jobs 8 \\
                --model-dir "$(pwd)/models_broad" \\
                --outdir results/experiment_corrected
        python3 scripts/analyse_experiment.py --outdir results/experiment_corrected

  4.  python3 scripts/detector_divergence.py --n-seeds 3
      Recompute on the corrected timing.

The runtime assertion added here will stop any future run whose cycle departs
from the specification, rather than allowing it to produce plausible results.
""")


if __name__ == "__main__":
    main()
