"""
Fix the boundary double-count in the controller's detector reading.

THE FAULT
The controller accumulated the identities of vehicles occupying each loop into a
set, and cleared that set at every cycle boundary. A vehicle still on the loop
when the boundary passed was therefore recorded in the cycle that ended and
again in the cycle that began. At roughly one to two seconds of occupancy per
vehicle against a 140 second cycle, on the order of one to two per cent of
vehicles straddle a boundary. The diagnostic measured 99 excess crossings
against 4,268 vehicles released, which is 2.3 per cent, and confirmed the excess
is spread evenly across the run rather than concentrated at its start or
attributable to a one-cycle misalignment.

THE FIX
Count a vehicle when it FIRST appears on the loop rather than accumulating and
clearing per cycle. The controller keeps the set of identities present on each
loop in the previous step, and increments the count only for identities that
were not present then. A vehicle is therefore counted once per continuous
occupancy, wherever the cycle boundary falls.

WHAT IS AND IS NOT AFFECTED
The training datasets are unaffected: extract_dataset.py reads SUMO's own
nVehContrib aggregation, which counts passages correctly, so the model was
trained on clean counts. The control decisions are only slightly affected,
because the inflation applied to all eight movements and the control law uses
each phase's share of the total, in which a common factor largely cancels. The
detector divergence figures ARE affected and must be recomputed after this fix.

    python3 scripts/patch_detector_count.py            # apply
    python3 scripts/patch_detector_count.py --check    # report only
"""

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "run_controller.py")

OLD_INIT = """    # Count each vehicle once per cycle, by identity."""

NEW_INIT_MARKER = "prev_on_loop"

OLD_SEEN_BLOCK = re.compile(
    r"    # Count each vehicle once per cycle, by identity\.\n"
    r"(?:    #[^\n]*\n|    #\n)*"
    r"    seen = \[set\(\) for _ in MOVEMENTS\]", re.M)

OLD_ACCUM = re.compile(
    r"        # accumulate detector counts for the cycle in progress\n"
    r"        for i, m in enumerate\(MOVEMENTS\):\n"
    r"            for vid in traci\.inductionloop\.getLastStepVehicleIDs\(f\"det_\{m\}\"\):\n"
    r"                seen\[i\]\.add\(vid\)\n", re.M)

NEW_ACCUM = (
"        # Count a vehicle when it first appears on the loop.\n"
"        #\n"
"        # Accumulating identities and clearing them at each cycle boundary\n"
"        # counts a vehicle twice when it is still on the loop as the boundary\n"
"        # passes: once in the cycle that ends and once in the cycle that\n"
"        # begins. Comparing against the previous step's occupants instead\n"
"        # counts each vehicle once per continuous occupancy, wherever the\n"
"        # boundary falls.\n"
"        for i, m in enumerate(MOVEMENTS):\n"
"            now = set(traci.inductionloop.getLastStepVehicleIDs(f\"det_{m}\"))\n"
"            counts_acc[i] += len(now - prev_on_loop[i])\n"
"            prev_on_loop[i] = now\n")

OLD_SNAPSHOT = re.compile(
    r"            counts = np\.array\(\[len\(sv\) for sv in seen\], dtype=float\)", re.M)
NEW_SNAPSHOT = "            counts = counts_acc.copy()"

OLD_RESET = re.compile(
    r"            for sv in seen:\n                sv\.clear\(\)", re.M)
NEW_RESET = "            counts_acc[:] = 0.0"


def apply_patch(text):
    changes = []

    if NEW_INIT_MARKER in text:
        return text, ["already patched - no change made"]

    new, n = OLD_SEEN_BLOCK.subn(
        "    # Per-movement running count for the cycle in progress, and the set of\n"
        "    # identities occupying each loop in the previous step. Counting only\n"
        "    # newly appeared identities avoids double-counting a vehicle that is\n"
        "    # still on the loop when a cycle boundary passes.\n"
        "    counts_acc = np.zeros(len(MOVEMENTS), dtype=float)\n"
        "    prev_on_loop = [set() for _ in MOVEMENTS]",
        text)
    if n != 1:
        raise SystemExit(f"could not locate the accumulator declaration (found {n})")
    changes.append("replaced the per-cycle identity sets with a running count plus previous-step sets")
    text = new

    text, n = OLD_ACCUM.subn(NEW_ACCUM, text)
    if n != 1:
        raise SystemExit(f"could not locate the accumulation loop (found {n})")
    changes.append("accumulation now counts only identities absent from the previous step")

    text, n = OLD_SNAPSHOT.subn(NEW_SNAPSHOT, text)
    if n != 1:
        raise SystemExit(f"could not locate the per-cycle snapshot (found {n})")
    changes.append("cycle snapshot now taken from the running count")

    text, n = OLD_RESET.subn(NEW_RESET, text)
    if n != 1:
        raise SystemExit(f"could not locate the per-cycle reset (found {n})")
    changes.append("per-cycle reset now zeroes the running count and retains loop occupancy")

    return text, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without writing")
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

    backup = args.target + ".prepatch"
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
    print("\nNext:")
    print("  1.  python3 scripts/diagnose_detector_excess.py")
    print("      The ratio over logged cycles should fall close to one. It will not")
    print("      reach exactly one: vehicles released in cycles the controller never")
    print("      logs, and vehicles still in transit when the run ends, both remain.")
    print("  2.  python3 scripts/detector_divergence.py --n-seeds 3")
    print("      Recompute the divergence figures on corrected counts.")
    print("  3.  Re-examine whether the controller comparison needs rerunning. The")
    print("      inflation applied to all movements and the control law uses each")
    print("      phase's share of the total, so a common factor largely cancels, but")
    print("      this should be tested rather than assumed.")


if __name__ == "__main__":
    main()
