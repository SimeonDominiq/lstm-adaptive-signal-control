"""
Direct test of the detector counting rule.

The conservation checks assert that detector counts never exceed the vehicles
released. That is an upper bound: it catches over-counting, which is what the
boundary double-count did, but it cannot show that each vehicle is counted
exactly once. A vehicle missed and another counted twice would pass it.

This test establishes the stronger property directly. It reads the two lines
that implement the rule from run_controller.py, so it tests the code actually
used rather than a copy of it, and drives them with per-step sets of vehicle
identities constructed so that the correct count is known in advance. No
simulator is needed: the rule is set arithmetic on identities.

Run from the project root:
    python3 scripts/test_detector_counting.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "run_controller.py")

# --------------------------------------------------------------------------
# bind the test to the implemented rule
# --------------------------------------------------------------------------
src = open(SRC).read()
ADD = "counts_acc[i] += len(now - prev_on_loop[i])"
KEEP = "prev_on_loop[i] = now"
for line in (ADD, KEEP):
    if line not in src:
        sys.exit(f"FAIL: the counting rule in run_controller.py has changed; "
                 f"expected to find:\n    {line}")


def implemented_rule(steps):
    """Apply the two implemented lines to a sequence of per-step occupant sets
    for one loop, returning the count accumulated."""
    counts_acc, prev_on_loop, i = [0], [set()], 0
    for now in steps:
        now = set(now)
        exec(ADD, {}, {"counts_acc": counts_acc, "now": now, "prev_on_loop": prev_on_loop, "i": i})
        exec(KEEP, {}, {"now": now, "prev_on_loop": prev_on_loop, "i": i})
    return counts_acc[0]


def counted_per_cycle(steps, boundary):
    """Split a run at a cycle boundary, keeping the loop's memory across it,
    as the controller does, and return the count in each cycle."""
    counts_acc, prev_on_loop, i = [0], [set()], 0
    per = []
    for k, now in enumerate(steps):
        if k == boundary:
            per.append(counts_acc[0]); counts_acc[0] = 0
        now = set(now)
        exec(ADD, {}, {"counts_acc": counts_acc, "now": now, "prev_on_loop": prev_on_loop, "i": i})
        exec(KEEP, {}, {"now": now, "prev_on_loop": prev_on_loop, "i": i})
    per.append(counts_acc[0])
    return per


def superseded_rule_per_cycle(steps, boundary):
    """The method the patch replaced: gather identities in a set and clear it
    at each cycle boundary."""
    seen, per = set(), []
    for k, now in enumerate(steps):
        if k == boundary:
            per.append(len(seen)); seen = set()
        seen |= set(now)
    per.append(len(seen))
    return per


# --------------------------------------------------------------------------
# cases: each has a known correct answer
# --------------------------------------------------------------------------
RESULTS = []


def check(name, got, want):
    ok = got == want
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         expected {want}, got {got}")


print("DETECTOR COUNTING - DIRECT TEST")
print("=" * 62)

# 1. a vehicle dwelling on the loop for several steps is one vehicle
check("one vehicle dwelling for five steps is counted once",
      implemented_rule([{"v1"}] * 5), 1)

# 2. the case the fault concerned: a vehicle on the loop as the boundary passes
straddle = [set(), {"v1"}, {"v1"}, {"v1"}, {"v1"}, set()]   # boundary at step 3
check("a vehicle straddling a cycle boundary is counted once in total",
      sum(counted_per_cycle(straddle, 3)), 1)
check("and it is attributed to the cycle in which it arrived",
      counted_per_cycle(straddle, 3), [1, 0])
check("the superseded method counted the same vehicle twice",
      sum(superseded_rule_per_cycle(straddle, 3)), 2)

# 3. distinct vehicles are counted separately
check("two vehicles in succession are counted twice",
      implemented_rule([{"v1"}, {"v1"}, set(), {"v2"}, {"v2"}]), 2)

# 4. one leaves and another arrives in the same step
check("a departure and an arrival in the same step are both counted",
      implemented_rule([{"v1"}, {"v1"}, {"v2"}, {"v2"}]), 2)

# 5. two vehicles on the loop together (adjacent lanes are separate loops,
#    but a long vehicle and a following one can overlap one loop briefly)
check("two vehicles overlapping on the loop are counted twice",
      implemented_rule([{"v1"}, {"v1", "v2"}, {"v2"}]), 2)

# 6. a queue standing on the loop across many cycles
standing = [{"q1", "q2"}] * 40
check("a queue standing on the loop across several cycles is counted once per vehicle",
      sum(counted_per_cycle(standing, 20)), 2)
check("the superseded method counted it again in every cycle",
      sum(superseded_rule_per_cycle(standing, 20)), 4)

# 7. nothing on the loop
check("an empty loop counts nothing",
      implemented_rule([set()] * 10), 0)

print("=" * 62)
passed = sum(RESULTS)
print(f"{passed} passed, {len(RESULTS) - passed} failed")
sys.exit(0 if passed == len(RESULTS) else 1)
