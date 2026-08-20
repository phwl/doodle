#!/usr/bin/env python3
"""
check_algorithm1.py -- consistency analysis of Algorithm 1 of the manuscript

Algorithm 1 has two halves:

  lines 5-6  the index formula      ii   = rotate_{n-1}(base, k+1)
                                    I^i[0] = rotate_n(2*ii, k)
  lines 7-11 the PRS control law    R^i = bsm(I^i[0])
                                    S^i = n-k-1  if (n-m) <= k <= (n-2), else 0

The control law of lines 7-11 is exactly what a conflict-free shift-down
schedule requires, provided stage k operates on the index bit

        h = n - 1 - k      (decimation in frequency)

because then S^i is simply the hole position whenever the hole falls inside the
m low-order bits, and S^i = 0 otherwise.  That is what flexbe.py implements and
what the test suite verifies.

The index formula of lines 5-6, read literally, is harder to reconcile: for
every size tested it selects two cycle representatives that cannot lie in
different conflict-free cycles.  This script makes that concrete in two ways:

  (a) it checks whether the I^i[0] values hit each cycle of the simulator's
      schedule exactly once, and
  (b) for small N it enumerates *every* conflict-free grouping of a stage and
      asks whether any of them admits the I^i[0] values as distinct cycle
      representatives.

Both rotation readings (k+1 as printed, and k) and both stage orders (h=n-1-k
and h=k) are tried.  This is offered as a cross-check on the listing, not on the
RTL: the released generator presumably emits a working sequence, so the most
likely explanation is a transcription slip in lines 5-6 or in how `base` and the
within-cycle index vector I^i[1..2P_bu-1] are meant to be enumerated.
"""

import itertools
import math

import numpy as np

import flexbe as fx


def alg1_I0(n: int, P_bu: int, k: int, j: int, rot_offset: int = 1) -> int:
    """Algorithm 1, lines 5-6 (rot_offset = 1 reproduces the listing)."""
    base = j * P_bu
    ii = fx.rotl(base, k + rot_offset, n - 1)
    return fx.rotl(2 * ii, k, n)


# ---------------------------------------------------------------------------
def check_against_schedule(n, P_bu, hole_rule, rot_offset):
    """(a) do the I^i[0] values hit each simulator cycle exactly once?"""
    m = (2 * P_bu).bit_length() - 1
    N, P = 1 << n, 1 << m
    for k in range(n):
        h = hole_rule(n, k)
        groups = list(fx._stage_groups(n, h, m))
        gid = np.zeros(N, dtype=int)
        for g, (idx, _, _) in enumerate(groups):
            gid[idx] = g
        reps = [alg1_I0(n, P_bu, k, j, rot_offset) for j in range(N // P)]
        if sorted(gid[reps]) != list(range(len(groups))):
            return k, h, reps
    return None


# ---------------------------------------------------------------------------
def _matchings(items, size):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for comb in itertools.combinations(rest, size - 1):
        remain = [x for x in rest if x not in comb]
        for tail in _matchings(remain, size):
            yield [(first,) + comb] + tail


def check_exhaustive(n, P_bu, hole_rule, rot_offset):
    """(b) is ANY conflict-free grouping compatible with the I^i[0] values?"""
    m = (2 * P_bu).bit_length() - 1
    N, P = 1 << n, 1 << m
    for k in range(n):
        h = hole_rule(n, k)
        lower = [a for a in range(N) if not (a >> h) & 1]
        parts = [g for g in _matchings(lower, P_bu)
                 if all(len({fx.bsm(x, m) for a in grp for x in (a, a + (1 << h))}) == P
                        for grp in g)]
        reps = [alg1_I0(n, P_bu, k, j, rot_offset) for j in range(N // P)]

        def cycle_of(part, r):
            for i, grp in enumerate(part):
                if any(r in (a, a + (1 << h)) for a in grp):
                    return i
            return -1

        ok = any(len({cycle_of(p, r) for r in reps}) == len(reps) for p in parts)
        if not ok:
            return k, h, reps, len(parts)
    return None


# ---------------------------------------------------------------------------
HOLE_RULES = {
    "DIF  h = n-1-k": lambda n, k: n - 1 - k,
    "DIT  h = k    ": lambda n, k: k,
}


def main():
    print(__doc__)
    print("=" * 78)
    print("(a) do the Algorithm-1 representatives hit each schedule cycle once?")
    print("=" * 78)
    for n, P_bu in [(3, 2), (4, 2), (5, 2), (6, 4), (8, 4), (10, 16), (12, 16)]:
        m = (2 * P_bu).bit_length() - 1
        if n < m:
            continue
        for label, rule in HOLE_RULES.items():
            for rot in (1, 0):
                bad = check_against_schedule(n, P_bu, rule, rot)
                tag = ("consistent" if bad is None
                       else f"fails at k={bad[0]} (h={bad[1]}), reps={bad[2][:4]}")
                print(f"  N={1<<n:6d} P_bu={P_bu:3d}  {label}  "
                      f"rotate by k+{rot}:  {tag}")

    print("\n" + "=" * 78)
    print("(b) exhaustive: does any conflict-free grouping admit them?")
    print("=" * 78)
    for n, P_bu in [(3, 2), (4, 2), (5, 2)]:
        for label, rule in HOLE_RULES.items():
            for rot in (1, 0):
                bad = check_exhaustive(n, P_bu, rule, rot)
                if bad is None:
                    tag = "feasible"
                else:
                    k, h, reps, npart = bad
                    tag = (f"INFEASIBLE at k={k} (h={h}): none of the {npart} "
                           f"conflict-free partitions admits reps={reps}")
                print(f"  N={1<<n:6d} P_bu={P_bu:3d}  {label}  "
                      f"rotate by k+{rot}:  {tag}")

    print("\n" + "=" * 78)
    print("(c) the proposed repair of lines 5-6 (see docs/algorithm1_fix.md)")
    print("=" * 78)
    print("""  h <- n-1-k
  if h < m:   H <- j;                    I^i[2t] <- (H << m) + ins(t, h);  S^i <- h
  else:       e <- j mod 2                                                        
              H <- ins(j >> 1, h-m);     I^i[2t] <- (H << m) + 2t + e;     S^i <- 0
  both:       I^i[2t+1] <- I^i[2t] + 2^h;  R^i <- bsm(I^i[0])      (lines 7-11 unchanged)
""")
    for n, P_bu in [(3, 1), (4, 2), (6, 2), (6, 4), (9, 4), (10, 16), (12, 16)]:
        m = (2 * P_bu).bit_length() - 1
        if n < m:
            continue
        N, P = 1 << n, 1 << m
        conflict_free = complete = pairing = law = True
        for k in range(n):
            h, seen = n - 1 - k, np.zeros(N, dtype=bool)
            want_S = (n - k - 1) if (n - m) <= k <= (n - 2) else 0
            for j in range(N // P):
                c = fx.algorithm1_fixed(n, P_bu, k, j)
                banks = fx.bsm_array(c.indices, m)
                conflict_free &= len(np.unique(banks)) == P
                complete &= not seen[c.indices].any()
                seen[c.indices] = True
                pairing &= bool(np.all(c.indices[1::2] - c.indices[0::2] == (1 << h)))
                law &= (c.R == fx.bsm(int(c.indices[0]), m)) and (c.S == want_S)
            complete &= seen.all()
        print(f"  N={N:6d} P_bu={P_bu:3d}:  conflict-free={conflict_free}  "
              f"complete={complete}  pairing={pairing}  lines-7-11={law}")

    print("\nend-to-end: an engine driven only by the repaired listing")
    for N, P_bu in [(256, 4), (4096, 16)]:
        n = int(math.log2(N))
        eng = fx.FlexBE(P_bu)
        eng._sched[(n, 1)] = fx.ButterflySchedule.from_algorithm1_fixed(
            n, P_bu, validate=True)
        rng = np.random.default_rng(N)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        y, st = eng.transform(x, bitrev=True, datapath="cycle")
        err = np.max(np.abs(y[0] - np.fft.fft(x))) / np.max(np.abs(np.fft.fft(x)))
        print(f"  N={N:6d} P_bu={P_bu:3d}: {st.butterfly_cycles:6d} butterfly cycles, "
              f"max relative error vs numpy.fft = {err:.2e}")

    print("\nThe control law of lines 7-11 is reproduced exactly; see "
          "test_flexbe.TestAlgorithm1Fix.")


if __name__ == "__main__":
    main()
