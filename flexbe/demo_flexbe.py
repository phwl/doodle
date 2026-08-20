#!/usr/bin/env python3
"""
demo_flexbe.py -- walk through the FlexBE simulator and reproduce the headline
numbers of the paper.

    python demo_flexbe.py            # fast (functional BSPNet at N = 4096)
    python demo_flexbe.py --full     # cycle-level 4 x 32768-point FFT as well
"""

import argparse
import math
import time

import numpy as np

import bspnet as bs
import flexbe as fx


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
def demo_schedule():
    rule("1. Conflict-free schedule and PRS control (Sec. 3.1, Alg. 1)")
    N, P_bu = 1024, 16
    n, m = int(math.log2(N)), (2 * P_bu).bit_length() - 1
    sched = fx.ButterflySchedule(n, m, validate=True)
    print(f"N = {N}, P_bu = {P_bu}  ->  2*P_bu = {1<<m} lanes, m = {m}")
    print(f"stages = {sched.n_stages}, cycles/stage = {sched.cycles_per_stage}, "
          f"total = {sched.total_cycles} cycles")
    print("validated: bank-conflict free, complete coverage, PRS == FCS, "
          "R^i = bsm(I^i[0])\n")
    print(f"{'k':>2} {'hole h':>6} {'S^i (ours)':>11} {'S^i (Alg. 1)':>13} "
          f"{'R^i of first 4 cycles':>26}")
    for k, stage in enumerate(sched.stages):
        rs = ", ".join(str(c.R) for c in stage[:4])
        print(f"{k:>2} {stage[0].hole:>6} {stage[0].S:>11} "
              f"{sched.paper_subset_state(k):>13} {rs:>26}")
    print("\nexample cycle (stage 0, cycle 0):")
    c = sched.stages[0][0]
    print(f"  I^i          = {c.indices[:8].tolist()} ...")
    print(f"  banks bsm(I) = {fx.bsm_array(c.indices, m)[:8].tolist()} ...")
    print(f"  Addr^i       = {c.depth_by_bank[:8].tolist()} ...  (per bank)")
    print(f"  (R^i, S^i)   = ({c.R}, {c.S})")


# ---------------------------------------------------------------------------
def demo_fft():
    rule("2. FFT correctness: float and Q1.15 fixed point (Sec. 5.2)")
    N, P_bu = 4096, 16
    rng = np.random.default_rng(0)
    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    x /= 4 * np.max(np.abs(x))

    eng = fx.FlexBE(P_bu)
    t0 = time.time()
    y, st = eng.transform(x, bitrev=True, datapath="cycle")
    dt = time.time() - t0
    ref = np.fft.fft(x)
    err = np.max(np.abs(y[0] - ref)) / np.max(np.abs(ref))
    print(f"float64   N={N}: {st}")
    print(f"           max relative error vs numpy.fft = {err:.2e}  "
          f"(simulated in {dt:.2f} s)")

    fmt = fx.FixedPointFormat()
    engq = fx.FlexBE(P_bu, fmt=fmt)
    xq = fmt.quantise(x)
    yq, stq = engq.transform(xq, bitrev=True, stage_scale=0.5, datapath="cycle")
    refq = np.fft.fft(xq) * stq.scale
    e = np.linalg.norm(yq[0] - refq) / np.linalg.norm(refq)
    print(f"fxp16 Q1.15 (1/2 per stage): NRMSE = {e:.2e}  "
          f"-> SQNR = {20*math.log10(1/e):.1f} dB")


# ---------------------------------------------------------------------------
def demo_subparallel():
    rule("3. Sub-parallelism: transforms shorter than 2*P_bu (Sec. 3.2, Fig. 3)")
    P_bu = 16
    eng = fx.FlexBE(P_bu)
    print(f"{'l':>4} {'N':>5} {'P_sub':>6} {'stages':>7} "
          f"{'cycles / 1024 vectors':>22} {'vs zero padding':>16}")
    for l in (2, 4, 8, 16, 32, 64):
        N, P_sub = eng.geometry(l)
        flex = fx.layer_cycles(1024, l, P_be=1, P_bu=P_bu)
        pad = fx.layer_cycles(1024, max(l, 2 * P_bu), P_be=1, P_bu=P_bu)
        print(f"{l:>4} {N:>5} {P_sub:>6} {int(math.log2(l)):>7} "
              f"{flex:>22,.0f} {pad/flex:>15.2f}x")

    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    y, st = eng.transform(x, bitrev=True, datapath="cycle")
    ok = np.allclose(y, np.fft.fft(x, axis=1))
    print(f"\neight 8-point FFTs on one P_bu=16 engine: P_sub={st.P_sub}, "
          f"{st.butterfly_cycles} butterfly cycles, matches numpy: {ok}")


# ---------------------------------------------------------------------------
def demo_bitrev():
    rule("4. Integrated bit-reversal write-back (Sec. 3.3, Alg. 3)")
    for N, P_bu in [(1024, 16), (32768, 16), (4096, 4)]:
        n, m = int(math.log2(N)), (2 * P_bu).bit_length() - 1
        j, k = fx.bitrev_schedule(n, m)
        P = 1 << m
        src_ok = all(len(np.unique(fx.bsm_array(jv, m))) == P for jv in j)
        dst_ok = all(len(np.unique(kv & (P - 1))) == P for kv in k)
        perm_ok = len(np.unique(j)) == N and np.array_equal(
            k, fx.bit_rev_array(n)[j])
        print(f"N={N:6d} P={P:3d}: {j.shape[0]:6d} cycles "
              f"(bound N/P = {N//P:6d})  read-conflict-free={src_ok}  "
              f"write-conflict-free={dst_ok}  permutation-correct={perm_ok}")


# ---------------------------------------------------------------------------
def demo_switch():
    rule("5. PRS versus FCS interconnect (Sec. 3.1, Fig. 9)")
    print(f"{'ports 2*P_bu':>12} {'m':>3} {'FCS 2:1 muxes':>15} "
          f"{'PRS 2:1 muxes':>15} {'saving':>8} "
          f"{'FCS regs':>10} {'PRS regs':>10}")
    for P_bu in (2, 4, 8, 16, 32, 64):
        c = fx.interconnect_cost(P_bu)
        print(f"{c['ports']:>12} {c['m']:>3} {c['fcs_mux2']:>15,} "
              f"{c['prs_mux2']:>15,} {c['ratio']:>7.2f}x "
              f"{c['fcs_pipeline_regs']:>10,} {c['prs_pipeline_regs']:>10,}")
    print("\nFCS = Theta(P_bu^2) (Eq. 4), PRS = Theta(m*P_bu); registers are for "
          "a fully pipelined 16-bit datapath.")


# ---------------------------------------------------------------------------
def demo_cycles(full=False):
    rule("6. Cycle model, Eq. (10)")
    print("abstract: four 32,768-point FFTs, P_be = 4, P_bu = 16")
    print(f"  Eq. (10) -> {fx.cycles_eq10(4, 32768, 1, 4, 16):,.0f} cycles "
          f"(paper: 15,360)")
    print(f"  at 300 MHz that is "
          f"{fx.cycles_eq10(4, 32768, 1, 4, 16)/300e6*1e6:.1f} us")

    if full:
        print("\nrunning the real 4 x 32768-point transform through the "
              "cycle-level datapath ...")
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 32768)) + 1j * rng.normal(size=(4, 32768))
        arr = fx.BEArray(P_be=4, P_bu=16)
        t0 = time.time()
        y, st = arr.transform(x, bitrev=True, datapath="cycle")
        dt = time.time() - t0
        err = np.max(np.abs(y - np.fft.fft(x, axis=1)))
        print(f"  measured: {st.butterfly_cycles:,} butterfly cycles + "
              f"{st.bitrev_cycles:,} bit-reversal cycles")
        print(f"  max abs error vs numpy.fft = {err:.2e}   "
              f"(simulation took {dt:.1f} s)")


# ---------------------------------------------------------------------------
def demo_bspnet(full=False):
    rule("7. BSPNet design-space exploration (Tables 7/10, Fig. 11)")
    print(bs.design_space_table())
    print("\nlatencies use F_max from Table 10 (BE-base-1 210 MHz, "
          "BE-base-2 115 MHz, FlexBE 310 MHz).")

    cfg = bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    rule("8. cfg-6 module breakdown on BSP-Flex (300 MHz, P_be=4, P_bu=16)")
    tot = 0.0
    for name, cyc in bs.cycle_breakdown(cfg, hw):
        tot += cyc
        print(f"  {name:44s} {cyc:12,.0f} cycles  {cyc/300e6*1e3:8.4f} ms")
    print(f"  {'total':44s} {tot:12,.0f} cycles  {tot/300e6*1e3:8.4f} ms")
    print(f"  single-batch throughput: "
          f"{bs.throughput_sps(cfg, hw):,.0f} samples/s")

    rule("9. BSPNet functional run on the engine array")
    N = 32768 if full else 4096
    small = bs.BSPNetConfig("demo", d_in=cfg.d_in, d_m=cfg.d_m,
                            L=min(cfg.L, N // cfg.d_in // 4),
                            n_mlp_blocks=cfg.n_mlp_blocks,
                            bit_reversal=True, N=N)
    net = bs.BSPNet(small, hw, seed=0)
    rng = np.random.default_rng(2)
    s = (rng.normal(size=N) + 1j * rng.normal(size=N)) / 8
    t0 = time.time()
    logits = net.forward(s)
    dt = time.time() - t0
    fe = net.features(s)
    ref = net.features_reference(s)
    ferr = max(np.max(np.abs(fe[k] - ref[k])) for k in ref)
    print(f"input: {N} complex samples;  engine cycles issued: {net.cycles:,}")
    print(f"Eq. (1) features S2..S8 / F2..F8 max error vs numpy: {ferr:.2e}")
    print(f"logits ({small.n_classes} classes): "
          f"{np.array2string(logits, precision=3)}")
    print(f"(random weights -- the point is that one datapath ran the "
          f"{N}-point FFTs and every BL layer;  {dt:.1f} s)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="also run the full 32768-point cycle-level workloads")
    args = ap.parse_args()

    demo_schedule()
    demo_fft()
    demo_subparallel()
    demo_bitrev()
    demo_switch()
    demo_cycles(args.full)
    demo_bspnet(args.full)
    print()


if __name__ == "__main__":
    main()
