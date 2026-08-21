#!/usr/bin/env python3
"""
demo_isa.py -- microcoded accelerator vs custom vector extension, quantified.

    python demo_isa.py

Everything below comes from generated-and-executed instruction traces run
through the issue model of isa_model.py; only the per-class rates, issue width
and clock are assumptions (all in MachineConfig).
"""

import math

import numpy as np

import bspnet as bs
import flexbe as fx
import isa_model as im


def rule(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def demo_verify():
    rule("0. The instruction traces are real (generated, then executed)")
    print(f"{'l':>6} {'n_seq':>6} {'VLEN':>6} {'VL':>5} {'variant':>24} "
          f"{'instr':>8} {'err vs numpy':>14}")
    for l, n_seq, vlen in [(8, 4096, 16384), (32, 1024, 16384), (4096, 1, 4096)]:
        for tag, kw in [("Zvbfly fused", dict(fused_butterfly=True)),
                        ("Zvshfl explicit", dict(fused_butterfly=False)),
                        ("RVV 1.0 vrgather",
                         dict(fused_butterfly=False, has_shuffle=False,
                              has_vtwid=False, has_complex=False))]:
            mc = im.MachineConfig("t", vlen_bits=vlen, lanes=128, **kw)
            rng = np.random.default_rng(l)
            x = rng.normal(size=(n_seq, l)) + 1j * rng.normal(size=(n_seq, l))
            mem = {"x": x.reshape(-1).astype(complex).copy()}
            prog = im.gen_transform(l, n_seq, mc)
            im.run_program(prog, mem, mc)
            rev = fx.bit_rev_array(int(math.log2(l)))
            ref = np.fft.fft(x, axis=1)[:, np.argsort(rev)]
            err = np.max(np.abs(mem["x"].reshape(n_seq, l) - ref))
            print(f"{l:>6} {n_seq:>6} {vlen:>6} {mc.VL:>5} {tag:>24} "
                  f"{len(prog):>8,} {err:>14.2e}")


def demo_compare():
    rule("1. cfg-6 single-batch inference on every option")
    rows = im.compare_options()
    print(f"{'option':44s} {'cycles':>10s} {'MHz':>5s} {'ms':>7s} {'x hw':>6s} "
          f"{'LUT':>8s} {'DSP':>5s} {'BRAM':>5s} {'fit':>4s} {'neck':>6s} "
          f"{'inf/s/kLUT':>10s}")
    for r in rows:
        print(f"{r['option'][:44]:44s} {r['cycles']:10,.0f} {r['f_mhz']:5.0f} "
              f"{r['latency_ms']:7.3f} {r['speedup_vs_hardwired']:6.2f} "
              f"{r['lut']:8,} {r['dsp']:5,} {r['bram']:5,} "
              f"{'yes' if r['fits'] else 'NO':>4s} {r['bottleneck']:>6s} "
              f"{r['perf_per_klut']:10.2f}")
    print("\nAll options carry the same 64 radix-2 butterfly lanes.  Area for the "
          "vector\noptions counts datapath + permute + VRF + banked scratchpad + "
          "core; the\naccelerator options start from the Table 11 BSP-Flex system.")


def demo_breakdown():
    rule("2. Where option C's cycles go (cfg-6, VLEN 16384, 128 lanes)")
    cfg = bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    acc = dict((n, c) for n, c in bs.cycle_breakdown(cfg, hw))
    for key in ("C-fused", "C-wide"):
        mc = im.MACHINES[key]
        tot, items = im.bspnet_cycles_isa(cfg, mc)
        print(f"\n{mc.name}")
        print(f"  {'layer':44s} {'accel':>10s} {'vector':>10s} {'ratio':>7s} "
              f"{'neck':>6s}")
        for name, cyc, neck in items:
            a = acc.get(name, float("nan"))
            print(f"  {name:44s} {a:10,.0f} {cyc:10,.0f} {cyc/a:7.2f} {neck:>6s}")
        print(f"  {'total':44s} {sum(acc.values()):10,.0f} {tot:10,.0f} "
              f"{tot/sum(acc.values()):7.2f}")


def demo_mix():
    rule("3. Instruction mix, 8-point BL layer (4096 rows) -- the BSPNet hot spot")
    for key in ("C-fused", "C-explicit", "D-rvv10"):
        mc = im.MACHINES[key]
        prog = im.gen_transform(8, 4096, mc)
        rep = im.IssueModel(mc).run(prog)
        mix = {}
        for i in prog:
            mix[i.cls] = mix.get(i.cls, 0) + 1
        order = ["load", "store", "twid", "perm", "arith", "mul"]
        counts = "  ".join(f"{c}={mix.get(c,0):>6,}" for c in order)
        print(f"\n{mc.name}\n  {counts}")
        print(f"  {rep}")


def demo_vlen():
    rule("4. VLEN sweep (option C-fused, cfg-6)")
    print(f"{'VLEN':>7} {'VL':>6} {'cycles':>11} {'ms':>8} {'x hw':>6} "
          f"{'LUT':>8} {'BRAM':>6} {'neck':>6}")
    hw_ms = bs.latency_ms(bs.CONFIGS["cfg-6"], bs.HW["BSP-Flex"])
    for r in im.vlen_sweep():
        print(f"{r['vlen']:>7} {r['VL']:>6} {r['cycles']:>11,.0f} "
              f"{r['latency_ms']:>8.3f} {r['latency_ms']/hw_ms:>6.2f} "
              f"{r['lut']:>8,} {r['bram']:>6,} {r['bottleneck']:>6}")
    print("\nVLEN saturates almost immediately: past VL = 64 the machine is not "
          "short of\nvector length, it is short of operand bandwidth.")


def demo_lsu():
    rule("5. LSU sensitivity -- the decisive parameter (option C-fused, cfg-6)")
    print(f"{'LSU elem/cyc':>13} {'B/cycle':>8} {'cycles':>11} {'ms':>8} "
          f"{'x hw':>6} {'neck':>6}")
    for r in im.lsu_sweep():
        print(f"{r['lsu']:>13} {r['bytes_per_cycle']:>8} {r['cycles']:>11,.0f} "
              f"{r['latency_ms']:>8.3f} {r['ratio']:>6.2f} {r['bottleneck']:>6}")
    print("\nClosing the gap needs ~512 B/cycle of load-store bandwidth, i.e. a "
          "wide banked\non-chip scratchpad feeding the vector unit -- which is "
          "option B with extra\nsteps: the operands stop living in memory and "
          "start living next to the BUs.")


def main():
    demo_verify()
    demo_compare()
    demo_breakdown()
    demo_mix()
    demo_vlen()
    demo_lsu()
    print()


if __name__ == "__main__":
    main()
