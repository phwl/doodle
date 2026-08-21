#!/usr/bin/env python3
"""
demo_zynq.py -- how the accelerator attaches to a Zynq UltraScale+ MPSoC.

    python demo_zynq.py

Accelerator cycles come from bspnet.cycle_breakdown; only the PS-side latencies
are new assumptions, and they all live in zynq_model.PSPLConfig.
"""

import bspnet as bs
import zynq_model as zm


def rule(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def main():
    cfg, hw = bs.CONFIGS["cfg-6"], bs.HW["BSP-Flex"]
    peak_us = bs.total_cycles(cfg, hw) / (hw.f_mhz * 1e6) * 1e6
    cmds = zm.bspnet_commands(cfg, hw)

    rule("1. Command stream for one cfg-6 inference")
    kinds = {}
    for c in cmds:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    print(f"{len(cmds)} commands: " +
          ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"datapath time (Table 11 model): {peak_us:.1f} us at {hw.f_mhz:.0f} MHz")
    print("\nfirst commands:")
    for c in cmds[:8]:
        d = f"{c.nbytes/1024:.0f} KB" if c.nbytes else f"{c.cycles:,.0f} cyc"
        print(f"  {c.name:34s} {c.kind:8s} {d:>12s}  deps={list(c.deps)}")

    rule("2. Issue mechanism, batch 1")
    print(f"{'platform':54s} {'us':>8s} {'ovh%':>7s} {'sps':>7s} {'critical':>9s}")
    for r in zm.compare_platforms():
        print(f"{r['name'][:54]:54s} {r['latency_us']:8.1f} "
              f"{r['overhead_pct']:7.1f} {r['throughput']:7.0f} "
              f"{r['critical']:>9s}")
    print("\nThe PYNQ flow is host bound: ~104 commands x (Python call + posted\n"
          "MMIO + polled completion) swamps a 214 us inference.  Any descriptor\n"
          "scheme lands within ~10% of the datapath, and the residual is the\n"
          "128 KB input transfer, which cannot overlap anything at batch 1.")

    rule("3. Descriptor prefetch depth (static ring, batch 1)")
    for r in zm.ring_depth_sweep():
        print(f"  depth {r['depth']:3d}: {r['latency_us']:7.1f} us  "
              f"+{r['overhead_pct']:5.1f}%   critical={r['critical']}")
    print("\n  4-8 descriptors is enough; deeper buys nothing at batch 1.")

    rule("4. Batch scaling (samples/s) -- compare with Fig. 15a")
    print(f"{'batch':>6} {'PYNQ':>9} {'static ring':>12} {'tuned R5':>10} "
          f"{'peak':>9} {'tuned/peak':>11}")
    for r in zm.batch_sweep(keys=("pynq", "static-a53", "tuned-r5")):
        print(f"{r['batch']:>6} {r['pynq']:>9.0f} {r['static-a53']:>12.0f} "
              f"{r['tuned-r5']:>10.0f} {r['peak']:>9.0f} "
              f"{r['tuned-r5']/r['peak']:>10.0%}")
    print("\n  'tuned' = static ring + per-resource queues + streamed input.\n"
          "  Per-resource queues matter only across inferences: with one shared\n"
          "  ring the next sample's load descriptor sits behind ~100 compute\n"
          "  descriptors and its DMA cannot start early.")

    rule("5. Link budget -- the PS-PL ports are not the constraint")
    b = zm.port_budget()
    print(f"  in {b['in_bytes']/1024:.0f} KB + out {b['out_bytes']/1024:.1f} KB "
          f"per {b['latency_ms']:.3f} ms inference")
    print(f"  required {b['required_gbs']:.2f} GB/s of {b['available_gbs']:.1f} "
          f"available ({b['utilisation']:.0%}); minimum HP ports = "
          f"{b['min_hp_ports']}")
    print("\n  The four HP ports buy burst overlap for a single record, not\n"
          "  throughput.  Bandwidth is never the limiter at these sizes.")
    print()


if __name__ == "__main__":
    main()
