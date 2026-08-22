#!/usr/bin/env python3
"""
gen_paper_data.py -- generate every number in the paper from the simulator.

Writes paper/generated/*.tex:
  numbers.tex        \\newcommand macros for all scalar results
  trace_table.tex    worked-example control trace (16-point FFT, P_bu = 2)
  program_head.tex   first descriptors of the cfg-6 program
  program_hex.tex    encoded descriptor bytes for the worked example
  timeline.tex       TikZ bars for the batch-2 execution timeline
  plots.tex          pgfplots coordinate macros (batch, prefetch, PRS cost)
  breakdown.tex      cfg-6 per-module cycle table rows
  platforms.tex      issue-mechanism comparison rows
"""

import math
import os

import numpy as np

import bspnet as bs
import flexbe as fx
import microcode as mc

OUT = "paper/generated"
os.makedirs(OUT, exist_ok=True)

cfg, hw = bs.CONFIGS["cfg-6"], bs.HW["BSP-Flex"]
N = {}   # name -> string for numbers.tex


def num(name, value):
    N[name] = value


def fmt(v, dec=0):
    if isinstance(v, float) and dec:
        return f"{v:,.{dec}f}"
    return f"{v:,.0f}" if isinstance(v, (int, float)) else str(v)


# ---------------------------------------------------------------------------
# 1. headline engine numbers
# ---------------------------------------------------------------------------
num("HeadlineCycles", f"{fx.cycles_eq10(4, 32768, 1, 4, 16):,.0f}")
num("HeadlineUs", f"{fx.cycles_eq10(4, 32768, 1, 4, 16)/300:,.1f}")

tot = bs.total_cycles(cfg, hw)
num("CfgCycles", f"{tot:,.0f}")
num("CfgMs", f"{tot/(hw.f_mhz*1e6)*1e3:.3f}")
num("CfgUs", f"{tot/(hw.f_mhz*1e6)*1e6:.1f}")
num("CfgSps", f"{1e6/(tot/(hw.f_mhz*1e6)*1e6):,.0f}")

c = fx.prs_cost(16)
num("PrsMux", f"{c['mux2']:,}")
num("PrsRegs", f"{c['pipeline_regs']:,}")
c64 = fx.prs_cost(64)
num("PrsMuxBig", f"{c64['mux2']:,}")
num("XbarMuxBig", f"{c64['ports']*(c64['ports']-1):,}")

# fxp16 accuracy
fmt16 = fx.FixedPointFormat()
rng = np.random.default_rng(9)
xq = fmt16.quantise((rng.normal(size=1024) + 1j*rng.normal(size=1024))/8)
yq, stq = fx.FlexBE(16, fmt=fmt16).transform(xq, bitrev=True, stage_scale=0.5,
                                             datapath="cycle")
ref = np.fft.fft(xq)*stq.scale
err = np.linalg.norm(yq[0]-ref)/np.linalg.norm(ref)
num("SqnrDb", f"{20*math.log10(1/err):.1f}")

# ---------------------------------------------------------------------------
# 2. worked example: 16-point FFT on P_bu = 2  (control trace)
# ---------------------------------------------------------------------------
n_we, Pbu_we = 4, 2
m_we = (2*Pbu_we).bit_length() - 1
sched = fx.ButterflySchedule(n_we, Pbu_we, validate=True)
rows = []
for k, stage in enumerate(sched.stages):
    for j, cc in enumerate(stage):
        banks = fx.bsm_array(cc.indices, m_we)
        rows.append(
            f"{k} & {cc.hole} & {j} & "
            f"({','.join(map(str, cc.indices))}) & "
            f"({','.join(map(str, banks))}) & "
            f"({','.join(map(str, cc.depth_by_bank))}) & "
            f"{cc.R} & {cc.S} & ({','.join(map(str, cc.coeff_index))}) \\\\")
    rows.append("\\addlinespace[1pt]")
with open(f"{OUT}/trace_table.tex", "w") as f:
    f.write("\\newcommand{\\tracetablerows}{%\n" +
            "\n".join(rows[:-1]) + "}\n")

# verify the trace numerically and record the result
x_we = np.arange(16).astype(complex)
eng_we = fx.FlexBE(Pbu_we)
y_we, st_we = eng_we.transform(x_we, bitrev=True, datapath="cycle")
assert np.allclose(y_we[0], np.fft.fft(x_we))
num("WeCycles", f"{st_we.butterfly_cycles}")
num("WeBrev", f"{st_we.bitrev_cycles}")
num("WeStages", f"{st_we.n_stages}")
num("WeCPS", f"{sched.cycles_per_stage}")

# ---------------------------------------------------------------------------
# 3. descriptors: encodings for the worked example + cfg-6 program
# ---------------------------------------------------------------------------
def hex_words(d):
    raw = d.encode()
    words = [f"{int.from_bytes(raw[i:i+4], 'little'):08x}" for i in range(0, 32, 4)]
    return ("\\shortstack[l]{\\texttt{" + " ".join(words[:4]) +
            "}\\\\\\texttt{" + " ".join(words[4:]) + "}}")

examples = [
    ("bs.load", mc.Descriptor("bs.load", dst=0x0000, count=32768, tag=0),
     "load one 32{,}768-sample IQ record to BSPM word 0"),
    ("bs.bfly (FFT)", mc.Descriptor("bs.bfly", mode=mc.MODES["FFT"], log2_l=15,
                                    P_N=4, stage_hi=14, coeff=0, tag=5),
     "four 32{,}768-point FFTs, twiddles at CRAM 0"),
    ("bs.store (BITREV)", mc.Descriptor("bs.store", flags=mc.FLAGS["BITREV"],
                                        count=32768, tag=6),
     "bit-reversal write-back fused into the store"),
    ("bs.bfly (BL)", mc.Descriptor("bs.bfly", mode=mc.MODES["BL"], log2_l=5,
                                   P_N=1024, flags=mc.FLAGS["RELU"],
                                   coeff=0x40, stage_hi=4, tag=9),
     "1{,}024 length-32 BL transforms, fused ReLU"),
    ("bs.bfly (short)", mc.Descriptor("bs.bfly", mode=mc.MODES["BL_EXPAND"],
                                      log2_l=3, log2_Psub=2, P_N=4096,
                                      coeff=0x80, stage_hi=2, tag=8),
     "4{,}096 length-8 transforms, $P_{sub}{=}4$ folded per pass"),
]
with open(f"{OUT}/program_hex.tex", "w") as f:
    f.write("\\newcommand{\\programhexrows}{%\n")
    for name, d, note in examples:
        f.write(f"\\texttt{{{name}}} & {hex_words(d)} & {note} \\\\[2.5pt]\n")
    f.write("}\n")

cmds = mc.build_program(cfg, hw)
kinds = {}
for c_ in cmds:
    kinds[c_.kind] = kinds.get(c_.kind, 0) + 1
num("ProgCmds", f"{len(cmds)}")
num("ProgBytes", f"{mc.program_bytes(cmds):,}")
num("ProgDmaIn", f"{kinds['dma_in']}")
num("ProgDmaOut", f"{kinds['dma_out']}")
num("ProgCompute", f"{kinds['compute']}")

def esc(t):
    return t.replace("_", "\\_")

with open(f"{OUT}/program_head.tex", "w") as f:
    f.write("\\newcommand{\\programheadrows}{%\n")
    for c_ in cmds[:9]:
        detail = (f"{c_.nbytes/1024:.0f}\\,KB" if c_.nbytes
                  else f"{c_.cycles:,.0f}\\,cyc")
        deps = ",".join(map(str, c_.deps)) or "--"
        f.write(f"{c_.desc.tag} & \\texttt{{{esc(c_.name.split()[0])}}} & "
                f"{esc(' '.join(c_.name.split()[1:]))} & {c_.kind.replace('_','\\_')} & "
                f"{detail} & {deps} \\\\\n")
    f.write("\\multicolumn{6}{c}{$\\vdots$ (8 branches "
            "$\\times$ 11 descriptors)} \\\\\n")
    c_ = cmds[-1]
    f.write(f"{c_.desc.tag} & \\texttt{{{esc(c_.name.split()[0])}}} & "
            f"{esc(' '.join(c_.name.split()[1:]))} & {c_.kind.replace('_','\\_')} & "
            f"{c_.nbytes/1024:.0f}\\,KB & "
            f"{cmds[-1].deps[0]}..{cmds[-1].deps[-1]} \\\\\n")
    f.write("}\n")

# ---------------------------------------------------------------------------
# 4. per-module cycle breakdown
# ---------------------------------------------------------------------------
with open(f"{OUT}/breakdown.tex", "w") as f:
    f.write("\\newcommand{\\breakdownrows}{%\n")
    for name, cyc in bs.cycle_breakdown(cfg, hw):
        n2 = name.replace(" x ", " $\\times$ ").replace(", x", ", $\\times$").replace("{0,1}", "\\{0,1\\}")
        f.write(f"{n2} & {cyc:,.0f} & {cyc/(hw.f_mhz*1e6)*1e6:.2f} \\\\\n")
    f.write("\\midrule\ntotal & %s & %s \\\\}\n"
            % (f"{tot:,.0f}", f"{tot/(hw.f_mhz*1e6)*1e6:.1f}"))

# ---------------------------------------------------------------------------
# 5. platform comparison, sweeps, area, link
# ---------------------------------------------------------------------------
plat = mc.compare_platforms()
short = {"pynq": "PYNQ / Jupyter flow", "mmio": "MMIO per descriptor (Linux)",
         "ring": "ring rebuilt per inference",
         "static": "static ring + doorbell",
         "tuned": "static ring, split queues, streamed input"}
with open(f"{OUT}/platforms.tex", "w") as f:
    f.write("\\newcommand{\\platformrows}{%\n")
    for r in plat:
        f.write(f"{short[r['key']]} & {r['latency_us']:.1f} & "
                f"{r['overhead_pct']:.1f} & {r['throughput']:,.0f} & "
                f"{r['critical'].replace('_','\\_')} \\\\\n")
    f.write("}\n")
pd = {r["key"]: r for r in plat}
num("PynqUs", f"{pd['pynq']['latency_us']:.0f}")
num("PynqOvh", f"{pd['pynq']['overhead_pct']:.0f}")
num("PynqSps", f"{pd['pynq']['throughput']:,.0f}")
num("StaticUs", f"{pd['static']['latency_us']:.1f}")
num("StaticOvh", f"{pd['static']['overhead_pct']:.1f}")
num("TunedUs", f"{pd['tuned']['latency_us']:.1f}")
num("TunedOvh", f"{pd['tuned']['overhead_pct']:.1f}")
num("MmioOvh", f"{pd['mmio']['overhead_pct']:.1f}")

bsw = mc.batch_sweep()
num("PeakSps", f"{bsw[0]['peak']:,.0f}")
num("TunedBatchTenSps", f"{bsw[-1]['tuned']:,.0f}")
num("TunedBatchTenPct", f"{bsw[-1]['tuned']/bsw[-1]['peak']*100:.0f}")
num("StaticBatchTenSps", f"{bsw[-1]['static']:,.0f}")

psw = mc.ring_depth_sweep()
area = mc.sequencer_area()
num("SeqLut", f"{area['sequencer_lut']:,}")
num("SeqBram", f"{area['sequencer_bram36']}")
num("SeqPct", f"{area['lut_overhead_pct']:.1f}")
num("TotLut", f"{area['total_lut']:,}")

link = mc.port_budget()
num("LinkNeed", f"{link['required_gbs']:.2f}")
num("LinkHave", f"{link['available_gbs']:.1f}")
num("LinkPct", f"{link['utilisation']*100:.0f}")
num("InKB", f"{link['in_bytes']/1024:.0f}")
num("OutKB", f"{link['out_bytes']/1024:.1f}")

with open(f"{OUT}/plots.tex", "w") as f:
    f.write("\\newcommand{\\batchpeak}{" +
            " ".join(f"({r['batch']},{r['peak']:.0f})" for r in bsw) + "}\n")
    for k in ("pynq", "static", "tuned"):
        f.write(f"\\newcommand{{\\batch{k}}}{{" +
                " ".join(f"({r['batch']},{r[k]:.0f})" for r in bsw) + "}\n")
    f.write("\\newcommand{\\prefetchsweep}{" +
            " ".join(f"({r['depth']},{r['latency_us']:.1f})" for r in psw) + "}\n")
    f.write("\\newcommand{\\prscost}{" +
            " ".join(f"({fx.prs_cost(p)['ports']},{fx.prs_cost(p)['mux2']})"
                     for p in (2, 4, 8, 16, 32, 64)) + "}\n")
    f.write("\\newcommand{\\xbarcost}{" +
            " ".join(f"({1<<((2*p).bit_length()-1)},"
                     f"{(1<<((2*p).bit_length()-1))*((1<<((2*p).bit_length()-1))-1)})"
                     for p in (2, 4, 8, 16, 32, 64)) + "}\n")

# ---------------------------------------------------------------------------
# 6. execution timeline (batch 2, tuned platform)
# ---------------------------------------------------------------------------
p = mc.PLATFORMS["tuned"]
rep = mc.Sequencer(p).run(mc._stream(cfg, hw, 2, p.input_chunks), trace=True)
lane_y = {"compute": 0, "dma_in": 1, "dma_out": 2}
colors = ["barA", "barB"]
bars = []
for name, res, s0, e0, inf in rep.timeline:
    if e0 - s0 < 0.15 and res == "compute":
        e0 = s0 + 0.15                       # keep slivers visible
    bars.append((lane_y[res], s0, e0, colors[inf % 2]))
with open(f"{OUT}/timeline.tex", "w") as f:
    f.write("% lanes: 0=compute 1=dma_in 2=dma_out; x in us\n")
    for y, s0, e0, col in bars:
        f.write(f"\\fill[{col}] ({s0:.2f},{y+0.12}) rectangle "
                f"({e0:.2f},{y+0.88});\n")
num("TimelineEnd", f"{rep.latency_us:.1f}")
num("TimelineTwoPerInf", f"{rep.per_inference_us:.1f}")

# glue-command share
glue = sum(c_.cycles for c_ in cmds
           if c_.desc.opcode in ("bs.pow", "bs.mag", "bs.pool"))
num("GlueUs", f"{glue/(hw.f_mhz*1e6)*1e6:.1f}")
num("GluePct", f"{glue/tot*100:.0f}")

with open(f"{OUT}/numbers.tex", "w") as f:
    for k, v in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

print(f"wrote {len(N)} macros + tables to {OUT}/")
for k in sorted(N):
    print(f"  \\{k:24s} = {N[k]}")
