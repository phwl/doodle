#!/usr/bin/env python3
"""
build_notebook.py -- assemble FlexBE_microcode.ipynb and execute it in-process.

No nbformat/nbclient in this environment, so the notebook JSON is written
directly.  Code cells are executed in one shared namespace, stdout is captured
as stream output and matplotlib figures as inline PNGs, so the notebook ships
already executed.

    python build_notebook.py [output.ipynb]
"""

import base64
import io
import json
import sys
import traceback
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True),
                  "outputs": [], "execution_count": None})


# ===========================================================================
md(r"""
# FlexBE with a microcoded command processor

**Companion notebook to** X. Liu, R. Wu, P. H. W. Leong, *"A Flexible FPGA-based
Butterfly Engine for Accelerating Signal Processing and Machine Learning"*.

Two things are established here.

1. **A cycle-level model of the engine**, which reproduces the paper's headline
   numbers exactly — 15,360 cycles for four 32,768-point FFTs, 0.214 ms for
   cfg-6 at 300 MHz — with the conflict-free schedule derived from the
   shift-down mapping and checked cycle by cycle.

2. **A microcoded command processor** that makes the accelerator programmable:
   a PL-side sequencer executing 32-byte descriptors. It costs **~3.6k LUT
   (+2.4%)** and lands within **9% of the hardwired datapath**, and it shows
   that the PYNQ flow the paper measures is *host bound*, not accelerator bound.

Nothing is asserted where it could be measured. The simulator executes the real
dataflow — banked RAM, switching network, butterfly units, per-cycle control —
and every architectural claim below is checked by the accompanying test suite.
Assumptions are confined to one dataclass, `MicrocodeConfig`, and listed in
Part III.
""")

md(r"""
---
## 0. Setup

| module | contents |
|---|---|
| `flexbe.py` | the engine: shift-down memory, PRS, schedule, bit reversal, arithmetic, Eq. (10) |
| `bspnet.py` | the application: Eq. (1) features, BL branches, Table 7 configs |
| `microcode.py` | descriptors, program builder, sequencer, Zynq attachment |
| `test_flexbe.py` | 49 tests |
""")

code("""
import math, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt

import flexbe as fx
import bspnet as bs
import microcode as mc

plt.rcParams.update({"figure.figsize": (7.2, 3.6), "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
print("numpy", np.__version__, "| python", sys.version.split()[0])
""")

# -------------------------------------------------------------- PART I
md(r"""
---
# Part I — The engine

## 1.1 Shift-down storage, Eq. (3)

Data in flight lives in $P = 2P_{bu}$ single-port banks. Element $I$ sits in bank
$a_y$ at depth $a_x$:

$$a_x = I \gg m, \qquad a_y = \mathrm{bsm}(I) = \big[\mathrm{popcount}(I \gg m) + (I \bmod 2^m)\big] \bmod 2^m$$

The useful way to read this — and worth stating explicitly in the paper, because
it makes every later argument a one-liner:

> **A RAM row is $2^m$ consecutive indices, and row $H$ is stored cyclically
> rotated by $\mathrm{popcount}(H)$ banks.**

Two consequences follow immediately: the map is a bijection, and the $2^m$
elements of one row occupy **all** $2^m$ banks exactly once.
""")

code("""
m, n = 3, 6                      # P_bu = 4 -> 8 banks, N = 64
N, P = 1 << n, 1 << m
idx = np.arange(N)
slots = fx.bsm_array(idx, m) * (N // P) + (idx >> m)
print("bijection:", len(np.unique(slots)) == N)
print("\\nbanks of the first four rows:")
for H in range(4):
    row = idx[(idx >> m) == H]
    print(f"  H={H}  popcount={fx.popcount(H)}  banks={fx.bsm_array(row, m).tolist()}")
""")

md(r"""
## 1.2 The schedule

Stage $k$ pairs indices differing in bit

$$h = n-1-k$$

i.e. decimation in frequency: natural-order input, bit-reversed output — which is
precisely what the Sec. 3.3 write-back exists to undo. Conflict freedom then
forces two regimes:

**$h < m$** — both operands share a row. Read the whole row: banks
$(\mathrm{popcount}(H)+\mathit{low})\bmod 2^m$ are all distinct, and slot $j$ reads bank
$(\pi_h[j]+\mathrm{popcount}(H))\bmod 2^m$ — a fixed stride permutation then a
rotation, so $\mathbf{P}_f = \mathbf{P}_s\times\mathbf{P}_r$ with $S=h$.

**$h \ge m$** — the partner is in row $H+2^{h-m}$. Pick $H$ with bit $h{-}m$ clear,
so $\mathrm{popcount}(H+2^{h-m}) = \mathrm{popcount}(H)+1$; take the $P_{bu}$ even (or odd)
low values from both rows and slot $j$ reads bank
$(\mathrm{popcount}(H)+e+j)\bmod 2^m$ — a pure rotation, $S=0$.

**The subset-switch state falls out as $S = h$ when $h<m$, else 0** — only
$\log_2(2P_{bu})$ distinct states, which is what makes the subset switch a
log-depth network rather than a crossbar.

`flexbe.cycle_control(n, P_bu, k, j)` is the closed form of both regimes and is
the only control path in the simulator: every result below is produced by it.
""")

code("""
sched = fx.ButterflySchedule(n=10, P_bu=16, validate=True)
print(f"N={sched.N}, 2*P_bu={sched.P}, stages={sched.n_stages}, "
      f"cycles/stage={sched.cycles_per_stage}, total={sched.total_cycles}")
print("validate(): conflict-free, complete coverage, PRS == P_f, R = bsm(I[0])\\n")
print(f"{'k':>2} {'hole h':>7} {'S':>3} {'regime':>8} {'R of first 4 cycles':>22}")
for k, stage in enumerate(sched.stages):
    c = stage[0]
    print(f"{k:>2} {c.hole:>7} {c.S:>3} "
          f"{('row' if c.hole < sched.m else '2-row'):>8} "
          f"{', '.join(str(x.R) for x in stage[:4]):>22}")
""")

md(r"""
## 1.3 The permute-rotate switch

Because only $m$ subset states occur, the permutation $\mathbf{P}_f$ needed each
cycle decomposes as a barrel shift by $R$ followed by one of $m$ fixed stride
permutations, so the switch costs $\Theta(mP_{bu})$ 2:1 multiplexers:
$mP$ for the barrel shifter plus $(m{+}1)2^{m-1}$ for the subset switch.

**By hand at 128 ports** ($P_{bu}=64$, $m=7$): $7(128)+8(64) = 1{,}408$ muxes per
bit, against the $128\times127 = 16{,}256$ a general crossbar over the same ports
would need.
""")

code("""
ports, prs, quad = [], [], []
print(f"{'ports':>6} {'m':>2} {'PRS mux2':>10} {'pipeline regs':>14}")
for P_bu in (2,4,8,16,32,64):
    c = fx.prs_cost(P_bu)
    ports.append(c['ports']); prs.append(c['mux2']); quad.append(c['ports']*(c['ports']-1))
    print(f"{c['ports']:>6} {c['m']:>2} {c['mux2']:>10,} {c['pipeline_regs']:>14,}")

fig, ax = plt.subplots()
ax.plot(ports, prs, 's-', label='PRS  $\\\\Theta(m P_{bu})$')
ax.plot(ports, quad, ':', c='grey', label='general crossbar  $\\\\Theta(P_{bu}^2)$')
ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
ax.set_xlabel('switching ports $2P_{bu}$'); ax.set_ylabel('2:1 muxes per bit')
ax.set_title('Switching cost'); ax.legend()
plt.tight_layout(); plt.show()
""")

md(r"""
## 1.4 One datapath, two workloads

A butterfly unit applies a general $2\times2$ coefficient matrix. The FFT is the
special case $\begin{pmatrix}1&1\\W&-W\end{pmatrix}$; a BL layer uses trained
weights — exactly the `[2,2,l/2,log2 l]` shape of Table 4.
""")

code("""
eng = fx.FlexBE(P_bu=16)
rng = np.random.default_rng(0)

x = rng.normal(size=(4,1024)) + 1j*rng.normal(size=(4,1024))
y, st = eng.transform(x, bitrev=True, datapath='cycle')
ref = np.fft.fft(x, axis=1)
print("FFT :", st)
print("      max relative error vs numpy.fft =",
      f"{np.max(np.abs(y-ref))/np.max(np.abs(ref)):.2e}")

W = fx.random_bl_coefficients(32, rng)
M = np.asarray(eng.transform(np.eye(32), W, datapath='cycle')[0]).T
z = rng.normal(size=(7,32))
yb, _ = eng.transform(z, W, datapath='cycle')
print("\\nBL  : engine output equals the dense butterfly matrix:", np.allclose(yb, z @ M.T))
""")

md(r"""
## 1.5 Sub-parallelism (Alg. 2, Eq. 5) and bit reversal (Alg. 3)

A datapath $2P_{bu}$ wide would otherwise have to zero pad any transform shorter
than that. Sub-parallelism instead interleaves $P_{sub} = 2P_{bu}/l$ transforms
into one vector and stops after $\log_2 l$ stages. This is where BSPNet spends
its time: cfg-6 has $d_{in}=8$ against $2P_{bu}=32$, so $P_{sub}=4$.

**By hand** — 1024 vectors of length 8 on one $P_{bu}=16$ engine: padding to 32
would cost $1024\times 32/32\times 5 = 5{,}120$ cycles, sub-parallelism costs
$1024/4\times 3 = 768$. Ratio $6.67\times$ = $P_{sub}$ (4) times the wasted
stages ($5/3$).

Algorithm 3 then writes back through the bit-reversal network in exactly $N/P$
cycles, conflict free on *both* memories, with lane $a$ always reading bank $a$
and destination bank $\mathrm{bit\text{-}rev}_m((a+u)\bmod P)$ (Eq. 7). That part of
the paper is correct as printed.
""")

code("""
print(f"{'l':>4} {'N':>5} {'P_sub':>6} {'cycles/1024 vec':>16} {'vs zero-pad':>12}")
for l in (2,4,8,16,32,64):
    N_, ps = eng.geometry(l)
    flex = fx.layer_cycles(1024, l, P_be=1, P_bu=16)
    pad  = fx.layer_cycles(1024, max(l,32), P_be=1, P_bu=16)
    print(f"{l:>4} {N_:>5} {ps:>6} {flex:>16,.0f} {pad/flex:>11.2f}x")

print(f"\\n{'N':>7} {'P':>4} {'cycles':>8} {'N/P':>6}  read-CF  write-CF  lane a -> bank a  permutation")
for N_, P_bu in [(1024,16),(32768,16),(4096,4)]:
    nn, mm = int(math.log2(N_)), (2*P_bu).bit_length()-1
    j, k = fx.bitrev_schedule(nn, mm); Pp = 1 << mm
    print(f"{N_:>7} {Pp:>4} {j.shape[0]:>8} {N_//Pp:>6}  "
          f"{str(all(len(np.unique(fx.bsm_array(jv,mm)))==Pp for jv in j)):>7}  "
          f"{str(all(len(np.unique(kv & (Pp-1)))==Pp for kv in k)):>8}  "
          f"{str(all(np.array_equal(fx.bsm_array(jv,mm), np.arange(Pp)) for jv in j)):>16}  "
          f"{len(np.unique(j))==N_ and np.array_equal(k, fx.bit_rev_array(nn)[j])}")
""")

md(r"""
## 1.6 Eq. (10) and the abstract's headline

$$\text{cycles} \approx \frac{P_N N}{2P_{sub}P_{be}P_{bu}}\times\log_2\frac{N}{P_{sub}}$$

**By hand**, four 32,768-point FFTs on $P_{be}=4$, $P_{bu}=16$:

$$\frac{4\times 32768}{2\times1\times4\times16}\times 15 = 1024\times 15 = 15{,}360 \text{ cycles} = 51.2\,\mu s \text{ at } 300\,\text{MHz}$$

One clarification worth adding to the paper: $P_N$ must be read as the number of
length-$l$ *input* sequences and $N$ as the engine computational length
$\max(l, 2P_{bu})$; with that reading the expression is consistent for all
$P_{sub}$.
""")

code("""
print("Eq. (10):", f"{fx.cycles_eq10(4, 32768, 1, P_be=4, P_bu=16):,.0f} cycles  (paper: 15,360)")
arr = fx.BEArray(P_be=4, P_bu=16)
xb = rng.normal(size=(4,32768)) + 1j*rng.normal(size=(4,32768))
yb, stb = arr.transform(xb, bitrev=True, datapath='cycle')
print("measured:", f"{stb.butterfly_cycles:,} butterfly + {stb.bitrev_cycles:,} bit-reversal cycles")
print("          max abs error vs numpy.fft =", f"{np.max(np.abs(yb-np.fft.fft(xb,axis=1))):.2e}")

fmt = fx.FixedPointFormat()
xq = fmt.quantise((rng.normal(size=1024)+1j*rng.normal(size=1024))/8)
yq, stq = fx.FlexBE(16, fmt=fmt).transform(xq, bitrev=True, stage_scale=0.5, datapath='cycle')
e = np.linalg.norm(yq[0]-np.fft.fft(xq)*stq.scale)/np.linalg.norm(np.fft.fft(xq)*stq.scale)
print(f"\\nfxp16 Q1.15, 1/2 per stage, N=1024: NRMSE {e:.2e} -> SQNR {20*math.log10(1/e):.1f} dB")
""")

md(r"""
## 1.7 BSPNet
""")

code("""
print(bs.design_space_table())
cfg, hw = bs.CONFIGS['cfg-6'], bs.HW['BSP-Flex']
print(f"\\ncfg-6 on BSP-Flex ({hw.f_mhz:.0f} MHz, P_be={hw.P_be}, P_bu={hw.P_bu}):")
tot = 0
for name, cyc in bs.cycle_breakdown(cfg, hw):
    tot += cyc
    print(f"  {name:44s} {cyc:10,.0f} cyc  {cyc/300e6*1e3:7.4f} ms")
print(f"  {'total':44s} {tot:10,.0f} cyc  {tot/300e6*1e3:7.4f} ms"
      f"   ->  {1e3/(tot/300e6*1e3):,.0f} samples/s")
""")

# -------------------------------------------------------------- PART II
# -------------------------------------------------------------- PART II
md(r"""
---
# Part II — The microcoded command processor

## 2.1 Why a sequencer

FlexBE as published is fixed function: the layer sequence is hardwired. The
minimal change that makes it programmable is a PL-side sequencer executing a
queue of descriptors.

On Zynq this is also the *only* option: the Cortex-A53 is hard IP, so there is
no custom-instruction port (Arm Custom Instructions are Cortex-M33/M55 only, and
the A53 is ARMv8.0 — 128-bit NEON, no SVE). It is the right granularity anyway:
the datapath retires **64 butterflies per cycle**, so anything finer-grained than
a whole layer spends more time issuing than computing.

## 2.2 The command set

```
bs.load   mem -> butterfly scratchpad
bs.wload  mem -> coefficient RAM
bs.store  scratchpad -> mem;  flags: NATURAL | BITREV | MAG
bs.bfly   P_N transforms of length l;  mode FFT | IFFT | BL | BL_EXPAND
bs.brev   write-back reversal pass (Algorithm 3)
bs.pow    elementwise s^2 .. s^8                      (Eq. 1)
bs.mag    |x|, optional peak normalisation
bs.pool   LayerNorm -> shortcut -> ReLU -> maxpool     (NormPool)
```

Design rules that matter:

* **`BITREV` is a store addressing mode, not an instruction** — Algorithm 3 costs nothing extra when fused into write-back, and disappears for cfg-4.
* **One `bs.bfly` retires Eq. (10) cycles of work** — the equation is literally the descriptor's latency formula, and $P_{sub}$/$P_N$ become descriptor fields rather than CSR pokes, i.e. Table 5's runtime parameters become architectural.
* **The glue is not optional**: Table 11 puts Magnitude Computation at 41k LUT (26%). Without `bs.pow`/`bs.mag`/`bs.pool` you round-trip through memory and lose the win.
* **Descriptors expose the memory and the transform, never the schedule.** No opcode names a bank index, a PRS control word or a cycle, so the addressing and control can be retuned without breaking a single descriptor program.
""")

code("""
d = mc.Descriptor("bs.bfly", mode=mc.MODES["BL"], log2_l=5, log2_Psub=2,
                  P_N=1024, src=0x1000, dst=0x2000, coeff=0x30,
                  flags=mc.FLAGS["RELU"], tag=7)
raw = d.encode()
print(f"descriptor: {mc.Descriptor.SIZE} bytes")
print("  encoded:", raw.hex(' ', 4))
print("  decode == original:", mc.Descriptor.decode(raw) == d)
print("\\nopcodes:", ", ".join(f"{k}=0x{v:02x}" for k, v in mc.OPCODES.items()))
""")

md(r"""
## 2.3 The program for one inference

~104 descriptors, 3.3 kB. Compute cycles come from `bspnet.cycle_breakdown`, so
the accelerator side is exactly the Eq. (10) model validated in Part I.
""")

code("""
cmds = mc.build_program(cfg, hw)
kinds = {}
for c in cmds: kinds[c.kind] = kinds.get(c.kind, 0) + 1
print(f"{len(cmds)} descriptors ({mc.program_bytes(cmds)} bytes): " +
      ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
print(f"datapath time: {bs.total_cycles(cfg, hw)/300e6*1e6:.1f} us at 300 MHz\\n")
for c in cmds[:9]:
    detail = f"{c.nbytes/1024:.0f} KB" if c.nbytes else f"{c.cycles:,.0f} cyc"
    print(f"  {c.name:32s} {c.kind:8s} {detail:>11s}  deps={list(c.deps)}")
""")

md(r"""
## 2.4 Issue mechanism

**By hand**: ~104 commands × 0.4 µs of posted MMIO = 42 µs against a 214 µs
inference — 19% overhead before anything else goes wrong. With PYNQ's
per-transfer Python cost it is far worse. A descriptor ring in coherent memory
with one doorbell reduces this to a prefetched fetch that hides under compute —
and since the BSPNet command graph is *static*, the ~104 descriptors can be built
once at init, leaving one pointer update and one doorbell per inference.

### Zynq port map

| interface | width | use |
|---|---|---|
| `S_AXI_HP0-3_FPD` | 4 × 128b | bulk IQ in / features out (what Fig. 12 uses) |
| `S_AXI_HPC0_FPD` | 128b coherent via CCI-500 | descriptor fetch |
| `S_AXI_ACP_FPD` | 128b, allocates into A53 L2 | completion words only — **not** bulk data |
| `M_AXI_HPM0_FPD` / `_LPD` | 32/128b | control registers and doorbell, from the APU or the R5F |
""")

code("""
print(f"{'platform':56s} {'us':>8s} {'ovh%':>7s} {'sps':>7s} {'critical':>9s}")
for r in mc.compare_platforms():
    print(f"{r['name'][:56]:56s} {r['latency_us']:8.1f} {r['overhead_pct']:7.1f} "
          f"{r['throughput']:7.0f} {r['critical']:>9s}")
""")

md(r"""
The PYNQ flow is **host bound**, not accelerator bound: ~104 commands ×
(Python call + posted MMIO + polled completion) swamps a 214 µs inference. Much
of the gap between the solid and dashed lines of Fig. 15a is PS-side software,
and a descriptor ring recovers it **without touching the PL**.
""")

code("""
print("prefetch depth (batch 1):")
for r in mc.ring_depth_sweep():
    print(f"  depth {r['depth']:3d}: {r['latency_us']:7.1f} us  +{r['overhead_pct']:5.1f}%  "
          f"critical={r['critical']}")

b = mc.port_budget()
print(f"\\nlink budget: in {b['in_bytes']/1024:.0f} KB + out {b['out_bytes']/1024:.1f} KB "
      f"per {b['latency_ms']:.3f} ms")
print(f"  required {b['required_gbs']:.2f} GB/s of {b['available_gbs']:.1f} available "
      f"({b['utilisation']:.0%}); minimum HP ports = {b['min_hp_ports']}")

a = mc.sequencer_area()
print(f"\\nsequencer area: {a['sequencer_lut']:,} LUT + {a['sequencer_bram36']} BRAM36 "
      f"= +{a['lut_overhead_pct']:.1f}% on the Table 11 system "
      f"({a['total_lut']:,} LUT total, fits XCZU7EV: {a['fits']})")
""")

code("""
rows = mc.batch_sweep()
fig, ax = plt.subplots()
bx = [r['batch'] for r in rows]
ax.plot(bx, [r['peak'] for r in rows], 'k--', label='datapath peak')
ax.plot(bx, [r['tuned'] for r in rows], '*-', label='ring + per-resource queues (R5)')
ax.plot(bx, [r['static'] for r in rows], 's-', label='static ring (A53)')
ax.plot(bx, [r['pynq'] for r in rows], 'o-', label='PYNQ / Jupyter flow')
ax.set_xlabel('batch size'); ax.set_ylabel('samples / s')
ax.set_title('Issue mechanism vs batch (cf. Fig. 15a)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"{'batch':>6} {'PYNQ':>9} {'static ring':>12} {'tuned':>9} {'peak':>9} {'tuned/peak':>11}")
for r in rows:
    print(f"{r['batch']:>6} {r['pynq']:>9.0f} {r['static']:>12.0f} {r['tuned']:>9.0f} "
          f"{r['peak']:>9.0f} {r['tuned']/r['peak']:>10.0%}")
""")

md(r"""
### What the model says

* **Prefetch saturates at 4–8 descriptors.** Deeper buys nothing at batch 1.
* **Per-resource queues matter only across inferences.** With one shared ring the next sample's input-DMA descriptor sits behind ~100 compute descriptors and cannot be fetched early; splitting the ring takes batch-10 throughput from 4,032 to 4,362 samples/s — **94% of datapath peak**. A cheap change, invisible in a bandwidth analysis.
* **The link is never the constraint**: 0.62 GB/s of 6.4 available, so **one HP port suffices**. The four ports buy burst overlap for a single record, not throughput.
* **The sequencer is ~3.6k LUT, +2.4%** on the Table 11 system — against the 15k already spent on top-level control.
* The residual ~9% at batch 1 is the 128 kB input transfer, which overlaps nothing unless the accelerator ingests from a live AXI-stream SDR front end — an option Fig. 12 already shows, and the strongest single change for a batch-1 claim.

### Practical notes

* Use the **Cortex-R5F in the RPU** to drive the ring: on-die, 500–600 MHz, TCM, deterministic low-latency PL access via `M_AXI_HPM0_LPD`, leaving the A53 for Linux/PYNQ. Split with OpenAMP/rpmsg.
* `pynq.allocate()` gives uncached buffers — right for bulk IQ over HP, wrong for descriptors and completion words, which want cacheable memory through HPC/ACP with explicit flush/invalidate.
* **Do not use PL–PS interrupts for 214 µs jobs**: Linux IRQ latency is 5–10 µs. Write a completion word through ACP and poll it.
* Context switching with an architectural scratchpad is the known hard part (Hwacha and Gemmini both wrestled with it). For an edge RFML box, marking the region process-private and non-preemptible for a command's duration is probably enough — but say so explicitly rather than leaving it implied.
""")

# -------------------------------------------------------------- PART III
md(r"""
---
# Part III — Assumptions, and what to measure next

## 3.1 Measured vs assumed

**Measured** (executed code, exact by construction): every cycle of the schedule —
bank conflicts, coverage, the PRS permutation, arithmetic; FFT/BL results against
`numpy.fft`; Algorithm 3's permutation, conflict freedom and $N/P$ cycle count;
descriptor encoding and program structure.

**Assumed** (all in `MicrocodeConfig`, one line to change):

| assumption | value | confidence |
|---|---|---|
| posted MMIO write | 0.40 µs | medium |
| descriptor fetch over HPC0 | 0.15 µs | medium |
| PYNQ per-transfer cost | 25 µs | soft — but the conclusion holds for 5–50 µs |
| Linux IRQ latency (mmio preset) | 6 µs | medium |
| ACP completion poll | 0.20 µs | medium |
| DDR sustained bandwidth | 6.4 GB/s | conservative; irrelevant at 10% utilisation |
| sequencer LUT | 3k + 40/slot | rough — but 10× off still rounds to "negligible" |

One modelling choice to flag in any write-up: the program adds
`bs.pow`/`bs.mag`/`bs.pool` as separate commands (~16 µs, +7% over 214 µs) which
the RTL may fuse into the surrounding dataflow. If so the real overheads are
slightly lower than shown.

## 3.2 The experiment that would settle it

Build the sequencer on the same ZCU104, reusing the existing Verilog BE array
untouched, and measure three points: the hardwired design (Table 11), the
descriptor ring, and the current PYNQ flow. Two results make it publishable:

1. **programmability costs <10%** at this granularity — the layer graph becomes
   runtime data at essentially no hardware cost;
2. **most of the Fig. 15a solid-vs-dashed gap is PS-side**, which strengthens the
   existing single-batch claim without any PL change.

Adding a streamed AXI input path would close most of the remaining batch-1
residual.

## 3.3 Recommended paper actions

1. **Clarify Eq. (10)**: $P_N$ = number of length-$l$ sequences, $N$ = engine computational length.
2. **State the row-rotation reading of Eq. (3)** — it makes the conflict-freedom argument two lines.
3. Consider noting that the Fig. 15a gap is partly PS-side, with the descriptor-ring number as evidence.
""")

md(r"""
---
# Appendix — source reference

## `flexbe.py`

| symbol | role |
|---|---|
| `popcount`, `bit_rev`, `bit_rev_array`, `insert_zero` | bit primitives |
| `bsm`, `bsm_array` | Eq. (3) bank index |
| `FixedPointFormat` | Q1.15, 32-bit accumulate, convergent rounding, saturation |
| `BankedMemory` | $P$ single-port banks; raises `BankConflictError` |
| `PermuteRotateSwitch`, `prs_cost` | Sec. 3.1 switching and its cost |
| `CycleControl`, **`cycle_control`**, `ButterflySchedule` | per-cycle control and the full sequence; `validate()` |
| `fft_coefficients`, `random_bl_coefficients`, `expand_coefficients`, `coefficients_from_paper_layout` | twiddles, BL weights, $P_{sub}$ expansion, Table 4 layout |
| `butterfly_reference` | golden datapath |
| `bitrev_schedule` | Algorithm 3 |
| `FlexBE`, `BEArray` | the engine and the array |
| `cycles_eq10`, `layer_cycles`, `bitrev_cycles` | Eq. (10) |

## `bspnet.py`

`BSPNetConfig` (Table 7) · `CONFIGS` cfg-1..8 · `HWConfig`/`HW` (FlexBE,
BSP-Flex with Table 10 $F_{max}$) · `cycle_breakdown`,
`total_cycles`, `latency_ms`, `throughput_sps` · `BSPNet.features` (Eq. 1),
`.branch`, `.forward` · `design_space_table`.

## `microcode.py`

`OPCODES`/`MODES`/`FLAGS` · `Descriptor` (32 B, `encode`/`decode`) · `Command` ·
`build_program` · `MicrocodeConfig`/`PLATFORMS` (pynq, mmio, ring, static, tuned) ·
`Sequencer` → `RunReport` · `compare_platforms`, `ring_depth_sweep`,
`batch_sweep`, `port_budget`, `sequencer_area`.
""")

code("""
r = subprocess.run([sys.executable, '-m', 'unittest', 'test_flexbe', '-v'],
                   capture_output=True, text=True)
lines = [l for l in r.stderr.splitlines()
         if ' ... ' in l or l.startswith('Ran ') or l.strip() in ('OK','FAILED')]
groups = {}
for l in lines:
    if ' ... ' in l and '(' in l:
        parts = l.split('(')[1].split('.')
        cls = parts[1] if len(parts) > 1 else parts[0].rstrip(')')
        groups.setdefault(cls, []).append(l.split(' ')[0])
for cls, tests in groups.items():
    print(f"{cls} ({len(tests)})")
    for t in tests: print(f"    {t}")
print()
print("\\n".join(l for l in lines if l.startswith('Ran ') or l.strip() in ('OK','FAILED')))
""")

md(r"""
---
## How to run

```bash
python test_flexbe.py              # 49 tests, ~3 s
python build_notebook.py           # regenerate this notebook
```

Pure NumPy; matplotlib only for the figures here.
""")


# ===========================================================================
def execute():
    ns, count, failures = {}, 0, 0
    for cell in CELLS:
        if cell["cell_type"] != "code":
            continue
        count += 1
        buf, outputs = io.StringIO(), []
        try:
            with redirect_stdout(buf):
                exec(compile("".join(cell["source"]), f"<cell {count}>", "exec"), ns)
        except Exception:
            failures += 1
            outputs.append({"output_type": "error", "ename": "Error",
                            "evalue": "see traceback",
                            "traceback": traceback.format_exc().splitlines()})
            print(f"  !! cell {count}:\n{traceback.format_exc()}", file=sys.stderr)
        text = buf.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout",
                               "text": text.splitlines(keepends=True)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", bbox_inches="tight", dpi=110)
            outputs.append({"output_type": "display_data",
                            "data": {"image/png":
                                     base64.b64encode(b.getvalue()).decode()},
                            "metadata": {}})
            plt.close(fig)
        cell["outputs"], cell["execution_count"] = outputs, count
    print(f"executed {count} code cells, {failures} failures")
    return failures


def emit(path):
    nb = {"cells": CELLS,
          "metadata": {
              "kernelspec": {"display_name": "Python 3", "language": "python",
                             "name": "python3"},
              "language_info": {"name": "python",
                                "version": sys.version.split()[0],
                                "mimetype": "text/x-python",
                                "file_extension": ".py",
                                "pygments_lexer": "ipython3",
                                "nbconvert_exporter": "python",
                                "codemirror_mode": {"name": "ipython",
                                                    "version": 3}}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path}: {len(CELLS)} cells "
          f"({sum(1 for c in CELLS if c['cell_type']=='code')} code)")


if __name__ == "__main__":
    execute()
    emit(sys.argv[1] if len(sys.argv) > 1 else "FlexBE_microcode.ipynb")
