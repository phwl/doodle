# FlexBE / BSPNet simulator

A NumPy simulator of the flexible butterfly engine and its BSPNet application
layer, written against the manuscript *"A Flexible FPGA-based Butterfly Engine
for Accelerating Signal Processing and Machine Learning"* (Liu, Wu, Leong).

It is a **verification model**, not a performance model of the RTL: it executes
the real dataflow (banked memory, switching network, butterfly units, control
sequence) cycle by cycle, checks the architectural invariants the paper claims,
and reports cycle counts that can be compared against Eq. (10) and the RTL.

```
FlexBE_analysis.ipynb executed notebook: the complete analysis, all calculations
flexbe.py             engine: memory, switches, schedule, bit reversal, arithmetic
bspnet.py             application: Eq. (1) features, BL branches, Table 7 configs
test_flexbe.py        60 unit tests (the "test program")
demo_flexbe.py        report reproducing the paper's headline numbers
isa_model.py          instruction-level backend: microcoded / RoCC / RVV options
demo_isa.py           option comparison, instruction mix, VLEN and LSU sweeps
zynq_model.py         PS-PL attachment: descriptor ring, DMA overlap, completion
demo_zynq.py          issue-mechanism comparison, prefetch depth, batch scaling
check_algorithm1.py   diagnosis of Algorithm 1 lines 5-6 + verification of the fix
docs/                 algorithm1_fix.md (proof), algorithm1_fixed.tex (listing)
```

## Quick start

```bash
python test_flexbe.py          # full verification suite, ~2 s
python demo_flexbe.py          # report; add --full for the real 4 x 32768 FFT
python check_algorithm1.py
```

```python
import numpy as np, flexbe as fx

eng = fx.FlexBE(P_bu=16)                       # 32 lanes, m = 5
x   = np.random.randn(4, 4096) + 1j*np.random.randn(4, 4096)

y, st = eng.transform(x, bitrev=True)          # 4 packed FFTs, natural order out
print(st)          # N=4096 l=4096 P_sub=1 P_N=4 stages=12 bfly_cycles=6144 ...
assert np.allclose(y, np.fft.fft(x, axis=1))

# same datapath, trained weights instead of twiddles
W = fx.random_bl_coefficients(32, np.random.default_rng(0))
h, _ = eng.transform(np.random.randn(1024, 32), W)      # P_sub = 1

# transforms shorter than 2*P_bu: sub-parallelism, Alg. 2
s, st = eng.transform(np.random.randn(4096, 8))         # st.P_sub == 4
```

Every call runs the full banked-memory + PRS datapath (`datapath="cycle"`).
`datapath="vector"` swaps in a golden datapath while still modelling cycles —
useful for the large BSPNet layers.

## What is modelled, and where it comes from

| Paper | Simulator |
|---|---|
| Eq. (3) shift-down storage `a_x = I>>m`, `a_y = bsm(I)` | `bsm`, `BankedMemory` |
| Eq. (2) read / permute / butterfly / restore / write | `FlexBE._run_cycles` |
| Sec. 3.1 PRS = barrel shifter + subset switch, `P_f = P_s x P_r` | `PermuteRotateSwitch`, `FullyConnectedSwitch` |
| Eq. (4), Fig. 9 interconnect cost | `interconnect_cost` |
| Alg. 1 control law `R^i = bsm(I^i[0])`, `S^i` | `ButterflySchedule`, `paper_subset_state` |
| Sec. 3.2 / Alg. 2 sub-parallelism, Eq. (5) interleaving | `FlexBE.geometry`, `interleave`, `expand_coefficients` |
| Sec. 3.3 / Alg. 3 bit-reversal write-back, Eqs. (6)–(9) | `bitrev_schedule`, `FlexBE._bitrev` |
| Sec. 3.4 `P_N` packing into one RAM array (Fig. 5) | `FlexBE.transform` (`x` with several rows) |
| Eq. (10) cycle model | `cycles_eq10`, `layer_cycles` |
| Sec. 5.2 Q1.15, 32-bit accumulate, convergent rounding, saturation | `FixedPointFormat` |
| Sec. 4, Eq. (1), Table 4, Table 7 | `bspnet.BSPNet`, `CONFIGS` |
| Table 10 `F_max` / Fig. 11 Pareto | `bspnet.HW`, `design_space_table` |

**Stage convention.** The engine runs a decimation-in-frequency dataflow, so
stage `k` pairs indices differing in bit `h = n-1-k`, natural-order input gives
bit-reversed output (hence the write-back path of Sec. 3.3), and the subset
switch state falls out as

```
S^i = h  when h < m,   S^i = 0 otherwise
```

which is *identical* to Algorithm 1 lines 8–11 (`S^i = n-k-1` for
`n-m <= k <= n-2`, else 0). Two regimes give conflict freedom:

* `h < m` — both elements of a butterfly sit in one RAM row; a cycle reads a
  whole row, so the banks `(popcount(H) + low) mod 2^m` are all distinct and the
  slot order is the stride permutation `pi_h` → `S = h`;
* `h >= m` — the partner sits in row `H + 2^(h-m)`, whose popcount is one
  greater; a cycle takes the `P_bu` even (or odd) low values from both rows, so
  the banks are `(popcount(H) + e + j) mod 2^m` → a pure rotation, `S = 0`.

## Verification

`test_flexbe.py` asserts, rather than assumes:

* every cycle of every stage is bank-conflict free, each stage covers each
  element exactly once, and the PRS pair `(R^i, S^i)` reproduces the FCS
  crossbar `P_f` cycle by cycle (`TestSchedule`);
* the subset switch uses at most `m` distinct states, and `algorithm1_fixed`
  reproduces the validated schedule cycle for cycle (`TestAlgorithm1Fix`);
* exact FFT/IFFT against `numpy.fft` for `N` = 32…4096 and `P_bu` = 2…16, with
  and without the bit-reversal write-back, and with `P_N` sequences packed into
  one RAM array;
* Algorithm 3 is a correct global bit-reversal, conflict free on *both*
  memories, lane `a` always reading bank `a`, in exactly `N/P` cycles, with the
  Eq. (7) destination-bank law `bit-rev_m((a+u) mod P)`;
* sub-parallelism computes `2*P_bu/l` independent short FFTs per pass, and the
  Eq. (5) interleaving round-trips;
* a BL layer equals the corresponding dense matrix and the golden model;
* fxp16 Q1.15 with 1/2-per-stage scaling gives > 40 dB SQNR at `N` = 1024;
* measured cycles equal Eq. (10), including
  **`cycles_eq10(4, 32768, 1, P_be=4, P_bu=16) == 15360`** — the abstract's
  "four 32k-point FFTs in approximately 15,360 clock cycles";
* BSPNet cfg-6 on BSP-Flex lands at 64,320 cycles = **0.214 ms at 300 MHz**
  (≈ 4.7k samples/s), consistent with Fig. 11 and the peak-throughput line of
  Fig. 15a, and FlexBE beats both BE-base baselines on all eight configs.

`demo_flexbe.py` prints all of the above as a report, including the module
breakdown for cfg-6 and the FCS/PRS cost table.

## Algorithm 1, lines 5–6: diagnosis and fix

The control law in lines 7–11 is exactly what a conflict-free shift-down
schedule needs. The index formula in lines 5–6,

```
ii     <- rotate_{n-1}(base, k+1)
I^i[0] <- rotate_n(2*ii, k)
```

read literally, does **not** select one representative per conflict-free cycle.
`check_algorithm1.py` tries both rotation amounts (`k+1` as printed and `k`) and
both stage orders (`h = n-1-k` and `h = k`), and:

* against a validated schedule it fails at the first stage in every size tested
  (`N` = 8…4096, `P_bu` = 2…16);
* for `N` = 8, 16, 32 with `P_bu` = 2 it enumerates *all* conflict-free
  groupings of a stage and finds that none admits the `I^i[0]` values as
  distinct cycle representatives — for `N` = 8 the only conflict-free grouping
  is row-based `{0,1,2,3} / {4,5,6,7}`, while the formula yields representatives
  `0` and `2`, which share a row.

Rewriting the printed expression as `ins(rotate_{n-1}(base, 2k+1), k)` (where
`ins` inserts a zero bit) suggests the origin: the operand is rotated twice by
roughly the stage index, and the hole lands at bit `k` instead of `n-1-k`.

**The fix.** Only lines 5–6 change; lines 7–11 stay verbatim. With
`ins(x,p) = ((x >> p) << (p+1)) | (x mod 2^p)`:

```
h <- n-1-k
if h < m:  H <- j                      I^i[2t] <- (H << m) + ins(t, h)     S^i <- h
else:      e <- j mod 2
           H <- ins(j >> 1, h-m)       I^i[2t] <- (H << m) + 2t + e        S^i <- 0
both:      I^i[2t+1] <- I^i[2t] + 2^h  R^i <- bsm(I^i[0])
```

`S^i = h` for `h < m` is identical to the published `S^i = n-k-1` on
`(n-m) <= k <= (n-2)`, and the revised listing additionally specifies the whole
vector `I^i[0..2P_bu-1]`, which the published version delegates to "a specific
circuit" of [11].

It is conflict free by construction: for `h < m` a cycle reads a whole RAM row,
whose banks `(popcount(H) + low) mod 2^m` are all distinct; for `h >= m` bit
`h-m` of `H` is zero, so `popcount(H + 2^(h-m)) = popcount(H) + 1` and slot `j`
reads bank `(popcount(H) + e + j) mod 2^m`, a pure rotation. Each branch emits
exactly `N/(2*P_bu)` cycles per stage and covers every index once.

`flexbe.algorithm1_fixed(n, P_bu, k, j)` is a direct transcription;
`TestAlgorithm1Fix` verifies conflict freedom, coverage, pairing and lines 7–11
straight from the formula for `N` = 8…4096, `P_bu` = 1…16, and runs an engine
whose control comes *only* from the revised listing against `numpy.fft`.
`docs/algorithm1_fix.md` has the full argument and a note on hardware cost;
`docs/algorithm1_fixed.tex` is a drop-in `algorithm2e` listing.

## Programmability study (`isa_model.py`)

Four ways to make the workload software programmable, all carrying the same 64
radix-2 butterfly lanes, measured on cfg-6 single-batch inference:

| option | cycles | MHz | ms | x hw | LUT | fits | bottleneck |
|---|---|---|---|---|---|---|---|
| hardwired (paper) | 64,320 | 300 | 0.214 | 1.00 | 152,553 | yes | - |
| A microcoded command processor | 64,592 | 300 | 0.215 | 1.00 | 160,553 | yes | datapath |
| B RoCC custom instruction | 64,524 | 285 | 0.226 | 1.06 | 172,553 | yes | datapath |
| C RVV + Zvbfly/Zvshfl/Zvtwid | 238,720 | 250 | 0.955 | 4.45 | 119,342 | yes | LSU |
| C* as C with 512 B/cycle LSU | 110,974 | 250 | 0.444 | 2.07 | 137,774 | yes | ALU |
| D RVV 1.0 (vrgather) | 1,277,296 | 250 | 5.109 | 23.8 | 356,910 | **no** | permute |

The instruction streams for C and D are *generated and executed* - `run_program`
implements the semantics of every proposed instruction and
`TestISAModel.test_generated_program_computes_the_transform` checks the result
against `numpy.fft` before any cycle is counted - so the instruction counts are
exact and only the per-class rates, issue width and clock are assumptions
(all in `MachineConfig`). Area is calibrated to Table 10 (962 LUT, 10 DSP per BU)
and the permute networks use the paper's own Eq. (4) cost model.

Three results fall out:

* **the vector options are operand-bandwidth bound, not compute bound.** VLEN
  saturates by VL = 64; what matters is load-store bandwidth, and closing the
  gap needs ~512 B/cycle, i.e. a wide banked on-chip scratchpad next to the
  lanes - which is option B with extra steps.
* **fusing the permute into the butterfly matters more than adding a permute
  instruction.** `vbfly.vv` with a stride field (PRS inside the functional unit)
  removes every permute instruction and is 1.4x faster than a separate
  `vshfl` + butterfly + `vunshfl`.
* **a full `vrgather` crossbar does not fit XCZU7EV at matched width.** By the
  paper's own Theta(P^2) vs Theta(m*P) argument the gather network is >8x the
  area of the shuffle network, which is what makes option D infeasible rather
  than merely slow.

## Zynq attachment (`zynq_model.py`)

On Zynq the A53 is hard IP, so there is no custom-instruction port: the
programmable option becomes a PL-side command sequencer fed by descriptors.
This model covers issue, DMA/compute overlap and completion for the ~104-command
cfg-6 stream (`bs.load` / `bs.pow` / `bs.bfly` / `bs.mag` / `bs.pool` /
`bs.store`), with accelerator cycles taken from `bspnet.cycle_breakdown`.

| issue mechanism | us | overhead | samples/s | critical |
|---|---|---|---|---|
| PYNQ / Jupyter flow | 878.3 | +281% | 1,139 | **host** |
| MMIO per command, Linux | 258.1 | +12.0% | 3,874 | compute |
| static descriptor ring, A53 | 252.5 | +9.6% | 3,961 | compute |
| static ring + per-resource queues + streamed input (R5) | 250.3 | +8.7% | 3,995 | compute |

* the PYNQ flow is **host bound**, not accelerator bound - most of the gap
  between the solid and dashed lines of Fig. 15a is PS-side software;
* a descriptor ring only needs **4-8 descriptors** of prefetch; deeper buys
  nothing;
* **per-resource queues matter only across inferences**: with a single shared
  ring, the next sample's input-DMA descriptor sits behind ~100 compute
  descriptors and cannot start early.  Splitting the ring takes batch-10
  throughput from 3,981 to 4,300 samples/s, i.e. 92% of the datapath peak;
* the PS-PL link is never the constraint: 0.62 GB/s of 6.4 GB/s available, so
  **one HP port suffices** - the four ports buy burst overlap for a single
  record, not throughput;
* the residual ~9% at batch 1 is the 128 KB input transfer, which overlaps
  nothing unless the accelerator ingests from a live AXI-stream SDR front end.

## Extending

* **Trained weights.** `coefficients_from_paper_layout` converts the
  `[2, 2, l/2, log2 l]` layout of Table 4 into the internal
  `(log2 l, l/2, 2, 2)` form, so exported PyTorch BL weights drop straight in.
* **Other engines.** `BEArray(P_be, P_bu)` spreads sequences over engines and
  reports the slowest one; `bspnet.HW` holds the BE-base-1 / BE-base-2 /
  FlexBE / BSP-Flex parameter sets, with `zero_pad_short=True` modelling the
  BE-base minimum-transform-length limitation.
* **RTL cross-check.** `ButterflySchedule.all_cycles()` yields `(I^i, Addr^i,
  R^i, S^i, coefficient addresses)` per cycle — a ready-made stimulus/reference
  stream for a UVM or cocotb testbench against the Verilog core.
