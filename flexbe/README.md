# FlexBE / BSPNet simulator

A NumPy simulator of the flexible butterfly engine and its BSPNet application
layer, written against the manuscript *"A Flexible FPGA-based Butterfly Engine
for Accelerating Signal Processing and Machine Learning"* (Liu, Wu, Leong).

It is a **verification model**, not a performance model of the RTL: it executes
the real dataflow (banked memory, switching network, butterfly units, control
sequence) cycle by cycle, checks the architectural invariants the paper claims,
and reports cycle counts that can be compared against Eq. (10) and the RTL.

```
flexbe.py             engine: memory, switches, schedule, bit reversal, arithmetic
bspnet.py             application: Eq. (1) features, BL branches, Table 7 configs
test_flexbe.py        33 unit tests (the "test program")
demo_flexbe.py        report reproducing the paper's headline numbers
check_algorithm1.py   consistency analysis of Algorithm 1 (see note below)
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
* the subset switch uses at most `m` distinct states;
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

## Note on Algorithm 1, lines 5–6

The control law in lines 7–11 is exactly what a conflict-free shift-down
schedule needs, and the simulator reproduces it. The index formula in lines 5–6,

```
ii     <- rotate_{n-1}(base, k+1)
I^i[0] <- rotate_n(2*ii, k)
```

read literally, does **not** appear to select one representative per
conflict-free cycle. `check_algorithm1.py` tries both rotation amounts (`k+1`
as printed and `k`) and both stage orders (`h = n-1-k` and `h = k`), and:

* against the simulator's schedule it fails at the first stage in every size
  tested (`N` = 8…4096, `P_bu` = 2…16);
* for `N` = 8, 16, 32 with `P_bu` = 2 it enumerates *all* conflict-free
  groupings of a stage and finds that none of them admits the `I^i[0]` values as
  distinct cycle representatives — e.g. for `N` = 8 the only conflict-free
  grouping is row-based `{0,1,2,3} / {4,5,6,7}`, while the formula yields
  representatives `0` and `2`, which share a row.

Since the released generator presumably emits a working sequence, the likely
explanation is a transcription slip in the listing (in lines 5–6, in how `base`
advances, or in how the remaining entries `I^i[1..2P_bu-1]` are enumerated),
rather than an error in the hardware — but it is worth a look before
resubmission, since a reader reimplementing from the listing would not get a
conflict-free schedule. The simulator instead derives the schedule from the
conflict-freedom requirement itself and matches lines 7–11 exactly.

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
