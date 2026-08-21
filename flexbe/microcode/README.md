# FlexBE simulator + microcoded command processor

A NumPy simulator of the flexible butterfly engine of

> X. Liu, R. Wu, P. H. W. Leong, *"A Flexible FPGA-based Butterfly Engine for
> Accelerating Signal Processing and Machine Learning"*

together with a model of the **microcoded command processor** that makes it
programmable, and its attachment to a Zynq UltraScale+ MPSoC.

It is a verification model, not a performance estimate of the RTL: it executes
the real dataflow — banked memory, switching network, butterfly units, per-cycle
control — and checks the architectural claims the paper makes.

```
FlexBE_microcode.ipynb   the analysis, already executed (read this first)
flexbe.py                the engine: memory, PRS, schedule, bit reversal, Eq. (10)
bspnet.py                the application: Eq. (1) features, BL branches, Table 7
microcode.py             descriptors, program builder, sequencer, Zynq attachment
test_flexbe.py           49 tests
build_notebook.py        regenerates the notebook (no Jupyter required)
```

## Quick start

```bash
python test_flexbe.py        # 49 tests, ~3 s
python build_notebook.py     # regenerate FlexBE_microcode.ipynb
```

```python
import numpy as np, flexbe as fx

eng = fx.FlexBE(P_bu=16)                       # 32 lanes, m = 5
x   = np.random.randn(4, 4096) + 1j*np.random.randn(4, 4096)

y, st = eng.transform(x, bitrev=True)          # 4 packed FFTs, natural order out
assert np.allclose(y, np.fft.fft(x, axis=1))
print(st)      # N=4096 l=4096 P_sub=1 P_N=4 stages=12 bfly=6144 bitrev=512 ...

W = fx.random_bl_coefficients(32, np.random.default_rng(0))
h, _ = eng.transform(np.random.randn(1024, 32), W)   # same datapath, BL weights
s, st = eng.transform(np.random.randn(4096, 8))      # short: st.P_sub == 4
```

Every call runs the full banked-memory + PRS datapath (`datapath="cycle"`);
`datapath="vector"` uses a golden datapath while still modelling cycles.

## What is modelled

| paper | simulator |
|---|---|
| Eq. (3) shift-down storage | `bsm`, `BankedMemory` |
| Eq. (2) read / permute / butterfly / restore / write | `FlexBE._run` |
| Sec. 3.1 PRS = barrel shifter + subset switch | `PermuteRotateSwitch`, `prs_cost` |
| per-cycle index vectors and PRS control | `cycle_control` |
| Sec. 3.2 / Alg. 2 sub-parallelism, Eq. (5) | `FlexBE.geometry`, `interleave`, `expand_coefficients` |
| Sec. 3.3 / Alg. 3 bit-reversal write-back | `bitrev_schedule`, `FlexBE._bitrev` |
| Sec. 3.4 `P_N` packing (Fig. 5) | `transform` with several input rows |
| Eq. (10) cycle model | `cycles_eq10`, `layer_cycles` |
| Sec. 5.2 Q1.15 arithmetic | `FixedPointFormat` |
| Sec. 4, Eq. (1), Tables 4 / 7 | `bspnet.BSPNet`, `CONFIGS` |
| Table 10 `F_max` | `bspnet.HW`, `design_space_table` |

**Stage convention.** Decimation in frequency: stage `k` pairs indices differing
in bit `h = n-1-k`, so natural-order input leaves the result bit reversed — which
is what the Sec. 3.3 write-back undoes. Conflict freedom then forces two
regimes, and the subset-switch state falls out of them:

```
h <- n-1-k
if h < m:  H <- j                      I^i[2t] <- (H << m) + ins(t, h)     S^i <- h
else:      e <- j mod 2
           H <- ins(j >> 1, h-m)       I^i[2t] <- (H << m) + 2t + e        S^i <- 0
both:      I^i[2t+1] <- I^i[2t] + 2^h  R^i <- bsm(I^i[0])
```

For `h < m` a cycle reads a whole RAM row, whose banks
`(popcount(H) + low) mod 2^m` are all distinct; for `h >= m` bit `h-m` of `H` is
clear, so `popcount(H + 2^(h-m)) = popcount(H) + 1` and slot `j` reads bank
`(popcount(H) + e + j) mod 2^m` — a pure rotation. Each branch emits exactly
`N/(2*P_bu)` cycles per stage and covers every index once.

`flexbe.cycle_control()` is the closed form of both regimes and the **only**
control path in the simulator.

## Microcoded command processor

A PL-side sequencer executing 32-byte descriptors. On Zynq this is the only
option — the A53 is hard IP with no custom-instruction port — and it is the right
granularity anyway, since the datapath retires 64 butterflies per cycle.

```
bs.load / bs.wload / bs.store        DMA;  store flags NATURAL | BITREV | MAG
bs.bfly                              P_N transforms of length l; FFT/IFFT/BL/BL_EXPAND
bs.brev                              standalone Algorithm 3 pass
bs.pow / bs.mag / bs.pool            Eq. (1), magnitude, NormPool
```

cfg-6 is ~104 descriptors, 3.3 kB. Measured on that program:

| issue mechanism | µs | overhead | samples/s | critical |
|---|---|---|---|---|
| PYNQ / Jupyter flow | 878.3 | +286.9% | 1,139 | **host** |
| MMIO per command, Linux | 254.9 | +12.3% | 3,923 | compute |
| static descriptor ring, A53 | 249.3 | +9.8% | 4,011 | compute |
| ring + per-resource queues, R5 | 247.2 | +8.9% | 4,046 | compute |

* the **PYNQ flow is host bound**, not accelerator bound — much of the gap
  between the solid and dashed lines of Fig. 15a is PS-side software, and a
  descriptor ring recovers it without touching the PL;
* prefetch saturates at **4–8 descriptors**;
* **per-resource queues matter only across inferences** — with one shared ring
  the next sample's input-DMA descriptor sits behind ~100 compute descriptors;
  splitting takes batch-10 throughput to 94% of datapath peak;
* the link is never the constraint: 0.62 GB/s of 6.4, so **one HP port suffices**;
* the sequencer costs **~3.6k LUT (+2.4%)** on the Table 11 system.

Design rule: descriptors expose the memory and the transform, never the
schedule. No opcode names a bank index, a PRS control word or a cycle, so the
addressing and control can be retuned without breaking a descriptor program.

## Verification

`test_flexbe.py` asserts rather than assumes: bank-conflict freedom and complete
coverage for every cycle of every stage; `P_f = P_s x P_r` cycle by cycle;
`R = bsm(I[0])` and the closed-form `S`; exact FFT/IFFT/BL against
`numpy.fft`; Algorithm 3 conflict free on both memories in exactly `N/P` cycles;
sub-parallelism; fxp16 SQNR > 40 dB; measured cycles equal Eq. (10) including
**`cycles_eq10(4, 32768, 1, 4, 16) == 15360`**; descriptor encode/decode
round-trip; and that the sequencer never beats the datapath it drives.

## Extending

* **Trained weights** — `coefficients_from_paper_layout` converts the
  `[2,2,l/2,log2 l]` layout of Table 4, so exported PyTorch BL weights drop in.
* **RTL cross-check** — `ButterflySchedule.all_cycles()` yields `(I^i, Addr^i,
  R^i, S^i, coefficient addresses)` per cycle: a ready-made stimulus/reference
  stream for a cocotb or UVM testbench against the Verilog core.
* **Other platforms** — add a `MicrocodeConfig` entry; every PS-side assumption
  lives there.
