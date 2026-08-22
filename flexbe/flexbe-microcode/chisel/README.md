# FlexBE microcoded command processor — Chisel

Synthesizable Chisel RTL for the design in `../paper/paper.pdf`. The modules
mirror the Python cycle-level model (`../flexbe.py`, `../microcode.py`), and the
tests assert the correspondence against golden vectors emitted from that model.

## Layout

```
src/main/scala/flexbe/
  FlexBits.scala           popcount, bit-reverse, insert-zero, bsm (Eq. 3)
  Complex.scala        Q1.15 complex type + general 2x2 butterfly unit
  PRS.scala            permute-rotate switch: barrel shifter + subset switch
  CycleControl.scala   per-cycle idx / addr / R / S / coeff (two regimes)
  BankedMemory.scala   P = 2*P_bu single-port banks, per-bank addressing
  Engine.scala         the per-cycle loop (Fig. 2): read, PRS-A, BU, PRS-B, write
  Descriptor.scala     32-byte command word + decode (matches microcode.py)
  Sequencer.scala      fetch / decode / dispatch, tag scoreboard, completion
  Emit.scala           SystemVerilog emission
src/test/scala/flexbe/
  Golden.scala         control-path vectors from flexbe.py (do not edit)
  NumGolden.scala      numeric I/O + twiddles from flexbe.py (do not edit)
  FlexBESpec.scala     backend-free golden checks + elaboration
  EngineSimSpec.scala  Engine FSM sequencing (needs a Verilog backend)
  EngineNumericSpec.scala  end-to-end 16-point FFT vs the fixed-point model
```

## Build

Requires JDK 17+ and, on first build, network access to Maven Central for the
Chisel 6.6.0 artifacts.

```
sbt test                          # golden checks + elaboration (no backend)
sbt "testOnly *EngineSimSpec"     # FSM sequencing in simulation (needs Verilator)
sbt "testOnly *EngineNumericSpec" # end-to-end FFT numerics (needs Verilator)
sbt "runMain flexbe.Emit"    # emit SystemVerilog into generated/
```

## Correspondence to the model

Each combinational block is a direct transcription of the Python reference, and
the identity was checked exhaustively before this RTL was written:

| Chisel | reference | check |
|---|---|---|
| `Bits.bsm` | `flexbe.bsm` | all 2^15 indices, m = 5 |
| `SubsetPerm.table` | `PermuteRotateSwitch.perm` | P = 4 |
| `PRS` read map | `PermuteRotateSwitch.read_map` | every (R, S) |
| `CycleControl` | `flexbe.cycle_control` | 17{,}285 cycles, N up to 2^15, P_bu = 16 |
| `Descriptor.decode` | `microcode.Descriptor` | byte layout `<8B6I` |

`FlexBESpec` re-runs the first three at elaboration time, replays the full
16-point `cycle_control` trace through the `CycleControl` module, and drives the
`Engine` FSM through one transform to confirm the `start → busy → done`
sequencing over `stages × (N/P)` cycles.

## Scope and simplifications

* The datapath is the in-place radix-2 loop for `P_sub = 1`. Sub-parallel
  interleaving (`P_sub > 1`) and the `P_N` depth packing are expressed in the
  addressing and the descriptor fields but are exercised in RTL only through the
  `P_sub = 1` path here; the Python model covers the general case.
* `BankedMemory` uses `SyncReadMem` (one-cycle read latency) with per-bank
  write strobes. The engine loop is pipelined one deep: cycle *t* issues the
  read, cycle *t+1* computes and writes back using control held in a delay
  register. Adjacent cycles walk disjoint (bank, depth) slots, so there is no
  read-after-write hazard; a final drain cycle commits the last write. The
  equivalence of this schedule to sequential read-modify-write was checked
  against the Python model (zero error).
* `Sequencer` models the control plane. Each dispatched command carries a cycle
  budget (from the Eq. 10 model) that its resource lane counts down, reproducing
  the DMA/compute overlap the Python `Sequencer` measures. Wiring the lanes to
  real AXI-DMA and to the `Engine`'s `start`/`done` is the integration step.
* The bit-reversal write-back (`bs.brev` / `BITREV`) and the pointwise units
  (`bs.pow` / `bs.mag` / `bs.pool`) are present in the command set and the
  cost/timing model; their datapaths are left as instantiation points.
