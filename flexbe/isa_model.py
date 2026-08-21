"""
isa_model.py -- instruction-level backend for the "how programmable?" question
=============================================================================

Compares four ways of making the FlexBE workload software programmable:

  A  microcoded command processor  -- hardwired inner loop, descriptor queue
  B  RoCC-style custom instruction -- butterfly scratchpad is architectural
                                      state, operands never cross a VRF
  C  RVV + custom extensions       -- Zvbfly / Zvshfl / Zvtwid, operands live
                                      in the vector register file
  D  RVV 1.0 as-is                 -- vrgather + vslide + scalar twiddle loads

A and B are modelled analytically on top of the FlexBE cycle model (they *are*
the FlexBE datapath, with different amounts of sequencing overhead).  C and D
are modelled by generating the actual instruction stream for each layer and
running it through an in-order issue model with functional-unit occupancy, VRF
port pressure and LSU bandwidth.

The instruction streams are executable: `run_program` implements the semantics
of every instruction, so `test_flexbe.TestISAModel` checks that the generated
program computes the same transform as numpy before any cycle is counted.  The
instruction *counts* are therefore exact; only the per-class rates, the issue
width and the clock are assumptions, and they all live in `MachineConfig`.

Proposed instructions modelled here
-----------------------------------
  vbfly.vv    vd, vs2, vs1, h, egw   fused stride-aware radix-2 butterfly:
                                     the PRS sits *inside* the functional unit
  vbflyx.vv   vd0,vd1, vs2,vs1, vs3  cross-register butterfly (stride >= VL)
  vshfl.vi    vd, vs2, S, egw        stride permutation P_s (log-depth)
  vunshfl.vi  vd, vs2, S, egw        its inverse
  vtwid.vx    vd, cfg                in-register twiddle generation
  vcmul/vcadd/vcsub                  complex arithmetic (Zvcplx style)

Note that the rotation P_r of the paper is *not* needed here: it exists to
align data with the shift-down RAM banks, and a vector register file has no
banks.  Only the subset switch P_s survives into the ISA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import bspnet as bs
import flexbe as fx

__all__ = [
    "MachineConfig", "MACHINES", "Inst", "Program", "RegAlloc",
    "gen_transform", "run_program", "IssueModel", "CycleReport",
    "AreaEstimate", "estimate_area", "accelerator_cycles",
    "layer_cycles_isa", "bspnet_cycles_isa", "compare_options",
    "vlen_sweep", "lsu_sweep", "DEVICE_XCZU7EV",
]


# ---------------------------------------------------------------------------
# machine description
# ---------------------------------------------------------------------------

@dataclass
class MachineConfig:
    """Everything assumed about a vector machine.

    Rates are in complex elements per cycle.  `lanes` is the arithmetic width:
    a machine with lanes = 128 retires 64 complex butterflies per cycle when
    fused butterflies are available, i.e. it matches the 64-BU FlexBE array.
    """
    name: str
    vlen_bits: int = 16384          # architectural VLEN
    elem_bits: int = 32             # complex fxp16 = 16 + 16
    lanes: int = 128                # arith elements/cycle
    mul_rate: Optional[int] = None  # complex-multiply elements/cycle (def. lanes)
    perm_rate: Optional[int] = None # vshfl/vrot elements/cycle (def. lanes)
    gather_rate: int = 16           # vrgather elements/cycle (multi-pass xbar)
    lsu_elems: int = 32             # load/store elements/cycle
    issue_width: float = 2.0        # instructions/cycle
    vrf_read_elems: int = 384       # total VRF read bandwidth, elements/cycle
    vrf_write_elems: int = 128      # total VRF write bandwidth
    f_mhz: float = 250.0
    unroll: int = 2                 # software pipelining depth assumed
    fused_butterfly: bool = True    # Zvbfly present
    has_shuffle: bool = True        # Zvshfl present (else vrgather)
    has_vtwid: bool = True          # Zvtwid present (else twiddles from memory)
    has_complex: bool = True        # Zvcplx present (else 4 mul + 2 add)
    separate_twid_unit: bool = False  # vtwid has its own pipe (not the vector ALU)
    hoist_twiddles: bool = True     # twiddle vectors are loop-invariant across
                                    # chunks; a compiler keeps them in registers
    n_twid_regs: int = 8            # architectural registers spared for them
    scratchpad_kb: int = 512        # on-chip working set the LSU is fed from
    scalar_overhead: float = 0.0    # extra cycles per loop body

    def __post_init__(self):
        if self.mul_rate is None:
            self.mul_rate = self.lanes
        if self.perm_rate is None:
            self.perm_rate = self.lanes

    @property
    def VL(self) -> int:
        """Complex elements per vector register."""
        return self.vlen_bits // self.elem_bits

    def rate(self, cls: str) -> float:
        return {
            "load": self.lsu_elems, "store": self.lsu_elems,
            "arith": self.lanes, "mul": self.mul_rate,
            "perm": self.perm_rate if self.has_shuffle else self.gather_rate,
            "twid": self.lanes,
        }[cls]


def _machines() -> Dict[str, MachineConfig]:
    """Reference points.  'matched' machines are sized to the 64-BU FlexBE."""
    m = {}
    m["C-fused"] = MachineConfig(
        "C  RVV + Zvbfly/Zvshfl/Zvtwid (fused)", vlen_bits=16384, lanes=128,
        f_mhz=250.0)
    m["C-explicit"] = MachineConfig(
        "C' RVV + Zvshfl/Zvtwid, separate permute", vlen_bits=16384, lanes=128,
        f_mhz=250.0, fused_butterfly=False)
    m["C-twidunit"] = MachineConfig(
        "C+ as C-fused, vtwid on its own pipe", vlen_bits=16384, lanes=128,
        f_mhz=250.0, separate_twid_unit=True)
    m["C-wide"] = MachineConfig(
        "C* C+ with 512 B/cycle LSU", vlen_bits=16384, lanes=128, f_mhz=250.0,
        separate_twid_unit=True, lsu_elems=128)
    m["D-rvv10"] = MachineConfig(
        "D  RVV 1.0 (vrgather, no Zvcplx/Zvtwid)", vlen_bits=16384, lanes=128,
        f_mhz=250.0, fused_butterfly=False, has_shuffle=False,
        has_vtwid=False, has_complex=False, gather_rate=16)
    return m


MACHINES = _machines()


DEVICE_XCZU7EV = {"LUT": 230_400, "FF": 460_800, "DSP": 1_728,
                  "BRAM36": 312, "URAM": 96}


# ---------------------------------------------------------------------------
# instructions
# ---------------------------------------------------------------------------

@dataclass
class Inst:
    op: str
    cls: str                        # load/store/arith/mul/perm/twid
    elems: int                      # elements processed (both dsts counted)
    dst: Tuple[int, ...] = ()
    src: Tuple[int, ...] = ()
    imm: Dict = field(default_factory=dict)


Program = List[Inst]


class RegAlloc:
    """Round-robin register sets, modelling `unroll`-deep software pipelining."""

    def __init__(self, unroll: int = 2, per_set: int = 6, base: int = 1):
        self.unroll, self.per_set, self.base = max(1, unroll), per_set, base
        self.i = 0

    def next_set(self) -> List[int]:
        s = self.base + (self.i % self.unroll) * self.per_set
        self.i += 1
        return [s + k for k in range(self.per_set)]


# ---------------------------------------------------------------------------
# program generation
# ---------------------------------------------------------------------------

def _twiddles(l: int, h: int, egw: int, VL: int, base: int) -> np.ndarray:
    """Twiddle for every lane of a register, DIF stage with hole bit h.

    Lane i holds element index (base + i); its butterfly twiddle is
    W_l ** ((a mod 2**h) * l / 2**(h+1)) with a the index inside the transform.
    """
    idx = (base + np.arange(VL, dtype=np.int64)) % egw if egw < VL else \
          (base + np.arange(VL, dtype=np.int64))
    e = (idx % (1 << h)) * (l >> (h + 1))
    return np.exp(-2j * np.pi * e / l)


class TwiddlePool:
    """Loop-invariant twiddle vectors kept live in registers (LRU, size N).

    The twiddle for stage h depends only on (index mod 2**h), which repeats
    every chunk, so one generation serves every chunk of a loop nest.
    """

    def __init__(self, mc: MachineConfig, base_reg: int = 24):
        self.mc, self.base = mc, base_reg
        self.map: Dict[Tuple, int] = {}
        self.order: List[Tuple] = []
        self.free: List[int] = [base_reg + k for k in range(mc.n_twid_regs)]

    def get(self, prog: Program, l: int, h: int, egw: int, base: int) -> int:
        mc = self.mc
        imm = dict(l=l, h=h, egw=egw, base=base)
        key = (l, h, base % (1 << h) if h < 62 else base)
        if mc.hoist_twiddles and key in self.map:
            return self.map[key]                       # already live: free
        if mc.hoist_twiddles and not self.free:
            self.free.append(self.map.pop(self.order.pop(0)))   # LRU evict
        reg = self.free.pop(0) if mc.hoist_twiddles else self.base
        if mc.has_vtwid:
            prog.append(Inst("vtwid.vx", "twid", mc.VL, (reg,), (), imm))
        else:
            prog.append(Inst("vle32.v", "load", mc.VL, (reg,), (),
                             dict(mem="twiddle", off=0, **imm)))
        if mc.hoist_twiddles:
            self.map[key] = reg
            self.order.append(key)
        return reg


def _emit_stage_in_reg(prog: Program, mc: MachineConfig, regs: List[int],
                       l: int, h: int, egw: int, base: int,
                       pool: "TwiddlePool") -> None:
    """One butterfly stage on data already resident in a vector register."""
    VL = mc.VL
    d, tmp = regs[0], regs[2]
    imm = dict(l=l, h=h, egw=egw, base=base)
    tw = pool.get(prog, l, h, egw, base)
    if mc.fused_butterfly:
        prog.append(Inst("vbfly.vv", "arith", VL, (d,), (d, tw), imm))
        if not mc.has_complex:              # no complex MAC: 4 mul + 2 add
            for _ in range(2):
                prog.append(Inst("vmul.vv", "mul", VL, (d,), (d, tw), imm))
        return
    # explicit permute: bring butterfly partners into adjacent lanes
    perm_op = "vshfl.vi" if mc.has_shuffle else "vrgather.vv"
    if not mc.has_shuffle:                  # vrgather needs an index vector
        prog.append(Inst("vle32.v", "load", VL, (tmp,), (),
                         dict(mem="gather_idx", off=0, **imm)))
    prog.append(Inst(perm_op, "perm", VL, (tmp,), (d,), dict(S=h, **imm)))
    prog.append(Inst("vbfly_adj.vv", "arith", VL, (tmp,), (tmp, tw), imm))
    if not mc.has_complex:
        for _ in range(2):
            prog.append(Inst("vmul.vv", "mul", VL, (tmp,), (tmp, tw), imm))
    inv_op = "vunshfl.vi" if mc.has_shuffle else "vrgather.vv"
    if not mc.has_shuffle:            # the inverse needs its own index vector
        prog.append(Inst("vle32.v", "load", VL, (regs[3],), (),
                         dict(mem="gather_idx", off=0, **imm)))
    prog.append(Inst(inv_op, "perm", VL, (d,), (tmp,), dict(S=h, inv=True, **imm)))


def gen_transform(l: int, n_seq: int, mc: MachineConfig,
                  mem: str = "x") -> Program:
    """Instruction stream for n_seq butterfly transforms of length l.

    Two regimes, exactly the ones that decide the answer:

      l <= VL   every stage is intra-register; one load and one store per
                chunk of VL/l transforms.  This is the BSPNet BL case, and it
                is where sub-parallelism appears in the ISA as an element-group
                width.

      l >  VL   stages with stride >= VL pair *different* registers and need no
                permute at all, but cost one load+store pass over the data per
                stage; the final log2(VL) stages are all done while a chunk is
                resident, i.e. one further pass.  Total memory passes
                log2(l) - log2(VL) + 1 -- this is what makes option C lose.
    """
    VL, nl = mc.VL, int(math.log2(l))
    prog: Program = []
    ra = RegAlloc(mc.unroll)
    pool = TwiddlePool(mc)

    if l <= VL:
        per_chunk = VL // l
        n_chunks = math.ceil(n_seq / per_chunk)
        for c in range(n_chunks):
            regs = ra.next_set()
            off = c * VL
            prog.append(Inst("vle32.v", "load", VL, (regs[0],), (),
                             dict(mem=mem, off=off)))
            for k in range(nl):
                _emit_stage_in_reg(prog, mc, regs, l, nl - 1 - k, l,
                                   off, pool)
            prog.append(Inst("vse32.v", "store", VL, (), (regs[0],),
                             dict(mem=mem, off=off)))
        return prog

    v = int(math.log2(VL))
    for s in range(n_seq):
        sbase = s * l
        # --- stages whose stride spans registers -------------------------
        for k in range(nl - v):
            h = nl - 1 - k
            for blk in range(0, l, 1 << (h + 1)):
                for c in range(0, 1 << h, VL):
                    regs = ra.next_set()
                    a, b, tmp = regs[0], regs[1], regs[3]
                    oa, ob = sbase + blk + c, sbase + blk + c + (1 << h)
                    imm = dict(l=l, h=h, egw=l, base=blk + c)
                    prog.append(Inst("vle32.v", "load", VL, (a,), (),
                                     dict(mem=mem, off=oa)))
                    prog.append(Inst("vle32.v", "load", VL, (b,), (),
                                     dict(mem=mem, off=ob)))
                    tw = pool.get(prog, l, h, l, blk + c)
                    if mc.fused_butterfly:
                        prog.append(Inst("vbflyx.vv", "arith", 2 * VL, (a, b),
                                         (a, b, tw), imm))
                    else:
                        prog.append(Inst("vcsub.vv", "arith", VL, (tmp,),
                                         (a, b), imm))
                        prog.append(Inst("vcadd.vv", "arith", VL, (a,),
                                         (a, b), imm))
                        n_mul = 1 if mc.has_complex else 6
                        for _ in range(n_mul):
                            prog.append(Inst("vcmul.vv", "mul", VL, (b,),
                                             (tmp, tw), imm))
                    prog.append(Inst("vse32.v", "store", VL, (), (a,),
                                     dict(mem=mem, off=oa)))
                    prog.append(Inst("vse32.v", "store", VL, (), (b,),
                                     dict(mem=mem, off=ob)))
        # --- final pass: every remaining stage while resident -------------
        for c in range(0, l, VL):
            regs = ra.next_set()
            off = sbase + c
            prog.append(Inst("vle32.v", "load", VL, (regs[0],), (),
                             dict(mem=mem, off=off)))
            for h in range(v - 1, -1, -1):
                _emit_stage_in_reg(prog, mc, regs, l, h, VL, c, pool)
            prog.append(Inst("vse32.v", "store", VL, (), (regs[0],),
                             dict(mem=mem, off=off)))
    return prog


# ---------------------------------------------------------------------------
# functional execution (so the traces are provably the right program)
# ---------------------------------------------------------------------------

def _apply_bfly(x: np.ndarray, tw: np.ndarray, h: int, egw: int) -> np.ndarray:
    """DIF radix-2 butterflies at stride 2**h inside element groups of egw."""
    y = x.copy()
    n = len(y)
    for g in range(0, n, egw):
        blk = y[g:g + egw].reshape(-1, 2, 1 << h)
        w = tw[g:g + egw].reshape(-1, 2, 1 << h)[:, 0, :]
        lo, hi = blk[:, 0, :].copy(), blk[:, 1, :].copy()
        blk[:, 0, :] = lo + hi
        blk[:, 1, :] = (lo - hi) * w
        y[g:g + egw] = blk.reshape(-1)
    return y


def _perm(S: int, egw: int, n: int) -> np.ndarray:
    """Lane map of vshfl.vi: the stride permutation P_s, per element group."""
    pi = np.empty(egw, dtype=np.int64)
    for t in range(egw // 2):
        pi[2 * t] = fx.insert_zero(t, S)
        pi[2 * t + 1] = pi[2 * t] + (1 << S)
    return np.concatenate([pi + g for g in range(0, n, egw)])


def run_program(prog: Program, mem: Dict[str, np.ndarray],
                mc: MachineConfig) -> Dict[str, np.ndarray]:
    """Execute the trace.  Returns the (mutated) memory image."""
    VL = mc.VL
    regs: Dict[int, np.ndarray] = {}
    for ins in prog:
        im = ins.imm
        if ins.op == "vle32.v":
            if im.get("mem") == "twiddle":
                regs[ins.dst[0]] = _twiddles(im["l"], im["h"], im["egw"], VL,
                                             im["base"])
            elif im.get("mem") == "gather_idx":
                regs[ins.dst[0]] = np.zeros(VL, dtype=np.complex128)
            else:
                o = im["off"]
                regs[ins.dst[0]] = mem[im["mem"]][o:o + VL].copy()
        elif ins.op == "vse32.v":
            o = im["off"]
            mem[im["mem"]][o:o + VL] = regs[ins.src[0]]
        elif ins.op == "vtwid.vx":
            regs[ins.dst[0]] = _twiddles(im["l"], im["h"], im["egw"], VL,
                                         im["base"])
        elif ins.op in ("vbfly.vv", "vbfly_adj.vv"):
            h = 0 if ins.op == "vbfly_adj.vv" else im["h"]
            egw = 2 if ins.op == "vbfly_adj.vv" else min(im["egw"], VL)
            src = regs[ins.src[0]]
            tw = regs[ins.src[1]]
            if ins.op == "vbfly_adj.vv":     # partners already adjacent
                tw = tw[_perm(im["h"], min(im["egw"], VL), VL)]
                egw = 2
            regs[ins.dst[0]] = _apply_bfly(src, tw, h, egw)
        elif ins.op == "vbflyx.vv":
            a, b, tw = (regs[r] for r in ins.src)
            regs[ins.dst[0]], regs[ins.dst[1]] = a + b, (a - b) * tw
        elif ins.op == "vcadd.vv":
            regs[ins.dst[0]] = regs[ins.src[0]] + regs[ins.src[1]]
        elif ins.op == "vcsub.vv":
            regs[ins.dst[0]] = regs[ins.src[0]] - regs[ins.src[1]]
        elif ins.op == "vcmul.vv":
            regs[ins.dst[0]] = regs[ins.src[0]] * regs[ins.src[1]]
        elif ins.op == "vmul.vv":
            pass                              # cost-only expansion of vcmul
        elif ins.op in ("vshfl.vi", "vrgather.vv", "vunshfl.vi"):
            p = _perm(im["S"], min(im["egw"], VL), VL)
            if im.get("inv") or ins.op == "vunshfl.vi":
                out = np.empty(VL, dtype=np.complex128)
                out[p] = regs[ins.src[0]]
                regs[ins.dst[0]] = out
            else:
                regs[ins.dst[0]] = regs[ins.src[0]][p]
        else:                                 # pragma: no cover
            raise ValueError(f"unknown op {ins.op}")
    return mem


# ---------------------------------------------------------------------------
# in-order issue model
# ---------------------------------------------------------------------------

@dataclass
class CycleReport:
    cycles: int = 0
    instructions: int = 0
    busy: Dict[str, float] = field(default_factory=dict)
    stall_cycles: int = 0
    bottleneck: str = ""

    @property
    def utilisation(self) -> Dict[str, float]:
        return {k: v / self.cycles for k, v in self.busy.items()} \
            if self.cycles else {}

    def __str__(self) -> str:
        u = ", ".join(f"{k}={v:.0%}" for k, v in sorted(self.utilisation.items()))
        return (f"{self.cycles:,} cycles, {self.instructions:,} instructions, "
                f"bottleneck={self.bottleneck} [{u}]")


class IssueModel:
    """In-order, single-issue-per-class machine with pipelined functional units.

    Per instruction the occupancy of its unit is
        occ = max(elems / rate(class), elems*reads / vrf_read_bw,
                  elems*writes / vrf_write_bw, 1 / issue_width)
    so VRF port pressure and LSU bandwidth throttle exactly like the unit rate.
    Instructions issue in order and stall on RAW; `unroll` independent register
    sets in the generated code model software pipelining.
    """

    def __init__(self, mc: MachineConfig):
        self.mc = mc
        self.FU = {"load": "lsu", "store": "lsu", "arith": "alu", "mul": "alu",
                   "perm": "perm",
                   "twid": "twid" if mc.separate_twid_unit else "alu"}

    def run(self, prog: Program) -> CycleReport:
        mc = self.mc
        unit_free: Dict[str, float] = {}
        reg_ready: Dict[int, float] = {}
        busy: Dict[str, float] = {}
        now = 0.0
        issue_slot = 0.0
        stalls = 0.0
        for ins in prog:
            fu = self.FU[ins.cls]
            occ = max(ins.elems / mc.rate(ins.cls),
                      ins.elems * max(1, len(ins.src)) / mc.vrf_read_elems,
                      ins.elems * max(1, len(ins.dst)) / mc.vrf_write_elems,
                      1.0 / mc.issue_width)
            dep = max((reg_ready.get(r, 0.0) for r in ins.src), default=0.0)
            start = max(dep, unit_free.get(fu, 0.0), issue_slot)
            stalls += max(0.0, start - max(unit_free.get(fu, 0.0), issue_slot))
            unit_free[fu] = start + occ
            issue_slot = start + 1.0 / mc.issue_width
            for r in ins.dst:
                reg_ready[r] = start + occ
            busy[fu] = busy.get(fu, 0.0) + occ
            now = max(now, start + occ)
        now += mc.scalar_overhead
        rep = CycleReport(int(math.ceil(now)), len(prog), busy,
                          int(round(stalls)))
        rep.bottleneck = max(busy, key=busy.get) if busy else ""
        return rep


# ---------------------------------------------------------------------------
# options A and B: sequencing overhead on top of the FlexBE datapath
# ---------------------------------------------------------------------------

@dataclass
class SequencerConfig:
    name: str
    cycles_per_command: float        # decode/handshake per issued command
    commands_per_layer: float        # 1 = whole layer, log2(l) = per stage
    f_derate: float = 1.0            # clock relative to the hardwired 300 MHz
    extra_lut: int = 0
    extra_bram: int = 0


SEQUENCERS = {
    "A": SequencerConfig("A  microcoded command processor", 8.0, 1.0, 1.00,
                         extra_lut=8_000, extra_bram=4),
    "B": SequencerConfig("B  RoCC custom instruction", 6.0, 1.0, 0.95,
                         extra_lut=20_000, extra_bram=2),
    "B-stage": SequencerConfig("B' RoCC, one instruction per stage", 6.0, 0.0,
                               0.95, extra_lut=20_000, extra_bram=2),
}


def accelerator_cycles(cfg: bs.BSPNetConfig, hw: bs.HWConfig,
                       seq: SequencerConfig) -> float:
    """FlexBE cycles plus per-command sequencing overhead."""
    total = 0.0
    for name, cyc in bs.cycle_breakdown(cfg, hw):
        n_cmd = seq.commands_per_layer
        if n_cmd == 0.0:                      # one command per butterfly stage
            n_cmd = 15.0 if "FFT" in name else 5.0
        rep = 8.0 if name.startswith("8 x") else 1.0
        total += cyc + seq.cycles_per_command * n_cmd * rep
    return total


# ---------------------------------------------------------------------------
# workload-level cycle counts for options C and D
# ---------------------------------------------------------------------------

def layer_cycles_isa(n_seq: int, l: int, mc: MachineConfig) -> CycleReport:
    prog = gen_transform(l, n_seq, mc)
    return IssueModel(mc).run(prog)


def bspnet_cycles_isa(cfg: bs.BSPNetConfig, mc: MachineConfig
                      ) -> Tuple[float, List[Tuple[str, float, str]]]:
    """Run every BSPNet layer through the instruction model."""
    items: List[Tuple[str, float, str]] = []
    r = layer_cycles_isa(4, cfg.N, mc)
    items.append((f"CC extraction: 4 x {cfg.N}-point FFT", r.cycles,
                  r.bottleneck))

    rows = cfg.N // cfg.d_in
    n_exp = cfg.d_m // cfg.d_in
    r = layer_cycles_isa(rows * n_exp, cfg.d_in, mc)
    items.append((f"8 x feature expansion (len {cfg.d_in}, x{n_exp})",
                  cfg.n_branches * r.cycles, r.bottleneck))

    for blk in range(cfg.n_mlp_blocks):
        rows_b = max(1, cfg.L // (cfg.pool ** blk))
        r = layer_cycles_isa(rows_b, cfg.d_m, mc)
        items.append((f"8 x MLP-{blk}-{{0,1}} (len {cfg.d_m}, {rows_b} rows)",
                      cfg.n_branches * 2 * r.cycles, r.bottleneck))
    return sum(c for _, c, _ in items), items


# ---------------------------------------------------------------------------
# area model, calibrated to Table 10 / Table 11
# ---------------------------------------------------------------------------

# FlexBE standalone BE array on ZCU104 (Table 10): 64 BUs -> 61,574 LUT, 642 DSP
LUT_PER_BU = 61_574 / 64
DSP_PER_BU = 642 / 64
BRAM36_BITS = 36 * 1024


@dataclass
class AreaEstimate:
    name: str
    lut: int
    dsp: int
    bram36: int
    detail: Dict[str, int] = field(default_factory=dict)

    def fits(self, dev=DEVICE_XCZU7EV) -> bool:
        return (self.lut <= dev["LUT"] and self.dsp <= dev["DSP"]
                and self.bram36 <= dev["BRAM36"])

    def __str__(self) -> str:
        return (f"{self.name:44s} LUT {self.lut:8,}  DSP {self.dsp:6,}  "
                f"BRAM {self.bram36:5,}  {'fits' if self.fits() else 'DOES NOT FIT'}")


def estimate_area(mc: MachineConfig, vrf_regs: int = 32,
                  vrf_port_factor: int = 3) -> AreaEstimate:
    """Rough resource estimate for a vector machine of this width.

    Butterfly arithmetic is calibrated against Table 10 (962 LUT and 10 DSP per
    radix-2 BU).  The permute network uses the paper's own cost model: a
    log-depth shuffle costs Theta(m*P) 2:1 muxes, a full vrgather crossbar
    Theta(P^2) -- which is the whole point of Sec. 3.1.
    """
    bu = mc.lanes // 2                        # complex butterflies per cycle
    lut = int(bu * LUT_PER_BU)
    dsp = int(bu * DSP_PER_BU)
    detail = {"butterfly_datapath": lut}

    # permute network over the lane group
    P = mc.lanes
    m = max(1, P.bit_length() - 1)
    if mc.has_shuffle:
        mux2 = m * P + (m + 1) * (P // 2)     # PRS-style, Sec. 3.1
    else:
        mux2 = P * (P - 1)                    # full crossbar, Eq. (4)
    perm_lut = int(mux2 * mc.elem_bits / 2)   # ~2 muxes per 6-LUT
    lut += perm_lut
    detail["permute_network"] = perm_lut

    # vector register file
    vrf_bits = vrf_regs * mc.vlen_bits
    bram = math.ceil(vrf_bits / BRAM36_BITS) * vrf_port_factor
    vrf_lut = int(mc.lanes * mc.elem_bits * 1.5)      # port muxing/bypass
    lut += vrf_lut
    detail["vrf_muxing"] = vrf_lut
    detail["vrf_bram36"] = bram

    # banked scratchpad + bank network needed to sustain lsu_elems/cycle.
    # This is the term that turns option C into option B: buying LSU bandwidth
    # means building the same wide banked memory the accelerator already has.
    banks = max(1, mc.lsu_elems)
    bm = max(1, banks.bit_length() - 1)
    bank_net_lut = int((bm * banks + (bm + 1) * (banks // 2)) * mc.elem_bits / 2)
    sp_bram = math.ceil(mc.scratchpad_kb * 8 * 1024 / BRAM36_BITS)
    lut += bank_net_lut
    bram += sp_bram
    detail["scratchpad_network"] = bank_net_lut
    detail["scratchpad_bram36"] = sp_bram

    # decode, hazard logic, scalar core
    ctrl = 25_000
    lut += ctrl
    detail["core_and_control"] = ctrl
    return AreaEstimate(mc.name, lut, dsp, bram, detail)


def accelerator_area(seq: SequencerConfig) -> AreaEstimate:
    """Table 11 BSP-Flex system plus the sequencer."""
    lut = 152_553 + seq.extra_lut
    return AreaEstimate(seq.name, lut, 869, int(268.5) + seq.extra_bram,
                        {"bsp_flex_table11": 152_553,
                         "sequencer": seq.extra_lut})


# ---------------------------------------------------------------------------
# top-level comparison
# ---------------------------------------------------------------------------

def compare_options(cfg: Optional[bs.BSPNetConfig] = None) -> List[Dict]:
    """cfg-6 on every option: cycles, clock, latency, area, perf per LUT."""
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    rows: List[Dict] = []

    base_cycles = bs.total_cycles(cfg, hw)
    base_lat = base_cycles / (hw.f_mhz * 1e6) * 1e3
    rows.append(dict(option="hardwired (paper)", cycles=base_cycles,
                     f_mhz=hw.f_mhz, latency_ms=base_lat, lut=152_553,
                     dsp=869, bram=268, fits=True, bottleneck="-",
                     programmability="none"))

    for key in ("A", "B"):
        seq = SEQUENCERS[key]
        cyc = accelerator_cycles(cfg, hw, seq)
        f = hw.f_mhz * seq.f_derate
        area = accelerator_area(seq)
        rows.append(dict(option=seq.name, cycles=cyc, f_mhz=f,
                         latency_ms=cyc / (f * 1e6) * 1e3, lut=area.lut,
                         dsp=area.dsp, bram=area.bram36, fits=area.fits(),
                         bottleneck="datapath",
                         programmability="descriptors" if key == "A" else "C"))

    for key in ("C-fused", "C-twidunit", "C-wide", "C-explicit", "D-rvv10"):
        mc = MACHINES[key]
        cyc, items = bspnet_cycles_isa(cfg, mc)
        area = estimate_area(mc)
        neck = max(set(b for _, _, b in items),
                   key=lambda b: sum(c for _, c, bb in items if bb == b))
        rows.append(dict(option=mc.name, cycles=cyc, f_mhz=mc.f_mhz,
                         latency_ms=cyc / (mc.f_mhz * 1e6) * 1e3, lut=area.lut,
                         dsp=area.dsp, bram=area.bram36, fits=area.fits(),
                         bottleneck=neck, programmability="RVV toolchain"))

    for r in rows:
        r["speedup_vs_hardwired"] = r["latency_ms"] / base_lat
        r["perf_per_klut"] = (1e3 / r["latency_ms"]) / (r["lut"] / 1e3)
    return rows


def vlen_sweep(cfg: Optional[bs.BSPNetConfig] = None,
               vlens=(1024, 2048, 4096, 8192, 16384, 32768, 65536),
               base: str = "C-fused") -> List[Dict]:
    cfg = cfg or bs.CONFIGS["cfg-6"]
    out = []
    for v in vlens:
        mc = MachineConfig(**{**MACHINES[base].__dict__, "vlen_bits": v,
                              "name": f"VLEN={v}"})
        cyc, items = bspnet_cycles_isa(cfg, mc)
        area = estimate_area(mc)
        neck = max(set(b for _, _, b in items),
                   key=lambda b: sum(c for _, c, bb in items if bb == b))
        out.append(dict(vlen=v, VL=mc.VL, cycles=cyc,
                        latency_ms=cyc / (mc.f_mhz * 1e6) * 1e3,
                        lut=area.lut, bram=area.bram36, fits=area.fits(),
                        bottleneck=neck))
    return out


def lsu_sweep(cfg: Optional[bs.BSPNetConfig] = None,
              rates=(8, 16, 32, 64, 128, 256), base: str = "C-fused"
              ) -> List[Dict]:
    """The decisive sensitivity: how much LSU bandwidth closes the gap?"""
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    ref = bs.total_cycles(cfg, hw) / (hw.f_mhz * 1e6) * 1e3
    out = []
    for r in rates:
        mc = MachineConfig(**{**MACHINES[base].__dict__, "lsu_elems": r,
                              "name": f"LSU={r} elem/cyc"})
        cyc, items = bspnet_cycles_isa(cfg, mc)
        neck = max(set(b for _, _, b in items),
                   key=lambda b: sum(c for _, c, bb in items if bb == b))
        lat = cyc / (mc.f_mhz * 1e6) * 1e3
        out.append(dict(lsu=r, bytes_per_cycle=r * 4, cycles=cyc,
                        latency_ms=lat, ratio=lat / ref, bottleneck=neck))
    return out
