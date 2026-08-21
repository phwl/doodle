"""
microcode.py -- the microcoded command processor in front of the engine
=======================================================================

FlexBE as published is fixed function: the layer sequence is hardwired.  This
module models the minimal change that makes it programmable -- a PL-side
sequencer that executes a queue of 32-byte descriptors -- and quantifies what
that costs on a Zynq UltraScale+ MPSoC.

Why a sequencer and not a custom instruction: the Cortex-A53 is hard IP, so
there is no custom-instruction port (Arm Custom Instructions are Cortex-M33/M55
only, and the A53 is ARMv8.0 -- 128-bit NEON, no SVE).  Descriptors are also
what keeps issue off the critical path: the datapath retires 64 butterflies per
cycle, so any per-command host round trip is expensive by comparison.

Contents
--------
  Descriptor    the 32-byte command word, with encode/decode
  OPCODES       bs.load / bs.wload / bs.store / bs.bfly / bs.brev /
                bs.pow / bs.mag / bs.pool
  build_program the ~100-descriptor program for one BSPNet inference
  MicrocodeConfig / PLATFORMS   issue, DMA, completion and clock assumptions
  Sequencer     event-driven model of prefetch, DMA/compute overlap, completion
  sweeps        ring_depth_sweep, batch_sweep, port_budget, area

Everything on the accelerator side comes from bspnet.cycle_breakdown, i.e. from
Eq. (10); only the PS-side latencies are new assumptions and they all live in
MicrocodeConfig.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import bspnet as bs

__all__ = [
    "OPCODES", "Descriptor", "Command", "build_program",
    "MicrocodeConfig", "PLATFORMS", "Sequencer", "RunReport",
    "compare_platforms", "ring_depth_sweep", "batch_sweep", "port_budget",
    "sequencer_area", "replace",
]


# ---------------------------------------------------------------------------
# the command set
# ---------------------------------------------------------------------------

OPCODES: Dict[str, int] = {
    "bs.load":  0x01,   # memory -> butterfly scratchpad
    "bs.wload": 0x02,   # memory -> coefficient RAM
    "bs.store": 0x03,   # scratchpad -> memory; flags: NATURAL | BITREV | MAG
    "bs.bfly":  0x10,   # P_N transforms of length l; mode FFT/IFFT/BL/BL_EXPAND
    "bs.brev":  0x11,   # standalone Algorithm 3 pass
    "bs.pow":   0x20,   # elementwise s^2 .. s^8            (Eq. 1)
    "bs.mag":   0x21,   # |x|, optional peak normalisation
    "bs.pool":  0x22,   # LayerNorm -> shortcut -> ReLU -> maxpool  (NormPool)
}

MODES = {"FFT": 0, "IFFT": 1, "BL": 2, "BL_EXPAND": 3}
FLAGS = {"NATURAL": 0x00, "BITREV": 0x01, "MAG": 0x02, "NORMALISE": 0x04,
         "RELU": 0x08, "LAYERNORM": 0x10, "SHORTCUT": 0x20, "MAXPOOL": 0x40}

# which unit executes each opcode
_RESOURCE = {"bs.load": "dma_in", "bs.wload": "dma_in", "bs.store": "dma_out",
             "bs.bfly": "compute", "bs.brev": "compute", "bs.pow": "compute",
             "bs.mag": "compute", "bs.pool": "compute"}


@dataclass
class Descriptor:
    """One 32-byte command word.

    layout (little endian):
      u8  opcode | u8 flags | u8 mode | u8 log2_l
      u8  log2_Psub | u8 stage_lo | u8 stage_hi | u8 pool
      u32 src | u32 dst | u32 coeff | u32 count | u32 P_N | u32 tag
    """
    opcode: str
    flags: int = 0
    mode: int = 0
    log2_l: int = 0
    log2_Psub: int = 0
    stage_lo: int = 0
    stage_hi: int = 31
    pool: int = 0
    src: int = 0
    dst: int = 0
    coeff: int = 0
    count: int = 0
    P_N: int = 0
    tag: int = 0

    SIZE = 32

    def encode(self) -> bytes:
        return struct.pack("<8B6I", OPCODES[self.opcode], self.flags, self.mode,
                           self.log2_l, self.log2_Psub, self.stage_lo,
                           self.stage_hi, self.pool, self.src, self.dst,
                           self.coeff, self.count, self.P_N, self.tag)

    @classmethod
    def decode(cls, raw: bytes) -> "Descriptor":
        f = struct.unpack("<8B6I", raw)
        name = next(k for k, v in OPCODES.items() if v == f[0])
        return cls(name, f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                   f[8], f[9], f[10], f[11], f[12], f[13])

    @property
    def resource(self) -> str:
        return _RESOURCE[self.opcode]


@dataclass
class Command:
    """A descriptor plus what it costs the hardware."""
    desc: Descriptor
    name: str
    cycles: float = 0.0             # compute commands
    nbytes: int = 0                 # DMA commands
    deps: Tuple[int, ...] = ()
    inference: int = 0

    @property
    def kind(self) -> str:
        return self.desc.resource


# ---------------------------------------------------------------------------
# program for one BSPNet inference
# ---------------------------------------------------------------------------

def build_program(cfg: bs.BSPNetConfig, hw: bs.HWConfig, inference: int = 0,
                  first_index: int = 0, chunks: int = 1) -> List[Command]:
    """The descriptor program for one inference.

    Compute cycles come from bspnet.cycle_breakdown, so the accelerator side is
    exactly the Eq. (10) model.  `chunks` splits the input load so the squaring
    layers can start before the whole record has landed.
    """
    br = dict(bs.cycle_breakdown(cfg, hw))
    fft = [v for k, v in br.items() if "FFT" in k][0]
    brev = [v for k, v in br.items() if "bit-reversal" in k]
    exp = [v for k, v in br.items() if "expansion" in k][0] / cfg.n_branches
    mlp = [v for k, v in br.items() if "MLP" in k]
    elem = 4                                        # complex fxp16

    cmds: List[Command] = []
    i = first_index

    def add(desc: Descriptor, name: str, cycles=0.0, nbytes=0, deps=()) -> int:
        nonlocal i
        desc.tag = i
        cmds.append(Command(desc, name, cycles, nbytes, tuple(deps), inference))
        i += 1
        return i - 1

    nch = max(1, chunks)
    ch = [add(Descriptor("bs.load", dst=0, count=cfg.N // nch),
              f"bs.load  IQ chunk {c}", nbytes=cfg.N * elem // nch)
          for c in range(nch)]
    src = ch[-1]

    pw = [add(Descriptor("bs.pow", mode=p, count=cfg.N), f"bs.pow   S{p}",
              cycles=cfg.N / 128, deps=[ch[0] if nch > 1 else src])
          for p in (2, 4, 6, 8)]
    if nch > 1:
        pw.append(src)

    nl = int(math.log2(cfg.N))
    last = add(Descriptor("bs.bfly", mode=MODES["FFT"], log2_l=nl, P_N=4,
                          stage_hi=nl - 1),
               f"bs.bfly  4 x {cfg.N}-point FFT", cycles=fft, deps=pw)
    if brev and cfg.bit_reversal:
        last = add(Descriptor("bs.store", flags=FLAGS["BITREV"], count=cfg.N),
                   "bs.store BITREV write-back", cycles=brev[0], deps=[last])

    tails = []
    lm, ld = int(math.log2(cfg.d_m)), int(math.log2(cfg.d_in))
    P = 2 * hw.P_bu
    psub = int(math.log2(P // cfg.d_in)) if cfg.d_in < P else 0
    for b in range(cfg.n_branches):
        m = add(Descriptor("bs.mag", flags=FLAGS["NORMALISE"], count=cfg.N),
                f"bs.mag   branch {b}", cycles=cfg.N / 128, deps=[last])
        e = add(Descriptor("bs.bfly", mode=MODES["BL_EXPAND"], log2_l=ld,
                           log2_Psub=psub, P_N=cfg.N // cfg.d_in,
                           stage_hi=ld - 1),
                f"bs.bfly  expansion {b}", cycles=exp, deps=[m])
        p = add(Descriptor("bs.pool", flags=FLAGS["MAXPOOL"], pool=4),
                f"bs.pool  branch {b}", cycles=cfg.N / cfg.d_in / 64, deps=[e])
        for k in range(cfg.n_mlp_blocks):
            c = mlp[k] / cfg.n_branches / 2
            rows = max(1, cfg.L // (cfg.pool ** k))
            a0 = add(Descriptor("bs.bfly", mode=MODES["BL"], log2_l=lm,
                                P_N=rows, flags=FLAGS["RELU"], stage_hi=lm - 1),
                     f"bs.bfly  MLP-{k}-0 b{b}", cycles=c, deps=[p])
            a1 = add(Descriptor("bs.bfly", mode=MODES["BL"], log2_l=lm,
                                P_N=rows, stage_hi=lm - 1),
                     f"bs.bfly  MLP-{k}-1 b{b}", cycles=c, deps=[a0])
            p = add(Descriptor("bs.pool", pool=cfg.pool,
                               flags=FLAGS["LAYERNORM"] | FLAGS["SHORTCUT"]
                               | FLAGS["RELU"] | FLAGS["MAXPOOL"]),
                    f"bs.pool  NormPool-{k} b{b}",
                    cycles=max(8.0, cfg.L / (cfg.pool ** k) / 8), deps=[a1])
        tails.append(p)

    rows_out = max(1, cfg.L // (cfg.pool ** cfg.n_mlp_blocks))
    nb = cfg.n_branches * rows_out * cfg.d_m * elem
    add(Descriptor("bs.store", count=nb // elem), "bs.store features",
        nbytes=nb, deps=tails)
    return cmds


def program_bytes(cmds: Sequence[Command]) -> int:
    return len(cmds) * Descriptor.SIZE


# ---------------------------------------------------------------------------
# platform assumptions
# ---------------------------------------------------------------------------

@dataclass
class MicrocodeConfig:
    """PS-PL and sequencer assumptions.  Times in microseconds."""
    name: str
    issue_mode: str = "static_ring"   # 'mmio' | 'ring' | 'static_ring'

    mmio_write_us: float = 0.40       # posted write + ordering, APU -> HPM0
    doorbell_us: float = 0.40
    desc_fetch_us: float = 0.15       # sequencer descriptor read over HPC0
    desc_build_us: float = 0.05       # host cost to write one descriptor
    prefetch_depth: int = 8
    ring_slots: int = 16
    per_resource_queues: bool = True  # one ring per unit
    input_chunks: int = 1

    n_hp_ports: int = 4
    port_bw_gbs: float = 4.8          # 128b @ 300 MHz
    dram_bw_gbs: float = 6.4
    dma_setup_us: float = 0.25
    python_call_us: float = 0.0

    completion_us: float = 0.20       # ACP write + poll
    f_mhz: float = 300.0
    double_buffer: bool = True

    @property
    def dma_bw_gbs(self) -> float:
        return min(self.n_hp_ports * self.port_bw_gbs, self.dram_bw_gbs)

    def dma_us(self, nbytes: int) -> float:
        return (self.dma_setup_us + self.python_call_us +
                nbytes / (self.dma_bw_gbs * 1e9) * 1e6)


PLATFORMS: Dict[str, MicrocodeConfig] = {
    "pynq": MicrocodeConfig(
        "PYNQ / Jupyter flow (as measured in the paper)", issue_mode="mmio",
        python_call_us=25.0, mmio_write_us=8.0, completion_us=20.0,
        dma_setup_us=1.0, per_resource_queues=False),
    "mmio": MicrocodeConfig(
        "MMIO per command, A53 + Linux driver", issue_mode="mmio",
        mmio_write_us=0.40, completion_us=6.0, per_resource_queues=False),
    "ring": MicrocodeConfig(
        "descriptor ring rebuilt per inference, A53", issue_mode="ring",
        per_resource_queues=False, prefetch_depth=4, ring_slots=8),
    "static": MicrocodeConfig(
        "static descriptor ring + doorbell, A53", issue_mode="static_ring",
        per_resource_queues=False, prefetch_depth=4, ring_slots=8),
    "tuned": MicrocodeConfig(
        "static ring, per-resource queues, streamed input (R5)",
        issue_mode="static_ring", doorbell_us=0.10, desc_fetch_us=0.10,
        per_resource_queues=True, prefetch_depth=8, ring_slots=16,
        input_chunks=8, completion_us=0.10),
}


# ---------------------------------------------------------------------------
# the sequencer model
# ---------------------------------------------------------------------------

@dataclass
class RunReport:
    latency_us: float = 0.0
    compute_us: float = 0.0
    n_commands: int = 0
    busy: Dict[str, float] = field(default_factory=dict)
    per_inference_us: float = 0.0
    critical: str = ""

    @property
    def overhead_pct(self) -> float:
        return (self.latency_us / self.compute_us - 1.0) * 100.0

    @property
    def throughput_sps(self) -> float:
        return 1e6 / self.per_inference_us

    def __str__(self) -> str:
        u = ", ".join(f"{k}={v/self.latency_us:.0%}"
                      for k, v in sorted(self.busy.items()))
        return (f"{self.latency_us:8.1f} us (compute {self.compute_us:7.1f} us, "
                f"+{self.overhead_pct:5.1f}%) critical={self.critical} [{u}]")


class Sequencer:
    """Event-driven model of descriptor issue, DMA and compute overlap.

    Resources: `host` (MMIO writes or descriptor building), `seq` (fetch and
    decode in PL), `dma_in`, `dma_out`, `compute`.  Commands issue in program
    order but occupy different resources, so a DMA overlaps compute exactly as
    in the Fig. 8 pipeline.  The sequencer runs `prefetch_depth` commands ahead,
    bounded by `ring_slots`; with `per_resource_queues` the prefetch gate is the
    previous command *of the same kind*, which is what lets the next sample's
    input DMA be fetched while the current inference is still computing.
    """

    def __init__(self, cfgm: MicrocodeConfig):
        self.p = cfgm

    def run(self, cmds: Sequence[Command]) -> RunReport:
        p = self.p
        free: Dict[str, float] = {}
        busy: Dict[str, float] = {}
        done = [0.0] * len(cmds)
        same: Dict[str, List[float]] = {"dma_in": [], "dma_out": [],
                                        "compute": []}
        t_end = compute_only = 0.0

        for n, c in enumerate(cmds):
            # ---- issue -----------------------------------------------------
            if p.issue_mode == "mmio":
                s = free.get("host", 0.0)
                free["host"] = s + p.mmio_write_us
                busy["host"] = busy.get("host", 0.0) + p.mmio_write_us
                ready = free["host"]
            else:
                build = p.desc_build_us if p.issue_mode == "ring" else 0.0
                if build:
                    s = free.get("host", 0.0)
                    free["host"] = s + build
                    busy["host"] = busy.get("host", 0.0) + build
                slot = done[n - p.ring_slots] if n >= p.ring_slots else 0.0
                s = max(free.get("seq", 0.0), slot,
                        free.get("host", 0.0) if build else 0.0)
                if n == 0:
                    s += p.doorbell_us
                free["seq"] = s + p.desc_fetch_us
                busy["seq"] = busy.get("seq", 0.0) + p.desc_fetch_us
                if p.per_resource_queues:
                    hist = same[c.kind]
                    ahead = (hist[-p.prefetch_depth]
                             if len(hist) >= p.prefetch_depth else 0.0)
                else:
                    ahead = (done[n - p.prefetch_depth]
                             if n >= p.prefetch_depth else 0.0)
                ready = max(free["seq"], ahead)

            # ---- execute ---------------------------------------------------
            if c.kind == "compute":
                res, dur = "compute", c.cycles / (p.f_mhz * 1e6) * 1e6
                compute_only += dur
            else:
                res = c.kind if p.double_buffer else "compute"
                dur = p.dma_us(c.nbytes)
            dep = max((done[d] for d in c.deps), default=0.0)
            start = max(ready, dep, free.get(res, 0.0))
            free[res] = start + dur
            busy[res] = busy.get(res, 0.0) + dur
            done[n] = start + dur
            same.setdefault(c.kind, []).append(done[n])
            t_end = max(t_end, done[n])

        t_end += p.completion_us
        n_inf = len(set(c.inference for c in cmds))
        rep = RunReport(t_end, compute_only, len(cmds), busy)
        rep.critical = max(busy, key=busy.get) if busy else ""
        rep.per_inference_us = t_end / max(1, n_inf)
        return rep


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def _stream(cfg: bs.BSPNetConfig, hw: bs.HWConfig, batch: int,
            chunks: int) -> List[Command]:
    out: List[Command] = []
    for b in range(batch):
        out.extend(build_program(cfg, hw, inference=b, first_index=len(out),
                                 chunks=chunks))
    return out


def compare_platforms(cfg: Optional[bs.BSPNetConfig] = None,
                      batch: int = 1) -> List[Dict]:
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    rows = []
    for key, p in PLATFORMS.items():
        rep = Sequencer(p).run(_stream(cfg, hw, batch, p.input_chunks))
        rows.append(dict(key=key, name=p.name, latency_us=rep.latency_us,
                         compute_us=rep.compute_us,
                         overhead_pct=rep.overhead_pct,
                         throughput=rep.throughput_sps, critical=rep.critical,
                         n_cmd=rep.n_commands))
    return rows


def ring_depth_sweep(cfg: Optional[bs.BSPNetConfig] = None,
                     depths=(1, 2, 4, 8, 16, 32), key: str = "tuned",
                     batch: int = 1) -> List[Dict]:
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    out = []
    for d in depths:
        p = replace(PLATFORMS[key], prefetch_depth=d, ring_slots=max(d, 2))
        rep = Sequencer(p).run(_stream(cfg, hw, batch, p.input_chunks))
        out.append(dict(depth=d, latency_us=rep.latency_us,
                        per_inference_us=rep.per_inference_us,
                        overhead_pct=rep.overhead_pct, critical=rep.critical))
    return out


def batch_sweep(cfg: Optional[bs.BSPNetConfig] = None,
                batches=(1, 2, 4, 6, 8, 10),
                keys=("pynq", "static", "tuned")) -> List[Dict]:
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    peak = 1e6 / (bs.total_cycles(cfg, hw) / (hw.f_mhz * 1e6) * 1e6)
    out = []
    for b in batches:
        row = dict(batch=b, peak=peak)
        for k in keys:
            p = PLATFORMS[k]
            rep = Sequencer(p).run(_stream(cfg, hw, b, p.input_chunks))
            row[k] = rep.throughput_sps
            row[k + "_us"] = rep.per_inference_us
        out.append(row)
    return out


def port_budget(cfg: Optional[bs.BSPNetConfig] = None,
                p: Optional[MicrocodeConfig] = None) -> Dict:
    """Is the PS-PL link ever the constraint?  (It is not.)"""
    cfg = cfg or bs.CONFIGS["cfg-6"]
    p = p or PLATFORMS["tuned"]
    hw = bs.HW["BSP-Flex"]
    lat_s = bs.total_cycles(cfg, hw) / (hw.f_mhz * 1e6)
    in_b = cfg.N * 4
    rows_out = max(1, cfg.L // (cfg.pool ** cfg.n_mlp_blocks))
    out_b = cfg.n_branches * rows_out * cfg.d_m * 4
    need = (in_b + out_b) / lat_s / 1e9
    return dict(in_bytes=in_b, out_bytes=out_b, latency_ms=lat_s * 1e3,
                required_gbs=need, available_gbs=p.dma_bw_gbs,
                utilisation=need / p.dma_bw_gbs,
                min_hp_ports=max(1, math.ceil(need / p.port_bw_gbs)))


# ---------------------------------------------------------------------------
# area
# ---------------------------------------------------------------------------

# Table 11, complete BSP-Flex system on ZCU104
BSP_FLEX_AREA = {"LUT": 152_553, "FF": 183_826, "BRAM36": 268, "URAM": 64,
                 "DSP": 869}
DEVICE_XCZU7EV = {"LUT": 230_400, "FF": 460_800, "DSP": 1_728,
                  "BRAM36": 312, "URAM": 96}


def sequencer_area(ring_slots: int = 16, microcode_words: int = 512) -> Dict:
    """Descriptor sequencer on top of the Table 11 system.

    A descriptor FSM with a small ROM and a prefetch buffer: a few thousand LUTs
    against the 15k already spent on top-level control, plus BRAM for the ring.
    """
    lut = 3_000 + 40 * ring_slots
    bram = math.ceil(ring_slots * Descriptor.SIZE * 8 / (36 * 1024)) + \
        math.ceil(microcode_words * 32 / (36 * 1024))
    total_lut = BSP_FLEX_AREA["LUT"] + lut
    return dict(sequencer_lut=lut, sequencer_bram36=max(1, bram),
                total_lut=total_lut,
                total_bram36=BSP_FLEX_AREA["BRAM36"] + max(1, bram),
                lut_overhead_pct=lut / BSP_FLEX_AREA["LUT"] * 100,
                fits=total_lut <= DEVICE_XCZU7EV["LUT"])
