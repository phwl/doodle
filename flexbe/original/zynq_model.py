"""
zynq_model.py -- how the accelerator attaches to a Zynq UltraScale+ MPSoC
=========================================================================

Option B cannot exist on Zynq in the RoCC sense: the Cortex-A53 is hard IP, so
there is no custom-instruction port (Arm Custom Instructions are Cortex-M33/M55
only, and the A53 is ARMv8.0 -- 128-bit NEON, no SVE).  What survives is a
PL-side command sequencer fed by descriptors, i.e. option A, which the
instruction-level study already measured at 1.00x of the hardwired datapath.

This module models the part that decides whether that 1.00x survives contact
with the PS: command issue, descriptor fetch, DMA/compute overlap, and
completion notification.  It answers three questions with numbers:

  1. how much of Fig. 15a's gap between the measured solid line and the dashed
     peak is PS-side software rather than the accelerator;
  2. how deep the descriptor ring must be to hide the PS round trip;
  3. what MMIO-per-command, a descriptor ring, a *static* ring and the paper's
     PYNQ flow each cost at batch 1.

Interfaces modelled (UltraScale+ port names)
--------------------------------------------
  S_AXI_HP0-3_FPD   4 x 128b non-coherent, bulk IQ in / features out
  S_AXI_HPC0_FPD    128b coherent via CCI-500, descriptor fetch
  S_AXI_ACP_FPD     128b, allocates into A53 L2, completion words only
  M_AXI_HPM0_FPD    control registers and doorbell from the APU
  M_AXI_HPM0_LPD    the same from the R5F in the RPU

The compute cycle counts come straight from bspnet.cycle_breakdown, so the
accelerator side is exactly the model the rest of the package validates; only
the PS-side latencies are new assumptions, and they all live in PSPLConfig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import bspnet as bs

__all__ = [
    "replace", "PSPLConfig", "PLATFORMS", "Command", "bspnet_commands", "PSPLModel",
    "RunReport", "compare_platforms", "ring_depth_sweep", "batch_sweep",
    "port_budget",
]


# ---------------------------------------------------------------------------
# platform description
# ---------------------------------------------------------------------------

@dataclass
class PSPLConfig:
    """PS-PL interface assumptions.  All times in microseconds."""
    name: str
    issue_mode: str = "ring"        # 'mmio' | 'ring' | 'static_ring'
    host: str = "a53_linux"         # 'a53_linux' | 'a53_baremetal' | 'r5'

    # command issue
    mmio_write_us: float = 0.40     # posted write + ordering, APU -> HPM0
    doorbell_us: float = 0.40       # one per inference for the ring modes
    desc_bytes: int = 32
    desc_fetch_us: float = 0.15     # sequencer descriptor read over HPC0
    desc_build_us: float = 0.05     # host cost to write one descriptor
    prefetch_depth: int = 4         # descriptors the sequencer runs ahead
    ring_slots: int = 8             # descriptors resident in the ring
    per_resource_queues: bool = False   # one ring per resource, so a DMA
                                    # descriptor is not stuck behind compute
    input_chunks: int = 1           # split the input load so compute can start
                                    # before the whole record has landed

    # data movement
    n_hp_ports: int = 4
    port_bw_gbs: float = 4.8        # 128b @ 300 MHz
    dram_bw_gbs: float = 6.4        # realistic sustained DDR4 for one stream
    dma_setup_us: float = 0.25      # per transfer, hardware descriptor mode
    python_call_us: float = 0.0     # PYNQ/Python per-transfer overhead

    # completion
    completion: str = "poll_acp"    # 'irq' | 'poll_acp' | 'poll_ocm'
    completion_us: float = 0.20

    # accelerator
    f_mhz: float = 300.0
    double_buffer: bool = True      # overlap next input DMA with current compute

    @property
    def dma_bw_gbs(self) -> float:
        return min(self.n_hp_ports * self.port_bw_gbs, self.dram_bw_gbs)

    def dma_us(self, nbytes: int) -> float:
        return self.dma_setup_us + self.python_call_us + \
            nbytes / (self.dma_bw_gbs * 1e9) * 1e6


def _platforms() -> Dict[str, PSPLConfig]:
    p: Dict[str, PSPLConfig] = {}
    p["pynq"] = PSPLConfig(
        "PYNQ / Jupyter flow (as measured in the paper)", issue_mode="mmio",
        host="a53_linux", python_call_us=25.0, mmio_write_us=8.0,
        completion="poll_ocm", completion_us=20.0, dma_setup_us=1.0)
    p["mmio-linux"] = PSPLConfig(
        "MMIO per command, A53 + Linux driver", issue_mode="mmio",
        host="a53_linux", mmio_write_us=0.40, completion="irq",
        completion_us=6.0)
    p["mmio-baremetal"] = PSPLConfig(
        "MMIO per command, A53 bare metal", issue_mode="mmio",
        host="a53_baremetal", mmio_write_us=0.25, completion="poll_acp",
        completion_us=0.20)
    p["ring-a53"] = PSPLConfig(
        "descriptor ring rebuilt per inference, A53", issue_mode="ring",
        host="a53_linux", completion="poll_acp", completion_us=0.20)
    p["static-a53"] = PSPLConfig(
        "static descriptor ring + doorbell, A53", issue_mode="static_ring",
        host="a53_linux", completion="poll_acp", completion_us=0.20)
    p["static-r5"] = PSPLConfig(
        "static ring, Cortex-R5F drives it (RPU)", issue_mode="static_ring",
        host="r5", doorbell_us=0.10, desc_fetch_us=0.10,
        completion="poll_ocm", completion_us=0.10)
    p["tuned-r5"] = PSPLConfig(
        "static ring, per-resource queues, streamed input (R5)",
        issue_mode="static_ring", host="r5", doorbell_us=0.10,
        desc_fetch_us=0.10, prefetch_depth=8, ring_slots=16,
        per_resource_queues=True, input_chunks=8,
        completion="poll_ocm", completion_us=0.10)
    return p


PLATFORMS = _platforms()


# ---------------------------------------------------------------------------
# command stream
# ---------------------------------------------------------------------------

@dataclass
class Command:
    name: str
    kind: str                       # 'dma_in' | 'dma_out' | 'compute'
    nbytes: int = 0
    cycles: float = 0.0
    deps: Tuple[int, ...] = ()
    inference: int = 0


def bspnet_commands(cfg: bs.BSPNetConfig, hw: bs.HWConfig,
                    inference: int = 0, first_index: int = 0,
                    prev_input: Optional[int] = None,
                    chunks: int = 1) -> List[Command]:
    """The ~100-command stream of one BSPNet inference.

    Mirrors the primitive set of the accelerator: bs.load / bs.pow / bs.bfly /
    bs.mag / bs.pool / bs.store.  Compute cycles are taken from
    bspnet.cycle_breakdown so the accelerator side stays consistent with the
    rest of the package.
    """
    br = dict(bs.cycle_breakdown(cfg, hw))
    fft = [v for k, v in br.items() if "FFT" in k][0]
    brev = [v for k, v in br.items() if "bit-reversal" in k]
    exp = [v for k, v in br.items() if "expansion" in k][0] / cfg.n_branches
    mlp = [v for k, v in br.items() if "MLP" in k]
    elem_bytes = 4                                  # complex fxp16

    cmds: List[Command] = []
    i = first_index

    def add(name, kind, nbytes=0, cycles=0.0, deps=()):
        nonlocal i
        cmds.append(Command(name, kind, nbytes, cycles, tuple(deps), inference))
        i += 1
        return i - 1

    n_chunk = max(1, chunks)
    chunk_ids = [add(f"bs.load  IQ chunk {c}", "dma_in",
                     cfg.N * elem_bytes // n_chunk,
                     deps=[prev_input] if prev_input is not None and c == 0
                     else [])
                 for c in range(n_chunk)]
    src = chunk_ids[-1]
    # the squaring layers consume the record chunk by chunk
    pw = [add(f"bs.pow   S{p}", "compute", cycles=cfg.N / 128,
              deps=[chunk_ids[min(len(chunk_ids) - 1, 0)]] if n_chunk > 1
              else [src])
          for p in (2, 4, 6, 8)]
    if n_chunk > 1:
        pw = pw + [src]             # the FFT still needs the whole record
    f = add("bs.bfly  4 x N-point FFT", "compute", cycles=fft, deps=pw)
    last = f
    if brev:
        last = add("bs.store BITREV write-back", "compute", cycles=brev[0],
                   deps=[f])

    tails = []
    for b in range(cfg.n_branches):
        m = add(f"bs.mag   branch {b}", "compute", cycles=cfg.N / 128,
                deps=[last])
        e = add(f"bs.bfly  expansion {b}", "compute", cycles=exp, deps=[m])
        p = add(f"bs.pool  branch {b}", "compute",
                cycles=cfg.N / cfg.d_in / 64, deps=[e])
        for k in range(cfg.n_mlp_blocks):
            c = mlp[k] / cfg.n_branches / 2
            a0 = add(f"bs.bfly  MLP-{k}-0 b{b}", "compute", cycles=c, deps=[p])
            a1 = add(f"bs.bfly  MLP-{k}-1 b{b}", "compute", cycles=c, deps=[a0])
            p = add(f"bs.pool  NormPool-{k} b{b}", "compute",
                    cycles=max(8.0, cfg.L / (cfg.pool ** k) / 8), deps=[a1])
        tails.append(p)

    rows_out = max(1, cfg.L // (cfg.pool ** cfg.n_mlp_blocks))
    add("bs.store features", "dma_out",
        cfg.n_branches * rows_out * cfg.d_m * elem_bytes, deps=tails)
    return cmds


# ---------------------------------------------------------------------------
# PS-PL timing model
# ---------------------------------------------------------------------------

@dataclass
class RunReport:
    latency_us: float = 0.0
    compute_us: float = 0.0         # the accelerator's own time
    n_commands: int = 0
    busy: Dict[str, float] = field(default_factory=dict)
    critical: str = ""
    per_inference_us: float = 0.0

    @property
    def overhead_pct(self) -> float:
        return (self.latency_us / self.compute_us - 1.0) * 100.0

    @property
    def throughput_sps(self) -> float:
        return 1e6 / self.per_inference_us

    def __str__(self) -> str:
        u = ", ".join(f"{k}={v/self.latency_us:.0%}"
                      for k, v in sorted(self.busy.items()))
        return (f"{self.latency_us:8.1f} us  (compute {self.compute_us:7.1f} us,"
                f" +{self.overhead_pct:5.1f}%)  critical={self.critical} [{u}]")


class PSPLModel:
    """Event-driven model of issue, DMA and compute overlap.

    Resources: `host` (MMIO writes / descriptor building), `seq` (descriptor
    fetch and decode in PL), `dma_in`, `dma_out`, `compute`.  Commands issue in
    program order but occupy different resources, so DMA for one inference
    overlaps compute of another exactly as in the Fig. 8 pipeline.  In the ring
    modes the sequencer may run `prefetch_depth` commands ahead, bounded by
    `ring_slots` outstanding descriptors.
    """

    def __init__(self, cfgp: PSPLConfig):
        self.p = cfgp

    def run(self, cmds: Sequence[Command]) -> RunReport:
        p = self.p
        free: Dict[str, float] = {}
        done: List[float] = [0.0] * len(cmds)
        busy: Dict[str, float] = {}
        issued: List[float] = [0.0] * len(cmds)
        same_kind: Dict[str, List[float]] = {"dma_in": [], "dma_out": [],
                                             "compute": []}
        t_end = 0.0
        compute_only = 0.0

        for n, c in enumerate(cmds):
            # ---- issue path ------------------------------------------------
            if p.issue_mode == "mmio":
                cost = p.mmio_write_us
                start_i = max(free.get("host", 0.0), 0.0)
                free["host"] = start_i + cost
                busy["host"] = busy.get("host", 0.0) + cost
                ready = free["host"]
            else:
                build = p.desc_build_us if p.issue_mode == "ring" else 0.0
                if build:
                    s = free.get("host", 0.0)
                    free["host"] = s + build
                    busy["host"] = busy.get("host", 0.0) + build
                # sequencer fetch, may run ahead but not past the ring
                slot_free = done[n - p.ring_slots] if n >= p.ring_slots else 0.0
                s = max(free.get("seq", 0.0), slot_free,
                        free.get("host", 0.0) if build else 0.0)
                if n == 0:
                    s += p.doorbell_us
                free["seq"] = s + p.desc_fetch_us
                busy["seq"] = busy.get("seq", 0.0) + p.desc_fetch_us
                # prefetch: a command may be fetched while earlier ones run.
                # With one shared ring the gate is the n-depth command overall;
                # with per-resource rings it is the n-depth command *of the same
                # kind*, which is what lets the next input DMA be fetched while
                # the current inference is still computing.
                if p.per_resource_queues:
                    hist = same_kind[c.kind]
                    ahead = hist[-p.prefetch_depth] if len(hist) >= p.prefetch_depth else 0.0
                else:
                    ahead = done[n - p.prefetch_depth] if n >= p.prefetch_depth else 0.0
                ready = max(free["seq"], ahead)
            issued[n] = ready

            # ---- execution -------------------------------------------------
            if c.kind == "compute":
                res, dur = "compute", c.cycles / (p.f_mhz * 1e6) * 1e6
                compute_only += dur
            else:
                res = c.kind
                dur = p.dma_us(c.nbytes)
                if not p.double_buffer:
                    res = "compute"          # serialised against the datapath
            dep = max((done[d] for d in c.deps), default=0.0)
            start = max(ready, dep, free.get(res, 0.0))
            free[res] = start + dur
            busy[res] = busy.get(res, 0.0) + dur
            done[n] = start + dur
            same_kind.setdefault(c.kind, []).append(done[n])
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

def _stream(cfg: bs.BSPNetConfig, hw: bs.HWConfig, batch: int = 1,
            chunks: int = 1) -> List[Command]:
    out: List[Command] = []
    for b in range(batch):
        out.extend(bspnet_commands(cfg, hw, inference=b, first_index=len(out),
                                   chunks=chunks))
    return out


def compare_platforms(cfg: Optional[bs.BSPNetConfig] = None,
                      batch: int = 1) -> List[Dict]:
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    rows = []
    for key, p in PLATFORMS.items():
        rep = PSPLModel(p).run(_stream(cfg, hw, batch, p.input_chunks))
        rows.append(dict(key=key, name=p.name, latency_us=rep.latency_us,
                         compute_us=rep.compute_us,
                         overhead_pct=rep.overhead_pct,
                         throughput=rep.throughput_sps,
                         critical=rep.critical, n_cmd=rep.n_commands))
    return rows


def ring_depth_sweep(cfg: Optional[bs.BSPNetConfig] = None,
                     depths=(1, 2, 4, 8, 16, 32), key: str = "static-a53"
                     ) -> List[Dict]:
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    cmds = _stream(cfg, hw, 1)
    out = []
    for d in depths:
        p = replace(PLATFORMS[key], prefetch_depth=d, ring_slots=max(d, 2))
        rep = PSPLModel(p).run(cmds)
        out.append(dict(depth=d, latency_us=rep.latency_us,
                        overhead_pct=rep.overhead_pct, critical=rep.critical))
    return out


def batch_sweep(cfg: Optional[bs.BSPNetConfig] = None,
                batches=(1, 2, 4, 6, 8, 10),
                keys=("pynq", "static-a53")) -> List[Dict]:
    cfg = cfg or bs.CONFIGS["cfg-6"]
    hw = bs.HW["BSP-Flex"]
    out = []
    for b in batches:
        row = dict(batch=b)
        cmds = _stream(cfg, hw, b)
        for k in keys:
            rep = PSPLModel(PLATFORMS[k]).run(cmds)
            row[k] = rep.throughput_sps
            row[k + "_us"] = rep.per_inference_us
        row["peak"] = 1e6 / (bs.total_cycles(cfg, hw) / (hw.f_mhz * 1e6) * 1e6)
        out.append(row)
    return out


def port_budget(cfg: Optional[bs.BSPNetConfig] = None,
                p: Optional[PSPLConfig] = None) -> Dict:
    """Is the PS-PL link ever the constraint?  (It is not.)"""
    cfg = cfg or bs.CONFIGS["cfg-6"]
    p = p or PLATFORMS["static-a53"]
    hw = bs.HW["BSP-Flex"]
    lat_s = bs.total_cycles(cfg, hw) / (hw.f_mhz * 1e6)
    in_bytes = cfg.N * 4
    rows_out = max(1, cfg.L // (cfg.pool ** cfg.n_mlp_blocks))
    out_bytes = cfg.n_branches * rows_out * cfg.d_m * 4
    need = (in_bytes + out_bytes) / lat_s / 1e9
    return dict(in_bytes=in_bytes, out_bytes=out_bytes, latency_ms=lat_s * 1e3,
                required_gbs=need, available_gbs=p.dma_bw_gbs,
                utilisation=need / p.dma_bw_gbs,
                min_hp_ports=math.ceil(need / p.port_bw_gbs))
