"""
bspnet.py -- BSPNet, the application layer that sits on top of FlexBE
====================================================================

Implements the co-designed model of Sec. 4:

  * cyclostationary feature extraction, Eq. (1):
        S2,S4,S6,S8 = s^2, s^4, s^6, s^8      F2,F4,F6,F8 = FFT{S2..S8}
    (four N = 32768-point FFTs executed on the butterfly engines),
  * eight branch models built from butterfly-linear (BL) layers:
        feature expansion (length d_in, expansion d_m/d_in)
        3 x [ MLP-x-0, MLP-x-1, NormPool ]     (length d_m)
    where NormPool = LayerNorm -> shortcut add -> ReLU -> max-pool 8,
  * concatenation and an output classification layer,
  * the eight design-space configurations of Table 7 and the Eq. (10) cycle
    model that produces the latency numbers of Figs. 11/14/15.

Weights are random unless supplied: this is a *hardware* simulator, so the
purpose of the forward pass is to show that one FlexBE datapath executes both
the long FFTs and every BL layer, and that the engine result matches a golden
NumPy model bit for bit (float) or within quantisation error (fxp16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from flexbe import (BEArray, FixedPointFormat, bitrev_cycles,
                    butterfly_reference, expand_coefficients, fft_coefficients,
                    layer_cycles, random_bl_coefficients)


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------

@dataclass
class BSPNetConfig:
    """One column of Table 7."""
    name: str
    d_in: int
    d_m: int
    L: int
    n_mlp_blocks: int
    bit_reversal: bool
    N: int = 32768           # input signal length
    n_branches: int = 8
    n_classes: int = 8
    pool: int = 8            # NormPool max-pool factor


CONFIGS: Dict[str, BSPNetConfig] = {
    "cfg-1": BSPNetConfig("cfg-1", d_in=8,  d_m=64, L=512,  n_mlp_blocks=3, bit_reversal=True),
    "cfg-2": BSPNetConfig("cfg-2", d_in=8,  d_m=64, L=512,  n_mlp_blocks=2, bit_reversal=True),
    "cfg-3": BSPNetConfig("cfg-3", d_in=4,  d_m=32, L=1024, n_mlp_blocks=3, bit_reversal=True),
    "cfg-4": BSPNetConfig("cfg-4", d_in=4,  d_m=32, L=1024, n_mlp_blocks=3, bit_reversal=False),
    "cfg-5": BSPNetConfig("cfg-5", d_in=8,  d_m=32, L=512,  n_mlp_blocks=3, bit_reversal=True),
    "cfg-6": BSPNetConfig("cfg-6", d_in=8,  d_m=32, L=1024, n_mlp_blocks=3, bit_reversal=True),
    "cfg-7": BSPNetConfig("cfg-7", d_in=16, d_m=32, L=1024, n_mlp_blocks=3, bit_reversal=True),
    "cfg-8": BSPNetConfig("cfg-8", d_in=32, d_m=32, L=1024, n_mlp_blocks=3, bit_reversal=True),
}


@dataclass
class HWConfig:
    """Butterfly-engine array parameters (Sec. 5.1.2)."""
    name: str = "FlexBE"
    P_be: int = 4
    P_bu: int = 16
    f_mhz: float = 300.0
    zero_pad_short: bool = False   # True models BE-base: no sub-parallelism


HW = {
    # BE-base-1: P_bu = 4 inherited from FABNet, P_be = 16 (same 64 BUs)
    "BE-base-1": HWConfig("BE-base-1", P_be=16, P_bu=4,  f_mhz=210.0, zero_pad_short=True),
    # BE-base-2: P_bu = 16, short transforms must be zero padded to 2*P_bu
    "BE-base-2": HWConfig("BE-base-2", P_be=4,  P_bu=16, f_mhz=115.0, zero_pad_short=True),
    # FlexBE: same BU count as BE-base-2 but with P_sub / P_N support
    "FlexBE":    HWConfig("FlexBE",    P_be=4,  P_bu=16, f_mhz=310.0, zero_pad_short=False),
    # the integrated BSP-Flex system runs at 300 MHz (Table 11)
    "BSP-Flex":  HWConfig("BSP-Flex",  P_be=4,  P_bu=16, f_mhz=300.0, zero_pad_short=False),
}


# ---------------------------------------------------------------------------
# cycle model
# ---------------------------------------------------------------------------

def _bl_cycles(n_seq: int, l: int, hw: HWConfig) -> float:
    """Cycles for n_seq BL/FFT transforms of length l on this engine array.

    A BE-base style engine cannot process l < 2*P_bu, so the vectors are zero
    padded to 2*P_bu (limitation 1 of Sec. 3) -- that is what zero_pad_short
    models.  FlexBE instead folds 2*P_bu/l transforms into one datapath pass.
    """
    P = 2 * hw.P_bu
    if hw.zero_pad_short and l < P:
        return layer_cycles(n_seq, P, hw.P_be, hw.P_bu)
    return layer_cycles(n_seq, l, hw.P_be, hw.P_bu)


def cycle_breakdown(cfg: BSPNetConfig, hw: HWConfig) -> List[Tuple[str, float]]:
    """Per-module cycle estimate for one single-batch inference."""
    N = cfg.N
    items: List[Tuple[str, float]] = []

    items.append(("CC extraction: 4 x %d-point FFT" % N,
                  _bl_cycles(4, N, hw)))
    if cfg.bit_reversal:
        items.append(("bit-reversal write-back",
                      bitrev_cycles(4, N, hw.P_be, hw.P_bu)))

    rows = N // cfg.d_in
    exp_seq = rows * (cfg.d_m // cfg.d_in)
    items.append(("8 x feature expansion (len %d, x%d)"
                  % (cfg.d_in, cfg.d_m // cfg.d_in),
                  cfg.n_branches * _bl_cycles(exp_seq, cfg.d_in, hw)))

    for blk in range(cfg.n_mlp_blocks):
        rows_b = max(1, cfg.L // (cfg.pool ** blk))
        items.append(("8 x MLP-%d-{0,1} (len %d, %d rows)"
                      % (blk, cfg.d_m, rows_b),
                      cfg.n_branches * 2 * _bl_cycles(rows_b, cfg.d_m, hw)))
    return items


def total_cycles(cfg: BSPNetConfig, hw: HWConfig) -> float:
    return sum(c for _, c in cycle_breakdown(cfg, hw))


def latency_ms(cfg: BSPNetConfig, hw: HWConfig) -> float:
    return total_cycles(cfg, hw) / (hw.f_mhz * 1e6) * 1e3


def throughput_sps(cfg: BSPNetConfig, hw: HWConfig) -> float:
    return 1.0 / (latency_ms(cfg, hw) * 1e-3)


# ---------------------------------------------------------------------------
# functional model
# ---------------------------------------------------------------------------

def _relu(x):
    return np.maximum(x, 0.0)


def _layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def _maxpool_rows(x, factor):
    rows = x.shape[0]
    factor = min(factor, rows)
    keep = (rows // factor) * factor
    return x[:keep].reshape(-1, factor, x.shape[1]).max(axis=1)


@dataclass
class BranchWeights:
    expansion: List[np.ndarray]          # d_m/d_in sets of length-d_in weights
    mlp: List[Tuple[np.ndarray, np.ndarray]]


class BSPNet:
    """BSPNet with every butterfly transform executed on a FlexBE array."""

    def __init__(self, cfg: BSPNetConfig, hw: HWConfig,
                 fmt: Optional[FixedPointFormat] = None,
                 seed: int = 0, datapath: str = "vector"):
        self.cfg = cfg
        self.hw = hw
        self.fmt = fmt
        self.datapath = datapath
        self.rng = np.random.default_rng(seed)
        self.array = BEArray(hw.P_be, hw.P_bu, fmt)
        self.fft_coeffs = fft_coefficients(cfg.N, fmt=fmt)
        n_exp = cfg.d_m // cfg.d_in
        self.branches: List[BranchWeights] = []
        for _ in range(cfg.n_branches):
            exp = [random_bl_coefficients(cfg.d_in, self.rng, scale=1.0, fmt=fmt)
                   for _ in range(n_exp)]
            mlp = [(random_bl_coefficients(cfg.d_m, self.rng, scale=1.0, fmt=fmt),
                    random_bl_coefficients(cfg.d_m, self.rng, scale=1.0, fmt=fmt))
                   for _ in range(cfg.n_mlp_blocks)]
            self.branches.append(BranchWeights(exp, mlp))
        rows_out = max(1, cfg.L // (cfg.pool ** cfg.n_mlp_blocks))
        fan = cfg.n_branches * rows_out * cfg.d_m
        self.W_cls = self.rng.normal(0, 1 / math.sqrt(fan), (fan, cfg.n_classes))
        self.b_cls = np.zeros(cfg.n_classes)
        self.cycles = 0

    # -- primitives --------------------------------------------------------
    def _bl(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Apply a butterfly-linear layer row-wise on the engine array."""
        y, st = self.array.transform(x, coeffs, bitrev=False,
                                     datapath=self.datapath)
        self.cycles += st.total_cycles
        return np.real_if_close(y).real if not np.iscomplexobj(x) else y

    def _fft(self, x: np.ndarray) -> np.ndarray:
        y, st = self.array.transform(x, self.fft_coeffs,
                                     bitrev=self.cfg.bit_reversal,
                                     datapath=self.datapath)
        self.cycles += st.total_cycles
        return y

    # -- Eq. (1) -----------------------------------------------------------
    def features(self, s: np.ndarray) -> Dict[str, np.ndarray]:
        """Cyclostationary features of Eq. (1) from one complex IQ record."""
        S = {p: s ** p for p in (2, 4, 6, 8)}
        stack = np.stack([S[2], S[4], S[6], S[8]])
        F = self._fft(stack)
        out = {f"S{p}": S[p] for p in (2, 4, 6, 8)}
        for i, p in enumerate((2, 4, 6, 8)):
            out[f"F{p}"] = F[i]
        return out

    # -- one branch --------------------------------------------------------
    def branch(self, x: np.ndarray, w: BranchWeights) -> np.ndarray:
        cfg = self.cfg
        rows = x.reshape(-1, cfg.d_in)
        expanded = np.concatenate([self._bl(rows, c) for c in w.expansion], axis=1)
        factor = max(1, expanded.shape[0] // cfg.L)
        h = _maxpool_rows(expanded, factor)
        for (W0, W1) in w.mlp:
            y = self._bl(_relu(self._bl(h, W0)), W1)
            y = _relu(_layer_norm(y) + h)
            h = _maxpool_rows(y, cfg.pool)
        return h

    # -- end to end --------------------------------------------------------
    def forward(self, s: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        assert s.shape[-1] == cfg.N, f"expected {cfg.N} IQ samples"
        feats = self.features(s)
        order = ["S2", "S4", "S6", "S8", "F2", "F4", "F6", "F8"]
        outs = []
        for name, w in zip(order, self.branches):
            v = np.abs(feats[name])                 # magnitude computation block
            peak = np.max(v)
            if peak > 0:
                v = v / peak
            outs.append(self.branch(v, w).ravel())
        flat = np.concatenate(outs)
        return flat @ self.W_cls + self.b_cls

    # -- golden model ------------------------------------------------------
    def features_reference(self, s: np.ndarray) -> Dict[str, np.ndarray]:
        S = {p: s ** p for p in (2, 4, 6, 8)}
        out = {f"S{p}": S[p] for p in (2, 4, 6, 8)}
        for p in (2, 4, 6, 8):
            out[f"F{p}"] = np.fft.fft(S[p])
        return out


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------

def design_space_table(hw_names=("BE-base-1", "BE-base-2", "FlexBE")) -> str:
    head = f"{'config':8s} {'d_in':>5s} {'d_m':>4s} {'L':>5s} {'blk':>4s} {'brev':>5s}"
    for h in hw_names:
        head += f" | {h + ' cyc':>16s} {'ms':>7s}"
    lines = [head, "-" * len(head)]
    for cfg in CONFIGS.values():
        row = (f"{cfg.name:8s} {cfg.d_in:5d} {cfg.d_m:4d} {cfg.L:5d} "
               f"{cfg.n_mlp_blocks:4d} {str(cfg.bit_reversal):>5s}")
        for h in hw_names:
            hwc = HW[h]
            row += f" | {total_cycles(cfg, hwc):16,.0f} {latency_ms(cfg, hwc):7.3f}"
        lines.append(row)
    return "\n".join(lines)
