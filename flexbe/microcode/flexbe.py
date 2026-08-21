"""
flexbe.py -- cycle-level simulator of the flexible butterfly engine
===================================================================

Model of the engine in

    X. Liu, R. Wu, P. H. W. Leong, "A Flexible FPGA-based Butterfly Engine for
    Accelerating Signal Processing and Machine Learning".

The simulator executes the real dataflow -- banked RAM, switching network,
butterfly units, per-cycle control -- so the architectural claims of the paper
can be checked rather than assumed:

  Eq. (3)   shift-down storage   a_x = I >> m,  a_y = bsm(I)
  Eq. (2)   read / permute / butterfly / restore / write
  Sec. 3.1  PRS = barrel shifter (R) + subset switch (S), P_f = P_s x P_r
  Alg. 1    per-cycle index vectors and PRS control  (repaired, see below)
  Sec. 3.2  sub-parallelism P_sub for l < 2*P_bu, Eq. (5) interleaving
  Sec. 3.3  integrated bit-reversal write-back, Alg. 3, in exactly N/P cycles
  Sec. 3.4  P_N packing of several sequences into one banked RAM array
  Eq. (10)  cycle model
  Sec. 5.2  Q1.15 with 32-bit accumulation, convergent rounding, saturation

Stage convention
----------------
The engine runs a decimation-in-frequency radix-2 dataflow: stage k pairs
indices differing in bit

    h = n - 1 - k                                  ("hole bit")

so natural-order input leaves the result bit reversed -- which is exactly what
the Sec. 3.3 write-back path exists to undo -- and the subset-switch state is
S = h when h < m and 0 otherwise, identical to Algorithm 1 lines 8-11.

Algorithm 1
-----------
`algorithm1()` below is the single source of per-cycle control in this
simulator.  It implements the *repaired* index construction: the published
lines 5-6 do not yield a conflict-free grouping (see docs/algorithm1_fix.md for
the diagnosis, the exhaustive counterexample and the proof of the replacement).
Lines 7-11 of the published listing are reproduced verbatim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "popcount", "bit_rev", "bit_rev_array", "insert_zero", "bsm", "bsm_array",
    "FixedPointFormat", "BankConflictError", "BankedMemory",
    "PermuteRotateSwitch", "fcs_matrix", "interconnect_cost",
    "CycleControl", "algorithm1", "ButterflySchedule",
    "fft_coefficients", "random_bl_coefficients", "expand_coefficients",
    "coefficients_from_paper_layout", "butterfly_reference",
    "bitrev_schedule", "TransformStats", "FlexBE", "BEArray",
    "cycles_eq10", "layer_cycles", "bitrev_cycles",
]


# ---------------------------------------------------------------------------
# bit-level helpers
# ---------------------------------------------------------------------------

def popcount(x: int) -> int:
    """Number of 1s in the binary representation of x."""
    return int(x).bit_count()


def bit_rev(x: int, n: int) -> int:
    """Reverse the n-bit binary representation of x (bit-rev_n of the paper)."""
    r = 0
    for _ in range(n):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def bit_rev_array(n: int) -> np.ndarray:
    """Vector v with v[i] = bit_rev(i, n)."""
    idx = np.arange(1 << n, dtype=np.int64)
    out = np.zeros(1 << n, dtype=np.int64)
    for b in range(n):
        out |= ((idx >> b) & 1) << (n - 1 - b)
    return out


def insert_zero(x: int, p: int) -> int:
    """ins(x, p): insert a 0 bit at position p."""
    return ((x >> p) << (p + 1)) | (x & ((1 << p) - 1))


def _popcount_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    try:
        return np.bitwise_count(a).astype(np.int64)      # NumPy >= 2.0
    except AttributeError:                               # pragma: no cover
        c, x = np.zeros_like(a), a.copy()
        while np.any(x):
            c += x & 1
            x >>= 1
        return c


def bsm(I: int, m: int) -> int:
    """Bank index of Eq. (3): (popcount(I >> m) + (I mod 2^m)) mod 2^m.

    Equivalently: a RAM row holds 2^m consecutive indices, and row H is stored
    cyclically rotated by popcount(H) banks.  Every statement about
    conflict-freedom in this file follows from that one sentence.
    """
    mask = (1 << m) - 1
    return (popcount(I >> m) + (I & mask)) & mask


def bsm_array(I: np.ndarray, m: int) -> np.ndarray:
    """Vectorised bsm()."""
    I = np.asarray(I, dtype=np.int64)
    mask = (1 << m) - 1
    return (_popcount_array(I >> m) + (I & mask)) & mask


# ---------------------------------------------------------------------------
# arithmetic format -- Sec. 5.2
# ---------------------------------------------------------------------------

@dataclass
class FixedPointFormat:
    """Q1.15 with 32-bit internal accumulation, convergent rounding, saturation.

    Butterfly products are evaluated at full precision (modelling the wide
    accumulator) and only the result is rounded back to the word format.
    """
    word_bits: int = 16
    frac_bits: int = 15
    acc_bits: int = 32
    saturate: bool = True

    @property
    def scale(self) -> float:
        return float(1 << self.frac_bits)

    @property
    def max_val(self) -> float:
        return ((1 << (self.word_bits - 1)) - 1) / self.scale

    @property
    def min_val(self) -> float:
        return -(1 << (self.word_bits - 1)) / self.scale

    def quantise(self, x):
        x = np.asarray(x)
        if np.iscomplexobj(x):
            return self.quantise(x.real) + 1j * self.quantise(x.imag)
        q = np.round(x * self.scale)                     # half-to-even
        if self.saturate:
            q = np.clip(q, -(1 << (self.word_bits - 1)),
                        (1 << (self.word_bits - 1)) - 1)
        return q / self.scale


# ---------------------------------------------------------------------------
# banked data RAM
# ---------------------------------------------------------------------------

class BankConflictError(RuntimeError):
    """Two lanes addressed the same RAM bank in one cycle."""


class BankedMemory:
    """P single-port banks of `depth` words, addressed by (bank, depth) vectors."""

    def __init__(self, num_banks: int, depth: int, dtype=np.complex128):
        self.P, self.depth = int(num_banks), int(depth)
        self.banks = np.zeros((self.P, self.depth), dtype=dtype)
        self.reads = self.writes = 0

    @staticmethod
    def _check(banks: np.ndarray) -> None:
        if len(np.unique(banks)) != len(banks):
            raise BankConflictError(f"bank conflict on {np.asarray(banks).tolist()}")

    def read_vector(self, banks, depths, check: bool = True) -> np.ndarray:
        if check:
            self._check(banks)
        self.reads += 1
        return self.banks[banks, depths]

    def write_vector(self, banks, depths, values, check: bool = True) -> None:
        if check:
            self._check(banks)
        self.writes += 1
        self.banks[banks, depths] = values

    def load_sequence(self, seq: np.ndarray, m: int, base_depth: int = 0) -> None:
        idx = np.arange(len(seq), dtype=np.int64)
        self.banks[bsm_array(idx, m), (idx >> m) + base_depth] = seq

    def read_sequence(self, N: int, m: int, base_depth: int = 0) -> np.ndarray:
        idx = np.arange(N, dtype=np.int64)
        return self.banks[bsm_array(idx, m), (idx >> m) + base_depth].copy()


# ---------------------------------------------------------------------------
# switching -- Sec. 3.1
# ---------------------------------------------------------------------------

def _subset_perm(S: int, m: int) -> np.ndarray:
    """The stride permutation P_s for state S (one of only m states)."""
    P = 1 << m
    pi = np.empty(P, dtype=np.int64)
    for t in range(P // 2):
        pi[2 * t] = insert_zero(t, S)
        pi[2 * t + 1] = pi[2 * t] + (1 << S)
    return pi


class PermuteRotateSwitch:
    """PRS: barrel shifter (rotation R) followed by a subset switch (state S).

    read_map(R, S)[j] = (pi_S[j] + R) mod P is the bank that butterfly slot j
    reads, so the PRS realises the FCS permutation P_f = P_s x P_r at
    Theta(m*P_bu) instead of Theta(P_bu^2) interconnect.
    """

    def __init__(self, m: int):
        self.m, self.P = int(m), 1 << int(m)
        self._perm = {S: _subset_perm(S, self.m) for S in range(max(1, self.m))}

    def perm(self, S: int) -> np.ndarray:
        return self._perm[S]

    def read_map(self, R: int, S: int) -> np.ndarray:
        return (self._perm[S] + R) % self.P

    def forward(self, D: np.ndarray, R: int, S: int) -> np.ndarray:
        """Bank order -> butterfly-slot order (PRS-A)."""
        return D[self.read_map(R, S)]

    def inverse(self, X: np.ndarray, R: int, S: int) -> np.ndarray:
        """Butterfly-slot order -> bank order (PRS-B)."""
        D = np.empty_like(X)
        D[self.read_map(R, S)] = X
        return D

    def mux_count(self, data_width: int = 1) -> int:
        """2:1 muxes: m*P for the barrel shifter, (m+1)*2^(m-1) for the subset."""
        return data_width * (self.m * self.P + (self.m + 1) * (self.P // 2))


def fcs_matrix(indices: np.ndarray, m: int) -> np.ndarray:
    """Baseline crossbar control: P_f[j,k] = 1 iff k = bsm(I[j])."""
    P = 1 << m
    Pf = np.zeros((P, P), dtype=np.int8)
    Pf[np.arange(P), bsm_array(indices, m)] = 1
    return Pf


def interconnect_cost(P_bu: int, data_width: int = 1) -> Dict[str, float]:
    """Model behind Fig. 9.  FCS: 2P_bu ports x (2P_bu - 1) 2:1 muxes (Eq. 4)."""
    m = (2 * P_bu).bit_length() - 1
    P = 1 << m
    fcs = data_width * P * (P - 1)
    prs = PermuteRotateSwitch(m).mux_count(data_width)
    return {"ports": P, "m": m, "fcs_mux2": fcs, "prs_mux2": prs,
            "fcs_pipeline_regs": P * (P - 1) * 16,
            "prs_pipeline_regs": P * m * 16, "ratio": fcs / prs}


# ---------------------------------------------------------------------------
# Algorithm 1 -- the single source of per-cycle control
# ---------------------------------------------------------------------------

@dataclass
class CycleControl:
    """Everything the flex-butterfly management module emits in one cycle."""
    stage: int                    # k
    hole: int                     # h = n - 1 - k
    R: int                        # barrel-shifter rotation
    S: int                        # subset-switch state
    indices: np.ndarray           # I^i, 2*P_bu element indices in slot order
    depth_by_bank: np.ndarray     # Addr^i, one read depth per RAM bank
    coeff_index: np.ndarray       # P_bu twiddle / weight addresses


def algorithm1(n: int, P_bu: int, k: int, j: int) -> CycleControl:
    """Index vectors and PRS control for cycle j of stage k (repaired listing).

        h <- n - 1 - k
        if h < m:                          both operands share RAM row H
            H <- j
            I^i[2t]   <- (H << m) + ins(t, h)              t = 0..P_bu-1
            S^i       <- h
        else:                              operands in rows H and H + 2^(h-m)
            e <- j mod 2,  H <- ins(j >> 1, h - m)
            I^i[2t]   <- (H << m) + 2t + e
            S^i       <- 0
        both:
            I^i[2t+1] <- I^i[2t] + 2^h
            R^i       <- bsm(I^i[0])                       (published line 7)

    Conflict freedom:

      h < m   one cycle reads the whole of row H, whose banks are
              (popcount(H) + low) mod 2^m for low = 0..2^m-1 -- all distinct.
              Slot j reads bank (pi_h[j] + popcount(H)) mod 2^m, i.e. a fixed
              stride permutation then a rotation: P_f = P_s x P_r with S = h.

      h >= m  bit h-m of H is 0 by construction, so
              popcount(H + 2^(h-m)) = popcount(H) + 1 and slot j reads bank
              (popcount(H) + e + j) mod 2^m -- a pure rotation, S = 0.

    Each branch emits exactly N/2^m = N/(2*P_bu) cycles per stage and visits
    every index once.  S^i also equals the published closed form
    (n-k-1 on (n-m) <= k <= (n-2), else 0).
    """
    m = (2 * P_bu).bit_length() - 1
    P = 1 << m
    if not (0 <= k < n and 0 <= j < (1 << (n - m))):
        raise ValueError(f"stage/cycle out of range for N=2^{n}, P_bu={P_bu}")
    h = n - 1 - k
    t = np.arange(P // 2, dtype=np.int64)
    I = np.empty(P, dtype=np.int64)

    if h < m:                                    # pair inside one row
        H = j
        I[0::2] = (H << m) + (((t >> h) << (h + 1)) | (t & ((1 << h) - 1)))
        S = h
    else:                                        # pair spans two rows
        e, step = j & 1, 1 << (h - m)
        r = j >> 1
        H = ((r >> (h - m)) << (h - m + 1)) | (r & (step - 1))
        I[0::2] = (H << m) + 2 * t + e
        I[1::2] = ((H + step) << m) + 2 * t + e
        S = 0
    I[1::2] = I[0::2] + (1 << h)

    banks = bsm_array(I, m)
    depth_by_bank = np.empty(P, dtype=np.int64)
    depth_by_bank[banks] = I >> m
    a = I[0::2]
    coeff = ((a >> (h + 1)) << h) | (a & ((1 << h) - 1))
    return CycleControl(k, h, bsm(int(I[0]), m), S, I, depth_by_bank, coeff)


class ButterflySchedule:
    """Every cycle of an N-point transform, emitted by algorithm1()."""

    def __init__(self, n: int, P_bu: int, P_sub: int = 1,
                 validate: bool = False):
        assert P_sub >= 1 and (P_sub & (P_sub - 1)) == 0
        self.n, self.P_bu = int(n), int(P_bu)
        self.m = (2 * P_bu).bit_length() - 1
        self.N, self.P = 1 << self.n, 1 << self.m
        if self.N < self.P:
            raise ValueError("transform shorter than the datapath: use P_sub")
        self.P_sub = P_sub
        self.s = int(math.log2(P_sub))
        self.n_stages = self.n - self.s
        self.cycles_per_stage = self.N // self.P
        self.stages: List[List[CycleControl]] = [
            [algorithm1(self.n, self.P_bu, k, j)
             for j in range(self.cycles_per_stage)]
            for k in range(self.n_stages)]
        if validate:
            self.validate()

    @property
    def total_cycles(self) -> int:
        return self.n_stages * self.cycles_per_stage

    def all_cycles(self) -> Iterator[CycleControl]:
        for st in self.stages:
            yield from st

    def published_subset_state(self, k: int) -> int:
        """S^i as printed in Algorithm 1, lines 8-11."""
        return (self.n - k - 1) if (self.n - self.m) <= k <= (self.n - 2) else 0

    def validate(self) -> None:
        """Assert conflict freedom, coverage, pairing and PRS == FCS."""
        prs = PermuteRotateSwitch(self.m)
        for k, stage in enumerate(self.stages):
            seen = np.zeros(self.N, dtype=bool)
            for c in stage:
                banks = bsm_array(c.indices, self.m)
                if len(np.unique(banks)) != self.P:
                    raise BankConflictError(f"stage {k}: bank conflict")
                if seen[c.indices].any():
                    raise AssertionError(f"stage {k}: element visited twice")
                seen[c.indices] = True
                if not np.all(c.indices[1::2] - c.indices[0::2] == (1 << c.hole)):
                    raise AssertionError("bad butterfly pairing")
                if np.any((c.indices[0::2] >> c.hole) & 1):
                    raise AssertionError("lower index must have hole bit 0")
                if not np.array_equal(prs.read_map(c.R, c.S), banks):
                    raise AssertionError("PRS != FCS permutation")
                if c.R != bsm(int(c.indices[0]), self.m):
                    raise AssertionError("R != bsm(I[0])")
                if c.S != self.published_subset_state(k):
                    raise AssertionError("S != published control law")
            if not seen.all():
                raise AssertionError(f"stage {k}: incomplete coverage")


# ---------------------------------------------------------------------------
# coefficients: FFT twiddles and butterfly-linear weights
# ---------------------------------------------------------------------------

def fft_coefficients(l: int, inverse: bool = False,
                     fmt: Optional[FixedPointFormat] = None) -> np.ndarray:
    """DIF twiddles, shape (log2 l, l//2, 2, 2); butterfly o is [[1,1],[W,-W]]."""
    nl = int(math.log2(l))
    assert 1 << nl == l
    C = np.zeros((nl, l // 2, 2, 2), dtype=np.complex128)
    sign = 1.0 if inverse else -1.0
    for k in range(nl):
        h = nl - 1 - k
        t = np.arange(1 << h, dtype=np.int64)
        W = np.exp(sign * 2j * np.pi * (t * (l >> (h + 1))) / l)
        if fmt is not None:
            W = fmt.quantise(W)
        blk = C[k].reshape(l >> (h + 1), 1 << h, 2, 2)
        blk[:, :, 0, 0] = blk[:, :, 0, 1] = 1.0
        blk[:, :, 1, 0] = W[None, :]
        blk[:, :, 1, 1] = -W[None, :]
    return C


def random_bl_coefficients(l: int, rng: np.random.Generator, scale: float = 1.0,
                           real: bool = True,
                           fmt: Optional[FixedPointFormat] = None) -> np.ndarray:
    """Trained-style butterfly-linear weights, shape (log2 l, l//2, 2, 2)."""
    shape = (int(math.log2(l)), l // 2, 2, 2)
    C = (rng.normal(0.0, scale / math.sqrt(2.0), shape) if real else
         rng.normal(0.0, scale / 2.0, shape) + 1j * rng.normal(0.0, scale / 2.0, shape))
    return fmt.quantise(C) if fmt is not None else C


def expand_coefficients(C: np.ndarray, P_sub: int) -> np.ndarray:
    """Map length-l coefficients onto the interleaved N = P_sub*l datapath."""
    if P_sub == 1:
        return C
    s = int(math.log2(P_sub))
    nl, half = C.shape[0], C.shape[1]
    l, N = 2 * half, P_sub * 2 * half
    out = np.zeros((nl, N // 2, 2, 2), dtype=C.dtype)
    for k in range(nl):
        hp = nl - 1 - k
        src = C[k].reshape(l >> (hp + 1), 1 << hp, 2, 2)
        out[k] = np.repeat(src, P_sub, axis=1).reshape(N // 2, 2, 2)
    return out


def coefficients_from_paper_layout(W: np.ndarray) -> np.ndarray:
    """[2, 2, l/2, log2 l] of Table 4 -> (log2 l, l/2, 2, 2)."""
    return np.transpose(np.asarray(W), (3, 2, 0, 1)).copy()


def butterfly_reference(comp: np.ndarray, C: np.ndarray, P_sub: int = 1,
                        stage_scale: Optional[float] = None,
                        fmt: Optional[FixedPointFormat] = None) -> np.ndarray:
    """Golden datapath: no memory, no switching, same arithmetic."""
    comp = np.atleast_2d(comp)
    n_comp, N = comp.shape
    n, s = int(math.log2(N)), int(math.log2(P_sub))
    n_stages = C.shape[0]
    assert n_stages == n - s
    y = comp.astype(np.result_type(comp.dtype, C.dtype, np.complex128)).copy()
    for k in range(n_stages):
        h = n - 1 - k
        v = y.reshape(n_comp, N >> (h + 1), 2, 1 << h)
        lo, hi = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
        Ck = C[k].reshape(N >> (h + 1), 1 << h, 2, 2)
        nl = Ck[None, :, :, 0, 0] * lo + Ck[None, :, :, 0, 1] * hi
        nh = Ck[None, :, :, 1, 0] * lo + Ck[None, :, :, 1, 1] * hi
        if stage_scale is not None:
            nl, nh = nl * stage_scale, nh * stage_scale
        if fmt is not None:
            nl, nh = fmt.quantise(nl), fmt.quantise(nh)
        v[:, :, 0, :], v[:, :, 1, :] = nl, nh
        y = v.reshape(n_comp, N)
    return y


# ---------------------------------------------------------------------------
# bit-reversal write-back -- Sec. 3.3, Algorithm 3
# ---------------------------------------------------------------------------

def bitrev_schedule(n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithm 3: per-cycle index vectors (j, k), shape (N/P, P).

    Lane a reads FlexBE RAM bank a at depth j[a] >> m and writes global-buffer
    bank k[a] mod 2^m; the destination-bank law is Eq. (7),
    bit-rev_m((a + u) mod P).  Requires the long-transform regime N >= P^2.
    """
    if n < 2 * m:
        raise ValueError("Algorithm 3 requires n >= 2m (N >= P^2)")
    P = 1 << m
    q = min(m, n - 2 * m)
    r = n - 2 * m - q
    a_vec = np.arange(P, dtype=np.int64)
    rows = []
    for u in range(P):
        b_vec = (a_vec + u) & (P - 1)
        for rr in range(1 << r):
            for qq in range(1 << q):
                high = (b_vec << (q + r)) | (qq << r) | rr
                x = (a_vec - _popcount_array(high)) & (P - 1)
                rows.append((high << m) | x)
    j = np.asarray(rows, dtype=np.int64)
    return j, bit_rev_array(n)[j]


def _bitrev_pairs(n: int, m: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Algorithm 3 where it applies, else a plain N/P-cycle model."""
    if n >= 2 * m:
        j, k = bitrev_schedule(n, m)
        return j, k, True
    N, P = 1 << n, 1 << m
    j = np.arange(N, dtype=np.int64).reshape(max(1, N // P), min(N, P))
    return j, bit_rev_array(n)[j], False


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

@dataclass
class TransformStats:
    N: int = 0                      # engine computational length
    l: int = 0                      # logical transform length
    P_sub: int = 1
    P_N: int = 0
    n_stages: int = 0
    butterfly_cycles: int = 0
    bitrev_cycles: int = 0
    ram_depth: int = 0
    scale: float = 1.0

    @property
    def total_cycles(self) -> int:
        return self.butterfly_cycles + self.bitrev_cycles

    def __str__(self) -> str:
        return (f"N={self.N} l={self.l} P_sub={self.P_sub} P_N={self.P_N} "
                f"stages={self.n_stages} bfly={self.butterfly_cycles} "
                f"bitrev={self.bitrev_cycles} total={self.total_cycles}")


class FlexBE:
    """One butterfly engine with P_bu radix-2 units.

    Compile-time: P_bu, arithmetic format.  Runtime (Table 5): l (or N),
    P_sub and P_N, all supplied per `transform()` call.
    """

    def __init__(self, P_bu: int = 16, fmt: Optional[FixedPointFormat] = None,
                 check_conflicts: bool = True, name: str = "FlexBE"):
        assert P_bu >= 1 and (P_bu & (P_bu - 1)) == 0
        self.P_bu, self.P = int(P_bu), 2 * int(P_bu)
        self.m = self.P.bit_length() - 1
        self.fmt, self.name = fmt, name
        self.check_conflicts = check_conflicts
        self.prs = PermuteRotateSwitch(self.m)
        self.cycles = 0
        self._lane = np.arange(self.P, dtype=np.int64)
        self._sched: Dict[Tuple[int, int], ButterflySchedule] = {}

    # -- geometry ----------------------------------------------------------
    def geometry(self, l: int) -> Tuple[int, int]:
        """(N, P_sub) for a logical length l -- Algorithm 2 initialisation."""
        if l < self.P:
            if self.P % l:
                raise ValueError("l must divide 2*P_bu")
            return self.P, self.P // l
        return l, 1

    def schedule(self, n: int, P_sub: int = 1,
                 validate: bool = False) -> ButterflySchedule:
        key = (n, P_sub)
        if key not in self._sched:
            self._sched[key] = ButterflySchedule(n, self.P_bu, P_sub, validate)
        elif validate:
            self._sched[key].validate()
        return self._sched[key]

    # -- Eq. (5) interleaving ---------------------------------------------
    def interleave(self, x: np.ndarray, P_sub: int) -> np.ndarray:
        if P_sub == 1:
            return x
        n_seq, l = x.shape
        n_comp = math.ceil(n_seq / P_sub)
        pad = np.zeros((n_comp * P_sub, l), dtype=x.dtype)
        pad[:n_seq] = x
        return pad.reshape(n_comp, P_sub, l).transpose(0, 2, 1).reshape(n_comp, -1)

    def deinterleave(self, comp: np.ndarray, l: int, P_sub: int,
                     n_seq: int) -> np.ndarray:
        if P_sub == 1:
            return comp[:n_seq]
        out = comp.reshape(comp.shape[0], l, P_sub).transpose(0, 2, 1)
        return out.reshape(-1, l)[:n_seq]

    # -- main entry point --------------------------------------------------
    def transform(self, x, coeffs: Optional[np.ndarray] = None,
                  bitrev: bool = False, stage_scale: Optional[float] = None,
                  datapath: str = "cycle",
                  validate: bool = False) -> Tuple[np.ndarray, TransformStats]:
        """Run P_N transforms of length l packed into the engine RAM.

        x            (P_N, l) or (l,), natural order
        coeffs       (log2 l, l//2, 2, 2); None -> FFT twiddles
        bitrev       apply the Sec. 3.3 write-back, returning natural order
        stage_scale  per-stage gain (use 0.5 for fixed-point FFTs)
        datapath     'cycle'  full banked-memory + PRS simulation
                     'vector' golden datapath, cycles still modelled
        """
        x = np.atleast_2d(np.asarray(x))
        n_seq, l = x.shape
        N, P_sub = self.geometry(l)
        n, nl = int(math.log2(N)), int(math.log2(l))
        sched = self.schedule(n, P_sub, validate=validate)

        if coeffs is None:
            coeffs = fft_coefficients(l, fmt=self.fmt)
        if coeffs.shape[0] != nl:
            raise ValueError("coefficient array must have log2(l) stages")
        C = expand_coefficients(coeffs, P_sub)

        comp = self.interleave(x.astype(np.complex128), P_sub)
        n_comp = comp.shape[0]
        depth_per_seq = N // self.P
        st = TransformStats(N=N, l=l, P_sub=P_sub, P_N=n_seq,
                            n_stages=sched.n_stages,
                            ram_depth=n_comp * depth_per_seq,
                            scale=1.0 if stage_scale is None
                            else stage_scale ** sched.n_stages)

        if datapath == "vector":
            out = butterfly_reference(comp, C, P_sub, stage_scale, self.fmt)
            st.butterfly_cycles = sched.total_cycles * n_comp
        elif datapath == "cycle":
            out, st.butterfly_cycles = self._run(comp, C, sched, depth_per_seq,
                                                 stage_scale)
        else:
            raise ValueError(f"unknown datapath '{datapath}'")

        if bitrev:
            out, st.bitrev_cycles = self._bitrev(out, n, P_sub, l, datapath)
        self.cycles += st.total_cycles
        return self.deinterleave(out, l, P_sub, n_seq), st

    # -- cycle-level datapath, Eq. (2) ------------------------------------
    def _run(self, comp, C, sched, depth_per_seq, stage_scale):
        P, m = self.P, self.m
        n_comp, N = comp.shape
        mem = BankedMemory(P, n_comp * depth_per_seq)
        for c in range(n_comp):
            mem.load_sequence(comp[c], m, c * depth_per_seq)

        lane, fmt, check = self._lane, self.fmt, self.check_conflicts
        cycles = 0
        for k, stage in enumerate(sched.stages):
            Ck = C[k]
            for ctrl in stage:
                rmap = self.prs.read_map(ctrl.R, ctrl.S)
                W = Ck[ctrl.coeff_index]
                for c in range(n_comp):
                    addr = ctrl.depth_by_bank + c * depth_per_seq
                    D = mem.read_vector(lane, addr, check=check)
                    X = D[rmap]                                   # PRS-A
                    a, b = X[0::2], X[1::2]
                    na = W[:, 0, 0] * a + W[:, 0, 1] * b
                    nb = W[:, 1, 0] * a + W[:, 1, 1] * b
                    if stage_scale is not None:
                        na, nb = na * stage_scale, nb * stage_scale
                    if fmt is not None:
                        na, nb = fmt.quantise(na), fmt.quantise(nb)
                    Xp = np.empty_like(X)
                    Xp[0::2], Xp[1::2] = na, nb
                    Dp = np.empty_like(Xp)
                    Dp[rmap] = Xp                                 # PRS-B
                    mem.write_vector(lane, addr, Dp, check=check)
                    cycles += 1
        out = np.stack([mem.read_sequence(N, m, c * depth_per_seq)
                        for c in range(n_comp)])
        return out, cycles

    # -- bit reversal ------------------------------------------------------
    def _bitrev(self, comp, n, P_sub, l, datapath):
        n_comp, N = comp.shape
        if P_sub > 1:                       # each interleaved stream separately
            s = int(math.log2(P_sub))
            rev = bit_rev_array(n - s)
            out = comp.reshape(n_comp, l, P_sub)[:, rev, :].reshape(n_comp, N)
            return out, max(1, N // self.P) * n_comp
        j, kk, exact = _bitrev_pairs(n, self.m)
        if exact and datapath == "cycle" and self.check_conflicts:
            for jv, kv in zip(j, kk):
                if len(np.unique(bsm_array(jv, self.m))) != self.P:
                    raise BankConflictError("bit-reversal read conflict")
                if len(np.unique(kv & (self.P - 1))) != self.P:
                    raise BankConflictError("bit-reversal write conflict")
        out = np.zeros_like(comp)
        for c in range(n_comp):
            out[c, kk.ravel()] = comp[c, j.ravel()]
        return out, j.shape[0] * n_comp


class BEArray:
    """P_be engines sharing a global data buffer (Fig. 7); latency = slowest."""

    def __init__(self, P_be: int = 4, P_bu: int = 16,
                 fmt: Optional[FixedPointFormat] = None,
                 check_conflicts: bool = True):
        self.P_be, self.P_bu = int(P_be), int(P_bu)
        self.engines = [FlexBE(P_bu, fmt, check_conflicts, f"BE{i}")
                        for i in range(self.P_be)]
        self.cycles = 0

    def transform(self, x, coeffs=None, bitrev: bool = False,
                  stage_scale: Optional[float] = None,
                  datapath: str = "vector") -> Tuple[np.ndarray, TransformStats]:
        x = np.atleast_2d(np.asarray(x))
        chunks = [c for c in np.array_split(np.arange(x.shape[0]), self.P_be)
                  if len(c)]
        outs, stats = [], []
        for eng, ids in zip(self.engines, chunks):
            y, s = eng.transform(x[ids], coeffs, bitrev, stage_scale, datapath)
            outs.append(y)
            stats.append(s)
        agg = TransformStats(N=stats[0].N, l=stats[0].l, P_sub=stats[0].P_sub,
                             P_N=x.shape[0], n_stages=stats[0].n_stages,
                             scale=stats[0].scale,
                             ram_depth=max(s.ram_depth for s in stats))
        agg.butterfly_cycles = max(s.butterfly_cycles for s in stats)
        agg.bitrev_cycles = max(s.bitrev_cycles for s in stats)
        self.cycles += agg.total_cycles
        return np.concatenate(outs, axis=0), agg


# ---------------------------------------------------------------------------
# performance model -- Eq. (10)
# ---------------------------------------------------------------------------

def cycles_eq10(P_N: int, N: int, P_sub: int, P_be: int, P_bu: int) -> float:
    """cycles ~= P_N*N / (2*P_sub*P_be*P_bu) * log2(N/P_sub).

    P_N is the number of length-l input sequences and N the engine
    computational length max(l, 2*P_bu); with that reading the expression is
    consistent for every P_sub.
    """
    return P_N * N / (2 * P_sub * P_be * P_bu) * math.log2(N / P_sub)


def layer_cycles(n_seq: int, l: int, P_be: int, P_bu: int) -> float:
    """Cycles for n_seq transforms of length l on a P_be x P_bu array."""
    P = 2 * P_bu
    P_sub, N = (P // l, P) if l < P else (1, l)
    return cycles_eq10(n_seq, N, P_sub, P_be, P_bu)


def bitrev_cycles(n_seq: int, N: int, P_be: int, P_bu: int) -> float:
    """Bit-reversal write-back: N/P cycles per sequence (Sec. 3.3)."""
    return n_seq * N / (2 * P_bu) / P_be
