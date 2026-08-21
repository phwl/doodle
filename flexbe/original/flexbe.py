"""
flexbe.py -- a functional + cycle-level simulator of the FlexBE butterfly engine
===============================================================================

Model of the architecture in

    X. Liu, R. Wu, P. H. W. Leong, "A Flexible FPGA-based Butterfly Engine for
    Accelerating Signal Processing and Machine Learning".

What is modelled
----------------
* "shift-down" (skewed) banked data RAM, Eq. (3):
      a_x = I >> m,   a_y = bsm(I) = (popcount(I >> m) + (I mod 2^m)) mod 2^m
* the conflict-free per-cycle index vectors I^i and the resulting read/write
  address vectors,
* the permute-rotate switch (PRS, Sec. 3.1) as a barrel shifter (rotation R^i)
  followed by a subset switch (state S^i), together with the baseline fully
  connected switch (FCS) so that the decomposition P_f = P_s x P_r can be
  checked cycle by cycle,
* radix-2 butterfly units carrying either FFT twiddle factors or trained
  butterfly-linear (BL) weights -- a general 2x2 coefficient matrix,
* sub-parallelism P_sub for transforms shorter than 2*P_bu (Sec. 3.2, Alg. 2)
  including the Eq. (5) interleaving,
* the integrated bit-reversal write-back (Sec. 3.3, Alg. 3) with bank-conflict
  checking on both the engine RAM and the global buffer, in exactly N/P cycles,
* P_N packing of several length-N sequences into one banked RAM (Sec. 3.4),
* the cycle model of Eq. (10) and an interconnect-complexity model for
  FCS (Theta(P_bu^2)) versus PRS (Theta(m*P_bu)).

Stage / hole-bit convention
---------------------------
The engine runs a decimation-in-frequency (Gentleman-Sande) radix-2 dataflow,
so stage k in [0, n) operates on index pairs that differ in bit

    h = n - 1 - k              ("hole bit")

Natural-order input therefore produces bit-reversed output, which is exactly
why the write-back path of Sec. 3.3 exists.  With this convention the subset
switch state derived from the conflict-free schedule is

    S^i = h  if h < m   (the hole lies inside the m low-order bits)
    S^i = 0  otherwise

which is identical to the control law of Algorithm 1,

    S^i = n - k - 1  for (n - m) <= k <= (n - 2),   S^i = 0 otherwise,

and the rotation is R^i = bsm(I^i[0]) as in Algorithm 1, line 7.

Author: written as a reference/verification model, plain NumPy, no dependencies
beyond the standard scientific stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "popcount", "bit_rev", "bit_rev_array", "rotl", "bsm", "bsm_array",
    "FixedPointFormat", "BankConflictError", "BankedMemory",
    "PermuteRotateSwitch", "FullyConnectedSwitch",
    "CycleControl", "ButterflySchedule",
    "fft_coefficients", "random_bl_coefficients", "expand_coefficients",
    "coefficients_from_paper_layout", "butterfly_reference",
    "TransformStats", "FlexBE", "BEArray",
    "algorithm1_fixed", "algorithm1_fixed_schedule",
    "bitrev_schedule", "cycles_eq10", "interconnect_cost",
]


# ---------------------------------------------------------------------------
# bit-level helpers
# ---------------------------------------------------------------------------

def popcount(x: int) -> int:
    """Number of 1s in the binary representation of x."""
    return int(x).bit_count()


def bit_rev(x: int, n: int) -> int:
    """Reverse the n-bit binary representation of x  (bit-rev_n of the paper)."""
    r = 0
    for _ in range(n):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def bit_rev_array(n: int) -> np.ndarray:
    """Vector v with v[i] = bit_rev(i, n) for i in [0, 2**n)."""
    idx = np.arange(1 << n, dtype=np.int64)
    out = np.zeros(1 << n, dtype=np.int64)
    for b in range(n):
        out |= ((idx >> b) & 1) << (n - 1 - b)
    return out


def rotl(x: int, q: int, width: int) -> int:
    """Circular left rotation of the `width`-bit word x by q positions."""
    mask = (1 << width) - 1
    x &= mask
    q %= width
    if q == 0:
        return x
    return ((x << q) | (x >> (width - q))) & mask


def _popcount_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    try:                                    # NumPy >= 2.0
        return np.bitwise_count(a).astype(np.int64)
    except AttributeError:                  # pragma: no cover
        c = np.zeros_like(a)
        x = a.copy()
        while np.any(x):
            c += (x & 1)
            x >>= 1
        return c


def bsm(I: int, m: int) -> int:
    """Bit-sum-and-modulo bank index of Eq. (3)."""
    mask = (1 << m) - 1
    return (popcount(I >> m) + (I & mask)) & mask


def bsm_array(I: np.ndarray, m: int) -> np.ndarray:
    """Vectorised bsm()."""
    I = np.asarray(I, dtype=np.int64)
    mask = (1 << m) - 1
    return (_popcount_array(I >> m) + (I & mask)) & mask


def insert_zero(t: int, h: int) -> int:
    """Insert a 0 bit at position h of t (t < 2**(w-1) -> result < 2**w)."""
    low = t & ((1 << h) - 1)
    return ((t >> h) << (h + 1)) | low


# ---------------------------------------------------------------------------
# fixed-point (Q1.15) model -- Sec. 5.2
# ---------------------------------------------------------------------------

@dataclass
class FixedPointFormat:
    """Q1.15 with 32-bit internal accumulation, convergent rounding, saturation.

    The multiply-accumulate of a butterfly unit is evaluated at full precision
    (modelling the 32-bit accumulator) and only the result is rounded back to
    the word format, exactly as described in Sec. 5.2 of the paper.
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
        """Round (half-to-even) and saturate to the word format."""
        x = np.asarray(x)
        if np.iscomplexobj(x):
            return self.quantise(x.real) + 1j * self.quantise(x.imag)
        q = np.round(x * self.scale)            # numpy rounds half to even
        if self.saturate:
            lo = -(1 << (self.word_bits - 1))
            hi = (1 << (self.word_bits - 1)) - 1
            q = np.clip(q, lo, hi)
        return q / self.scale


# ---------------------------------------------------------------------------
# banked data RAM with "shift-down" storage
# ---------------------------------------------------------------------------

class BankConflictError(RuntimeError):
    """Raised when two lanes address the same RAM bank in the same cycle."""


class BankedMemory:
    """P parallel single-port banks of `depth` words each.

    A read (or write) supplies one address per bank; the bank indices used in a
    cycle must be pairwise distinct, which is what the shift-down scheme of
    Eq. (3) guarantees.
    """

    def __init__(self, num_banks: int, depth: int, dtype=np.complex128):
        self.P = int(num_banks)
        self.depth = int(depth)
        self.banks = np.zeros((self.P, self.depth), dtype=dtype)
        self.read_cycles = 0
        self.write_cycles = 0

    # -- vector access -----------------------------------------------------
    def _check(self, banks: np.ndarray) -> None:
        if len(np.unique(banks)) != len(banks):
            raise BankConflictError(f"bank conflict on {np.asarray(banks).tolist()}")

    def read_vector(self, banks: np.ndarray, depths: np.ndarray,
                    check: bool = True) -> np.ndarray:
        if check:
            self._check(banks)
        self.read_cycles += 1
        return self.banks[banks, depths]

    def write_vector(self, banks: np.ndarray, depths: np.ndarray,
                     values: np.ndarray, check: bool = True) -> None:
        if check:
            self._check(banks)
        self.write_cycles += 1
        self.banks[banks, depths] = values

    # -- helpers -----------------------------------------------------------
    def load_sequence(self, seq: np.ndarray, m: int, base_depth: int = 0) -> None:
        """Store a natural-order sequence using the shift-down mapping."""
        N = len(seq)
        idx = np.arange(N, dtype=np.int64)
        self.banks[bsm_array(idx, m), (idx >> m) + base_depth] = seq

    def read_sequence(self, N: int, m: int, base_depth: int = 0) -> np.ndarray:
        idx = np.arange(N, dtype=np.int64)
        return self.banks[bsm_array(idx, m), (idx >> m) + base_depth].copy()


# ---------------------------------------------------------------------------
# switching networks -- Sec. 3.1
# ---------------------------------------------------------------------------

class PermuteRotateSwitch:
    """PRS: barrel shifter (rotation R) followed by a subset switch (state S).

    forward(D, R, S)[j] = D[(pi_S[j] + R) mod P]
    where pi_S is one of the m fixed stride permutations

        pi_S[2t]   = insert_zero(t, S)
        pi_S[2t+1] = pi_S[2t] + 2**S

    Interconnect complexity is Theta(m * P_bu) instead of the Theta(P_bu**2) of
    the fully connected switch.
    """

    def __init__(self, m: int):
        self.m = int(m)
        self.P = 1 << self.m
        self._perms: Dict[int, np.ndarray] = {}
        for S in range(max(1, self.m)):
            pi = np.empty(self.P, dtype=np.int64)
            for t in range(self.P // 2):
                pi[2 * t] = insert_zero(t, S)
                pi[2 * t + 1] = pi[2 * t] + (1 << S)
            self._perms[S] = pi

    def perm(self, S: int) -> np.ndarray:
        return self._perms[S]

    def read_map(self, R: int, S: int) -> np.ndarray:
        """Bank read by butterfly slot j, i.e. (pi_S[j] + R) mod P."""
        return (self._perms[S] + R) % self.P

    def forward(self, D: np.ndarray, R: int, S: int) -> np.ndarray:
        """Bank-ordered data -> butterfly-slot-ordered data (PRS-A)."""
        return D[self.read_map(R, S)]

    def inverse(self, X: np.ndarray, R: int, S: int) -> np.ndarray:
        """Butterfly-slot-ordered data -> bank-ordered data (PRS-B)."""
        D = np.empty_like(X)
        D[self.read_map(R, S)] = X
        return D

    # -- cost model (Sec. 3.1) --------------------------------------------
    def mux_count(self, data_width: int = 1) -> int:
        """2-to-1 multiplexers required, per Sec. 3.1: m*P for the barrel
        shifter plus (m+1)*2**(m-1) subset-switch connections."""
        return data_width * (self.m * self.P + (self.m + 1) * (self.P // 2))

    def pipeline_registers(self, data_width: int = 16) -> int:
        return self.P * self.m * data_width


class FullyConnectedSwitch:
    """Baseline FCS: a general P-to-P crossbar driven by the matrix P_f."""

    def __init__(self, m: int):
        self.m = int(m)
        self.P = 1 << self.m

    @staticmethod
    def matrix(indices: np.ndarray, m: int) -> np.ndarray:
        """P_f[j,k] = 1 iff k = bsm(I[j])   (Sec. 3, unnumbered equation)."""
        P = 1 << m
        Pf = np.zeros((P, P), dtype=np.int8)
        Pf[np.arange(P), bsm_array(indices, m)] = 1
        return Pf

    @staticmethod
    def forward(D: np.ndarray, Pf: np.ndarray) -> np.ndarray:
        return D[np.argmax(Pf, axis=1)]

    def mux_count(self, data_width: int = 1) -> int:
        """2P_bu output ports, each a 2P_bu-to-1 mux = (2P_bu - 1) 2-to-1 muxes,
        i.e. P*(P-1) in total -- Eq. (4)."""
        return data_width * self.P * (self.P - 1)

    def pipeline_registers(self, data_width: int = 16) -> int:
        return self.P * (self.P - 1) * data_width


def interconnect_cost(P_bu: int, data_width: int = 1) -> Dict[str, int]:
    """Model behind Fig. 9: FCS grows as Theta(P_bu^2), PRS as Theta(m*P_bu)."""
    m = (2 * P_bu).bit_length() - 1
    prs = PermuteRotateSwitch(m)
    fcs = FullyConnectedSwitch(m)
    return {
        "ports": 2 * P_bu,
        "m": m,
        "fcs_mux2": fcs.mux_count(data_width),
        "prs_mux2": prs.mux_count(data_width),
        "fcs_pipeline_regs": fcs.pipeline_registers(),
        "prs_pipeline_regs": prs.pipeline_registers(),
        "ratio": fcs.mux_count(data_width) / prs.mux_count(data_width),
    }


# ---------------------------------------------------------------------------
# conflict-free schedule  (index vectors I^i and the PRS controls R^i, S^i)
# ---------------------------------------------------------------------------

@dataclass
class CycleControl:
    """Everything the flex-butterfly management module emits in one cycle."""
    stage: int                    # k
    hole: int                     # h = n-1-k
    R: int                        # rotation of the barrel shifter
    S: int                        # subset-switch state
    indices: np.ndarray           # I^i, 2*P_bu element indices in slot order
    depth_by_bank: np.ndarray     # Addr^i: read depth for each RAM bank
    coeff_index: np.ndarray       # P_bu coefficient/twiddle addresses


def _subset_perm(S: int, m: int) -> np.ndarray:
    P = 1 << m
    pi = np.empty(P, dtype=np.int64)
    for t in range(P // 2):
        pi[2 * t] = insert_zero(t, S)
        pi[2 * t + 1] = pi[2 * t] + (1 << S)
    return pi


def _stage_groups(n: int, h: int, m: int) -> Iterator[Tuple[np.ndarray, int, int]]:
    """Yield (I^i, R^i, S^i) for every cycle of the stage whose hole bit is h.

    Two regimes, both conflict free under the shift-down mapping of Eq. (3):

    h < m   -- both elements of a butterfly live in the same RAM row.  One
               cycle handles one complete row (all 2^m low values), so the
               banks (popcount(H) + low) mod 2^m are pairwise distinct.
               The slot order is the stride permutation pi_h  ->  S = h.

    h >= m  -- the partner lives in row H + 2^(h-m), whose popcount is
               popcount(H) + 1.  One cycle takes P_bu even (or odd) low values
               from both rows; the banks are then (popcount(H) + e + j) mod 2^m
               for slot j, i.e. a pure rotation  ->  S = 0.
    """
    P = 1 << m
    N = 1 << n
    n_rows = N >> m

    if h < m:
        pi = _subset_perm(h, m)
        for H in range(n_rows):
            yield (H << m) + pi, popcount(H) & (P - 1), h
    else:
        step = 1 << (h - m)
        t2 = 2 * np.arange(P // 2, dtype=np.int64)
        for H in range(n_rows):
            if H & step:
                continue
            for e in (0, 1):
                idx = np.empty(P, dtype=np.int64)
                idx[0::2] = (H << m) + t2 + e
                idx[1::2] = ((H + step) << m) + t2 + e
                yield idx, (popcount(H) + e) & (P - 1), 0


class ButterflySchedule:
    """Full per-stage, per-cycle control sequence for one N-point transform.

    Parameters
    ----------
    n      : log2 of the engine computational length N
    m      : log2(2*P_bu)
    P_sub  : sub-parallelism; the last log2(P_sub) stages are skipped (Alg. 2)
    """

    def __init__(self, n: int, m: int, P_sub: int = 1, validate: bool = False):
        assert P_sub >= 1 and (P_sub & (P_sub - 1)) == 0
        self.n, self.m = int(n), int(m)
        self.N, self.P = 1 << self.n, 1 << self.m
        assert self.N >= self.P, "transform shorter than the datapath: use P_sub"
        self.s = int(math.log2(P_sub))
        self.P_sub = P_sub
        self.n_stages = self.n - self.s
        self.cycles_per_stage = self.N // self.P

        self.stages: List[List[CycleControl]] = []
        for k in range(self.n_stages):
            h = self.n - 1 - k
            cycles: List[CycleControl] = []
            for idx, R, S in _stage_groups(self.n, h, self.m):
                banks = bsm_array(idx, self.m)
                dbb = np.empty(self.P, dtype=np.int64)
                dbb[banks] = idx >> self.m
                a = idx[0::2]
                cidx = ((a >> (h + 1)) << h) | (a & ((1 << h) - 1))
                cycles.append(CycleControl(k, h, int(R), int(S), idx, dbb, cidx))
            assert len(cycles) == self.cycles_per_stage
            self.stages.append(cycles)
        if validate:
            self.validate()

    # -- properties --------------------------------------------------------
    @property
    def total_cycles(self) -> int:
        return self.n_stages * self.cycles_per_stage

    def all_cycles(self) -> Iterator[CycleControl]:
        for st in self.stages:
            yield from st

    # -- self checks -------------------------------------------------------
    def validate(self) -> None:
        """Assert conflict freedom, full coverage and the PRS == FCS identity."""
        prs = PermuteRotateSwitch(self.m)
        for k, stage in enumerate(self.stages):
            seen = np.zeros(self.N, dtype=bool)
            for c in stage:
                banks = bsm_array(c.indices, self.m)
                if len(np.unique(banks)) != self.P:
                    raise BankConflictError(
                        f"stage {k}: bank conflict for I^i={c.indices.tolist()}")
                if np.any(seen[c.indices]):
                    raise AssertionError(f"stage {k}: element visited twice")
                seen[c.indices] = True
                # butterfly pairing
                if not np.all(c.indices[1::2] - c.indices[0::2] == (1 << c.hole)):
                    raise AssertionError("bad butterfly pairing")
                if np.any((c.indices[0::2] >> c.hole) & 1):
                    raise AssertionError("lower index must have hole bit 0")
                # PRS decomposition == FCS crossbar
                if not np.array_equal(prs.read_map(c.R, c.S), banks):
                    raise AssertionError(
                        f"stage {k}: PRS(R={c.R},S={c.S}) != FCS map")
                if c.R != bsm(int(c.indices[0]), self.m):
                    raise AssertionError("R^i != bsm(I^i[0])  (Alg. 1 line 7)")
            if not seen.all():
                raise AssertionError(f"stage {k}: incomplete coverage")

    # -- paper cross-check -------------------------------------------------
    def paper_subset_state(self, k: int) -> int:
        """S^i as given by Algorithm 1, lines 8-11."""
        n, m = self.n, self.m
        return (n - k - 1) if (n - m) <= k <= (n - 2) else 0


# ---------------------------------------------------------------------------
# coefficients: FFT twiddles and butterfly-linear weights
# ---------------------------------------------------------------------------

def fft_coefficients(l: int, inverse: bool = False,
                     fmt: Optional[FixedPointFormat] = None) -> np.ndarray:
    """Radix-2 decimation-in-frequency twiddles for a length-l transform.

    Returns an array C of shape (log2(l), l//2, 2, 2) where C[k, o] is the 2x2
    matrix of butterfly o in stage k:  [[1, 1], [W, -W]].
    """
    nl = int(math.log2(l))
    assert 1 << nl == l
    C = np.zeros((nl, l // 2, 2, 2), dtype=np.complex128)
    sign = 1.0 if inverse else -1.0
    for k in range(nl):
        h = nl - 1 - k
        nb = l >> (h + 1)
        t = np.arange(1 << h, dtype=np.int64)
        e = t * (l >> (h + 1))
        W = np.exp(sign * 2j * np.pi * e / l)
        if fmt is not None:
            W = fmt.quantise(W)
        blk = C[k].reshape(nb, 1 << h, 2, 2)
        blk[:, :, 0, 0] = 1.0
        blk[:, :, 0, 1] = 1.0
        blk[:, :, 1, 0] = W[None, :]
        blk[:, :, 1, 1] = -W[None, :]
    return C


def random_bl_coefficients(l: int, rng: np.random.Generator,
                           scale: float = 1.0, real: bool = True,
                           fmt: Optional[FixedPointFormat] = None) -> np.ndarray:
    """Random trained-style butterfly-linear weights, shape (log2 l, l//2, 2, 2)."""
    nl = int(math.log2(l))
    shape = (nl, l // 2, 2, 2)
    if real:
        C = rng.normal(0.0, scale / math.sqrt(2.0), shape)
    else:
        C = (rng.normal(0.0, scale / 2.0, shape)
             + 1j * rng.normal(0.0, scale / 2.0, shape))
    if fmt is not None:
        C = fmt.quantise(C)
    return C


def expand_coefficients(C: np.ndarray, P_sub: int) -> np.ndarray:
    """Map length-l coefficients onto the N = P_sub*l interleaved datapath.

    With the Eq. (5) interleaving, element position a of the computational
    vector belongs to stream a mod P_sub and has sub-index a >> log2(P_sub);
    every logical butterfly is therefore replicated P_sub times.
    """
    if P_sub == 1:
        return C
    s = int(math.log2(P_sub))
    nl, half, _, _ = C.shape
    l = 2 * half
    N = P_sub * l
    n = int(math.log2(N))
    out = np.zeros((nl, N // 2, 2, 2), dtype=C.dtype)
    for k in range(nl):
        hp = nl - 1 - k                 # hole bit inside the length-l transform
        h = hp + s                      # hole bit inside the length-N datapath
        nb = l >> (hp + 1)
        src = C[k].reshape(nb, 1 << hp, 2, 2)
        dst = np.repeat(src, P_sub, axis=1)          # t >> s
        out[k] = dst.reshape(N // 2, 2, 2)
        assert (N >> (h + 1)) == nb
    return out


def coefficients_from_paper_layout(W: np.ndarray) -> np.ndarray:
    """Convert weights stored as [2, 2, l/2, log2 l] (Table 4) to (log2 l, l/2, 2, 2)."""
    return np.transpose(np.asarray(W), (3, 2, 0, 1)).copy()


def butterfly_reference(comp: np.ndarray, C: np.ndarray, P_sub: int = 1,
                        stage_scale: Optional[float] = None,
                        fmt: Optional[FixedPointFormat] = None) -> np.ndarray:
    """Golden model of the butterfly datapath (no memory, no switching).

    comp : (n_comp, N) already-interleaved computational vectors
    C    : (n_stages, N//2, 2, 2) coefficients on the length-N datapath
    """
    comp = np.atleast_2d(comp)
    n_comp, N = comp.shape
    n = int(math.log2(N))
    s = int(math.log2(P_sub))
    n_stages = C.shape[0]
    assert n_stages == n - s
    y = comp.astype(np.result_type(comp.dtype, C.dtype, np.complex128)).copy()
    for k in range(n_stages):
        h = n - 1 - k
        v = y.reshape(n_comp, N >> (h + 1), 2, 1 << h)
        lo = v[:, :, 0, :].copy()
        hi = v[:, :, 1, :].copy()
        Ck = C[k].reshape(N >> (h + 1), 1 << h, 2, 2)
        nl = Ck[None, :, :, 0, 0] * lo + Ck[None, :, :, 0, 1] * hi
        nh = Ck[None, :, :, 1, 0] * lo + Ck[None, :, :, 1, 1] * hi
        if stage_scale is not None:
            nl = nl * stage_scale
            nh = nh * stage_scale
        if fmt is not None:
            nl = fmt.quantise(nl)
            nh = fmt.quantise(nh)
        v[:, :, 0, :] = nl
        v[:, :, 1, :] = nh
        y = v.reshape(n_comp, N)
    return y


# ---------------------------------------------------------------------------
# bit-reversal write-back -- Sec. 3.3, Algorithm 3
# ---------------------------------------------------------------------------

def bitrev_schedule(n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithm 3: per-cycle index vectors (j, k) for the bit-reversal path.

    Returns j, k of shape (N/P, P).  Lane a of a cycle reads FlexBE RAM bank a
    at depth j[a] >> m and writes global-buffer bank k[a] mod 2^m at depth
    k[a] >> m.  Requires the long-transform regime N >= P^2 (n >= 2m).
    """
    P = 1 << m
    if n < 2 * m:
        raise ValueError("Algorithm 3 requires n >= 2m (N >= P^2)")
    q = min(m, n - 2 * m)
    r = n - 2 * m - q
    js: List[np.ndarray] = []
    a_vec = np.arange(P, dtype=np.int64)
    for u in range(P):
        b_vec = (a_vec + u) & (P - 1)
        for rr in range(1 << r):
            for qq in range(1 << q):
                high = (b_vec << (q + r)) | (qq << r) | rr
                x = (a_vec - _popcount_array(high)) & (P - 1)
                js.append((high << m) | x)
    j = np.asarray(js, dtype=np.int64)
    rev = bit_rev_array(n)
    return j, rev[j]


def _bitrev_pairs(n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithm 3 when applicable, otherwise a plain N/P-cycle fallback."""
    P = 1 << m
    N = 1 << n
    if n >= 2 * m:
        return bitrev_schedule(n, m)
    j = np.arange(N, dtype=np.int64).reshape(N // P, P)
    return j, bit_rev_array(n)[j]


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

@dataclass
class TransformStats:
    """Cycle and structural counters for one transform call."""
    N: int = 0                      # engine computational length
    l: int = 0                      # logical transform length
    P_sub: int = 1
    P_N: int = 0                    # sequences packed into the RAM array
    n_stages: int = 0
    butterfly_cycles: int = 0
    bitrev_cycles: int = 0
    butterfly_ops: int = 0
    ram_depth: int = 0
    scale: float = 1.0              # overall gain applied by stage scaling

    @property
    def total_cycles(self) -> int:
        return self.butterfly_cycles + self.bitrev_cycles

    def __str__(self) -> str:
        return (f"N={self.N} l={self.l} P_sub={self.P_sub} P_N={self.P_N} "
                f"stages={self.n_stages} bfly_cycles={self.butterfly_cycles} "
                f"bitrev_cycles={self.bitrev_cycles} "
                f"total={self.total_cycles}")


class FlexBE:
    """One flexible butterfly engine with P_bu radix-2 butterfly units.

    Runtime-reconfigurable parameters (Table 5): N (or l), P_sub and P_N.
    Compile-time parameters: P_bu and the arithmetic format.
    """

    def __init__(self, P_bu: int = 16, fmt: Optional[FixedPointFormat] = None,
                 check_conflicts: bool = True, name: str = "FlexBE"):
        assert P_bu >= 1 and (P_bu & (P_bu - 1)) == 0
        self.P_bu = int(P_bu)
        self.P = 2 * self.P_bu
        self.m = self.P.bit_length() - 1
        self.fmt = fmt
        self.name = name
        self.check_conflicts = check_conflicts
        self.prs = PermuteRotateSwitch(self.m)
        self.cycles = 0                       # lifetime cycle counter
        self._sched: Dict[Tuple[int, int], ButterflySchedule] = {}
        self._lane = np.arange(self.P, dtype=np.int64)

    # -- schedules ---------------------------------------------------------
    def schedule(self, n: int, P_sub: int = 1,
                 validate: bool = False) -> ButterflySchedule:
        key = (n, P_sub)
        if key not in self._sched:
            self._sched[key] = ButterflySchedule(n, self.m, P_sub, validate)
        elif validate:
            self._sched[key].validate()
        return self._sched[key]

    # -- geometry ----------------------------------------------------------
    def geometry(self, l: int) -> Tuple[int, int]:
        """Return (N, P_sub) for a logical transform length l  (Alg. 2)."""
        if l < self.P:
            assert self.P % l == 0, "l must divide 2*P_bu"
            return self.P, self.P // l
        return l, 1

    # -- interleaving, Eq. (5) --------------------------------------------
    def interleave(self, x: np.ndarray, P_sub: int) -> np.ndarray:
        n_seq, l = x.shape
        if P_sub == 1:
            return x
        n_comp = math.ceil(n_seq / P_sub)
        pad = np.zeros((n_comp * P_sub, l), dtype=x.dtype)
        pad[:n_seq] = x
        return pad.reshape(n_comp, P_sub, l).transpose(0, 2, 1).reshape(n_comp, -1)

    def deinterleave(self, comp: np.ndarray, l: int, P_sub: int,
                     n_seq: int) -> np.ndarray:
        if P_sub == 1:
            return comp[:n_seq]
        n_comp = comp.shape[0]
        out = comp.reshape(n_comp, l, P_sub).transpose(0, 2, 1).reshape(-1, l)
        return out[:n_seq]

    # -- main entry point --------------------------------------------------
    def transform(self, x, coeffs: Optional[np.ndarray] = None,
                  bitrev: bool = False, stage_scale: Optional[float] = None,
                  datapath: str = "cycle",
                  validate: bool = False) -> Tuple[np.ndarray, TransformStats]:
        """Run P_N transforms of length l packed into the engine RAM.

        Parameters
        ----------
        x            : (P_N, l) or (l,) input, natural order
        coeffs       : (log2 l, l//2, 2, 2) coefficients; None -> FFT twiddles
        bitrev       : apply the integrated bit-reversal write-back so that the
                       result is returned in natural order
        stage_scale  : optional per-stage gain (use 0.5 for fixed-point FFTs)
        datapath     : 'cycle'  -> full banked-memory + PRS cycle simulation
                       'vector' -> golden datapath, cycles still modelled
        """
        x = np.atleast_2d(np.asarray(x))
        n_seq, l = x.shape
        N, P_sub = self.geometry(l)
        n = int(math.log2(N))
        nl = int(math.log2(l))
        sched = self.schedule(n, P_sub, validate=validate)

        if coeffs is None:
            coeffs = fft_coefficients(l, fmt=self.fmt)
        assert coeffs.shape[0] == nl, "coefficient array must have log2(l) stages"
        C = expand_coefficients(coeffs, P_sub)

        comp = self.interleave(x.astype(np.complex128), P_sub)
        n_comp = comp.shape[0]
        depth_per_seq = N // self.P

        stats = TransformStats(N=N, l=l, P_sub=P_sub, P_N=n_seq,
                               n_stages=sched.n_stages,
                               ram_depth=n_comp * depth_per_seq,
                               scale=1.0 if stage_scale is None
                               else stage_scale ** sched.n_stages)

        if datapath == "vector":
            out = butterfly_reference(comp, C, P_sub, stage_scale, self.fmt)
            stats.butterfly_cycles = sched.total_cycles * n_comp
            stats.butterfly_ops = sched.total_cycles * n_comp * self.P_bu
        elif datapath == "cycle":
            out, cyc = self._run_cycles(comp, C, sched, depth_per_seq,
                                        stage_scale)
            stats.butterfly_cycles = cyc
            stats.butterfly_ops = cyc * self.P_bu
        else:
            raise ValueError(f"unknown datapath '{datapath}'")

        if bitrev:
            out, bcyc = self._bitrev(out, n, P_sub, l, datapath)
            stats.bitrev_cycles = bcyc

        self.cycles += stats.total_cycles
        return self.deinterleave(out, l, P_sub, n_seq), stats

    # -- cycle-level datapath ---------------------------------------------
    def _run_cycles(self, comp: np.ndarray, C: np.ndarray,
                    sched: ButterflySchedule, depth_per_seq: int,
                    stage_scale: Optional[float]):
        P, m = self.P, self.m
        n_comp, N = comp.shape
        mem = BankedMemory(P, n_comp * depth_per_seq, dtype=np.complex128)
        for c in range(n_comp):
            mem.load_sequence(comp[c], m, c * depth_per_seq)

        lane = self._lane
        check = self.check_conflicts
        fmt = self.fmt
        cycles = 0
        for k, stage in enumerate(sched.stages):
            Ck = C[k]
            for ctrl in stage:
                rmap = self.prs.read_map(ctrl.R, ctrl.S)
                W = Ck[ctrl.coeff_index]
                for c in range(n_comp):
                    addr = ctrl.depth_by_bank + c * depth_per_seq
                    D = mem.read_vector(lane, addr, check=check)
                    X = D[rmap]                       # PRS-A
                    a, b = X[0::2], X[1::2]
                    na = W[:, 0, 0] * a + W[:, 0, 1] * b
                    nb = W[:, 1, 0] * a + W[:, 1, 1] * b
                    if stage_scale is not None:
                        na = na * stage_scale
                        nb = nb * stage_scale
                    if fmt is not None:
                        na = fmt.quantise(na)
                        nb = fmt.quantise(nb)
                    Xp = np.empty_like(X)
                    Xp[0::2] = na
                    Xp[1::2] = nb
                    Dp = np.empty_like(Xp)
                    Dp[rmap] = Xp                     # PRS-B
                    mem.write_vector(lane, addr, Dp, check=check)
                    cycles += 1
        out = np.stack([mem.read_sequence(N, m, c * depth_per_seq)
                        for c in range(n_comp)])
        return out, cycles

    # -- bit reversal ------------------------------------------------------
    def _bitrev(self, comp: np.ndarray, n: int, P_sub: int, l: int,
                datapath: str):
        """Write back through the bit-reversal network, Sec. 3.3."""
        n_comp, N = comp.shape
        if P_sub > 1:
            # each interleaved stream is bit-reversed independently
            s = int(math.log2(P_sub))
            nl = n - s
            rev = bit_rev_array(nl)
            view = comp.reshape(n_comp, l, P_sub)
            out = view[:, rev, :].reshape(n_comp, N)
            return out, math.ceil(N / self.P) * n_comp
        j, kk = _bitrev_pairs(n, self.m)
        # Algorithm 3 is defined for the long-transform regime N >= P^2; below
        # it the write-back is modelled as a plain N/P-cycle permutation and is
        # not claimed to be bank-conflict free, so it is not checked.
        exact = n >= 2 * self.m
        out = np.zeros_like(comp)
        for c in range(n_comp):
            if exact and datapath == "cycle" and self.check_conflicts:
                for jv, kv in zip(j, kk):
                    src_bank = bsm_array(jv, self.m)
                    dst_bank = kv & (self.P - 1)
                    if len(np.unique(src_bank)) != self.P:
                        raise BankConflictError("bit-reversal read conflict")
                    if len(np.unique(dst_bank)) != self.P:
                        raise BankConflictError("bit-reversal write conflict")
            out[c, kk.ravel()] = comp[c, j.ravel()]
        return out, j.shape[0] * n_comp


class BEArray:
    """P_be butterfly engines sharing one global data buffer (Fig. 7)."""

    def __init__(self, P_be: int = 4, P_bu: int = 16,
                 fmt: Optional[FixedPointFormat] = None,
                 check_conflicts: bool = True):
        self.P_be = int(P_be)
        self.P_bu = int(P_bu)
        self.engines = [FlexBE(P_bu, fmt, check_conflicts, name=f"BE{i}")
                        for i in range(self.P_be)]
        self.cycles = 0

    def transform(self, x, coeffs=None, bitrev: bool = False,
                  stage_scale: Optional[float] = None,
                  datapath: str = "vector") -> Tuple[np.ndarray, TransformStats]:
        """Spread P_N sequences over the engines; latency = slowest engine."""
        x = np.atleast_2d(np.asarray(x))
        n_seq = x.shape[0]
        chunks = np.array_split(np.arange(n_seq), self.P_be)
        outs, stats_list = [], []
        for eng, ids in zip(self.engines, chunks):
            if len(ids) == 0:
                continue
            y, st = eng.transform(x[ids], coeffs, bitrev, stage_scale, datapath)
            outs.append(y)
            stats_list.append(st)
        y = np.concatenate(outs, axis=0)
        st = stats_list[0]
        agg = TransformStats(N=st.N, l=st.l, P_sub=st.P_sub, P_N=n_seq,
                             n_stages=st.n_stages, scale=st.scale,
                             ram_depth=max(s.ram_depth for s in stats_list))
        agg.butterfly_cycles = max(s.butterfly_cycles for s in stats_list)
        agg.bitrev_cycles = max(s.bitrev_cycles for s in stats_list)
        agg.butterfly_ops = sum(s.butterfly_ops for s in stats_list)
        self.cycles += agg.total_cycles
        return y, agg


# ---------------------------------------------------------------------------
# performance model -- Eq. (10)
# ---------------------------------------------------------------------------

def cycles_eq10(P_N: int, N: int, P_sub: int, P_be: int, P_bu: int) -> float:
    """cycles ~= P_N*N / (2*P_sub*P_be*P_bu) * log2(N / P_sub)   -- Eq. (10).

    N is the engine computational length (max(l, 2*P_bu)), P_N the number of
    length-l sequences, P_sub the sub-parallelism.
    """
    return P_N * N / (2 * P_sub * P_be * P_bu) * math.log2(N / P_sub)


def layer_cycles(n_seq: int, l: int, P_be: int, P_bu: int) -> float:
    """Cycles for n_seq butterfly transforms of length l on a P_be x P_bu array."""
    P = 2 * P_bu
    if l < P:
        P_sub, N = P // l, P
    else:
        P_sub, N = 1, l
    return cycles_eq10(n_seq, N, P_sub, P_be, P_bu)


def bitrev_cycles(n_seq: int, N: int, P_be: int, P_bu: int) -> float:
    """Bit-reversal write-back: exactly N/P cycles per sequence (Sec. 3.3)."""
    return n_seq * N / (2 * P_bu) / P_be


# ---------------------------------------------------------------------------
# proposed repair of Algorithm 1, lines 5-6
# ---------------------------------------------------------------------------

def algorithm1_fixed(n: int, P_bu: int, k: int, j: int) -> CycleControl:
    """Closed-form, conflict-free replacement for Algorithm 1 lines 5-6.

    Stage k pairs indices differing in bit h = n-1-k (decimation in frequency).
    With ins(x, p) = ((x >> p) << (p+1)) | (x mod 2**p)  (insert a 0 bit at
    position p) the index vector of cycle j of stage k is

        h < m   (both operands in RAM row H):
            H = j
            I^i[2t]   = (H << m) + ins(t, h)          t = 0 .. P_bu-1
            S^i       = h

        h >= m  (operands in rows H and H + 2**(h-m)):
            e = j mod 2,   H = ins(j >> 1, h - m)
            I^i[2t]   = (H << m) + 2t + e
            S^i       = 0

        both:   I^i[2t+1] = I^i[2t] + 2**h
                R^i       = bsm(I^i[0])               (Algorithm 1, line 7)

    Lines 7-11 of the published listing are unchanged: R^i is still bsm(I^i[0])
    and S^i still equals n-k-1 on (n-m) <= k <= (n-2) and 0 elsewhere.

    Conflict freedom (see docs/algorithm1_fix.md for the two-line proof):
      * h < m : one cycle reads the whole of row H, so the banks are
        (popcount(H) + low) mod 2**m for low = 0..2**m-1, all distinct;
      * h >= m: bit h-m of H is 0 by construction, hence
        popcount(H + 2**(h-m)) = popcount(H) + 1 and the banks are
        (popcount(H) + e + slot) mod 2**m, i.e. a pure rotation -- all distinct.
    """
    m = (2 * P_bu).bit_length() - 1
    P = 1 << m
    assert 0 <= k < n and 0 <= j < (1 << (n - m))
    h = n - 1 - k
    t = np.arange(P // 2, dtype=np.int64)
    I = np.empty(P, dtype=np.int64)

    if h < m:                                   # pair inside one row
        H = j
        I[0::2] = (H << m) + (((t >> h) << (h + 1)) | (t & ((1 << h) - 1)))
        S = h
    else:                                       # pair spans two rows
        e = j & 1
        step = 1 << (h - m)
        r = j >> 1
        H = ((r >> (h - m)) << (h - m + 1)) | (r & (step - 1))
        I[0::2] = (H << m) + 2 * t + e
        I[1::2] = ((H + step) << m) + 2 * t + e
        S = 0
    I[1::2] = I[0::2] + (1 << h)
    R = bsm(int(I[0]), m)

    banks = bsm_array(I, m)
    dbb = np.empty(P, dtype=np.int64)
    dbb[banks] = I >> m
    a = I[0::2]
    cidx = ((a >> (h + 1)) << h) | (a & ((1 << h) - 1))
    return CycleControl(k, h, int(R), int(S), I, dbb, cidx)


def algorithm1_fixed_schedule(n: int, P_bu: int) -> List[CycleControl]:
    """Every cycle of an N-point transform, emitted by algorithm1_fixed()."""
    m = (2 * P_bu).bit_length() - 1
    return [algorithm1_fixed(n, P_bu, k, j)
            for k in range(n) for j in range(1 << (n - m))]


def _schedule_from_algorithm1_fixed(n: int, P_bu: int,
                                    validate: bool = False) -> ButterflySchedule:
    """A ButterflySchedule whose control comes only from algorithm1_fixed().

    Used to show that the repaired listing is self-contained: an engine driven
    purely by it computes a correct FFT with no bank conflicts.
    """
    m = (2 * P_bu).bit_length() - 1
    sch = ButterflySchedule.__new__(ButterflySchedule)
    sch.n, sch.m = n, m
    sch.N, sch.P = 1 << n, 1 << m
    sch.s, sch.P_sub = 0, 1
    sch.n_stages = n
    sch.cycles_per_stage = sch.N // sch.P
    sch.stages = [[algorithm1_fixed(n, P_bu, k, j)
                   for j in range(sch.cycles_per_stage)] for k in range(n)]
    if validate:
        sch.validate()
    return sch


ButterflySchedule.from_algorithm1_fixed = staticmethod(_schedule_from_algorithm1_fixed)
