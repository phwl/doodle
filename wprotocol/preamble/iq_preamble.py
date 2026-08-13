"""
iq_preamble.py — preamble/sync-word matched-filter bank for iq_protocol_id.

Why this exists
---------------
The blind feature classifier in ``iq_protocol_id`` needs roughly 20 dB SNR. That
floor is not a coding deficiency, it is structural: every blind feature is a
*second-order* statistic (a bandwidth, an envelope variance, a histogram of the
frequency discriminator). Second-order statistics of a signal in noise converge
at a rate that no amount of cleverness improves, and the frequency discriminator
in particular has a hard threshold effect — below about 10 dB its output noise
stops being small-signal and starts producing clicks.

A matched filter is a *first-order* statistic. It is the optimal detector for a
known waveform in additive white Gaussian noise, and it converts template length
directly into processing gain. Most protocols begin their frames with a fixed,
publicly documented preamble and sync word, so the waveform genuinely is known.

Detection theory
----------------
Correlate the capture against a unit-energy reference ``r`` of ``N`` samples,
normalised so the statistic is scale-free:

                    | sum_k  x[n+k] conj(r[k]) |
        rho[n]  =  --------------------------------
                    ||x[n:n+N]||  *  ||r||

*Noise only.* For circular complex Gaussian noise, rho^2 is Beta(1, N-1), so

        P(rho > t)  ~=  exp( -(N-1) t^2 )                                   (1)

which is the whole reason this works: the false-alarm probability falls
exponentially in template length, so the threshold needed for a fixed P_fa
shrinks as 1/sqrt(N).

*Signal present.* Write the per-sample SNR as gamma. With a perfectly matched,
constant-amplitude reference, the numerator accumulates coherently (~ A*N) while
the denominator grows as sqrt(N(A^2 + sigma^2)) * sqrt(N), giving

        E[rho]  ~=  sqrt( gamma / (1 + gamma) )                             (2)

Note what (2) does *not* contain: N. The peak correlation is set purely by SNR
and saturates at 1. The gain from a longer template lives entirely in (1), in
the threshold we are allowed to use.

*Combining them.* Detection is possible when the peak clears the threshold.
With ``M`` effective trials (samples x templates x frequency bins) and a target
false-alarm probability ``P_fa``, inverting (1) gives

        t  =  sqrt( ln(M / P_fa) / (N - 1) )                                (3)

and setting (2) above (3), using gamma/(1+gamma) ~= gamma for small gamma:

        gamma_min  ~=  ln(M / P_fa) / N                                     (4)

So the processing gain over any fixed-threshold scheme is N / ln(M/P_fa) — the
template length discounted by the logarithm of the search space. For a 40-bit
BLE advertising preamble at 8 samples/symbol, N = 320 and ln(M/P_fa) ~= 29, so
gamma_min ~= -10.4 dB. For the 8 upchirps of a LoRa SF7 preamble, N ~= 4096 and
gamma_min ~= -21 dB, which is why LoRa demodulates below the noise floor.

Frequency offset
----------------
Coherent correlation is the reason this works and also its main fragility. A
residual carrier offset ``df`` rotates the reference through ``df * T`` cycles
over the template duration ``T``, and the coherent sum degrades as
``|sinc(df*T)|`` — a full cycle of rotation destroys the gain completely. At
915 MHz a 10 ppm crystal is 9 kHz; over a 112 us SiK preamble that is one full
cycle. So we search a frequency grid, spaced at ``0.8/T`` to hold the worst-case
scalloping loss under about 1 dB.

The grid is nearly free. De-rotating the signal by ``-df`` is equivalent to
rotating the reference by ``+df``, and rotating the reference in time is a
*shift of its spectrum*. Since rho takes a magnitude, the constant phase that
the shift introduces is irrelevant. So we transform each template once and
circularly shift its spectrum per frequency bin: one forward FFT plus one
inverse FFT per bin, rather than a fresh correlation each time.

Relationship to channel filtering
---------------------------------
Filtering to the signal bandwidth before correlating buys no detection
performance — the matched filter already *is* the optimal filter, and it is easy
to see why in terms of (4): decimating by D raises gamma by ~D and shrinks N by
the same factor, leaving gamma*N invariant. It buys a large amount of compute,
which is why we still do it.

Usage
-----
    from iq_protocol_id import identify
    from iq_preamble import PreambleBank

    bank = PreambleBank(p_fa=1e-6)
    results = identify(iq, fs, rf_center_hz=915e6, preamble_bank=bank)

    # or standalone
    for hit in bank.scan(iq, fs, rf_center_hz=915e6):
        print(hit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import fft as sp_fft
from scipy import signal as sps_signal


# ---------------------------------------------------------------------------
# Reference waveform synthesis
# ---------------------------------------------------------------------------
# All synthesis supports a *fractional* samples-per-symbol, because the capture
# sample rate is whatever the SDR gave us and is rarely an integer multiple of
# the protocol's symbol rate.

def _bits(spec) -> np.ndarray:
    """Accept a bit list, a '1010' string, or ('hex', '2DD4', nbits)."""
    if isinstance(spec, str):
        return np.array([int(c) for c in spec if c in "01"], dtype=np.int8)
    if isinstance(spec, tuple) and spec and spec[0] == "hex":
        val, nbits = int(spec[1], 16), spec[2]
        return np.array([(val >> (nbits - 1 - i)) & 1 for i in range(nbits)],
                        dtype=np.int8)
    return np.asarray(spec, dtype=np.int8)


def synth_gfsk(bits: np.ndarray, fs: float, symbol_rate: float,
               deviation: float, bt: float = 0.5) -> np.ndarray:
    """Gaussian-filtered 2-FSK reference.

    The Gaussian pulse follows the usual definition with
    ``sigma = sqrt(ln 2) / (2*pi*BT)`` in symbol units. ``deviation`` is the
    *peak* deviation, so the modulation index is ``h = 2*deviation/symbol_rate``
    (h = 0.5 for BLE, giving 250 kHz at 1 Msym/s).
    """
    sps = fs / symbol_rate
    n = int(round(bits.size * sps))
    idx = np.clip((np.arange(n) / sps).astype(int), 0, bits.size - 1)
    nrz = 2.0 * bits[idx] - 1.0

    sigma = math.sqrt(math.log(2.0)) / (2.0 * math.pi * bt) * sps
    half = max(1, int(math.ceil(3.0 * sigma)))
    t = np.arange(-half, half + 1)
    g = np.exp(-0.5 * (t / max(sigma, 1e-9)) ** 2)
    g /= g.sum()
    shaped = np.convolve(nrz, g, mode="same")

    phase = 2.0 * np.pi * deviation * np.cumsum(shaped) / fs
    return np.exp(1j * phase).astype(np.complex64)


def synth_css_upchirps(fs: float, bw: float, sf: int, n_chirps: int) -> np.ndarray:
    """``n_chirps`` back-to-back base (symbol-zero) LoRa upchirps.

    A LoRa symbol sweeps the full bandwidth once, so the chirp rate is
    ``k = bw / T_sym`` with ``T_sym = 2^sf / bw``, i.e. ``k = bw^2 / 2^sf``.
    """
    n_sym = int(round(2 ** sf * fs / bw))
    t = np.arange(n_sym) / fs
    k = bw * bw / (2 ** sf)
    f = -bw / 2.0 + k * t
    one = np.exp(1j * 2.0 * np.pi * np.cumsum(f) / fs)
    return np.tile(one, n_chirps).astype(np.complex64)


def synth_ook_pulses(fs: float, pulses: Sequence[tuple[float, float]],
                     total_s: float, rise_s: float = 0.05e-6) -> np.ndarray:
    """On-off keyed reference from (start_seconds, width_seconds) pulse list.

    Returned real-valued. Correlating a real reference against complex IQ is
    correct here: the ``|.|`` in rho makes the statistic invariant to the
    unknown constant carrier phase.
    """
    n = int(round(total_s * fs))
    x = np.zeros(n, dtype=np.float64)
    for start, width in pulses:
        a, b = int(round(start * fs)), int(round((start + width) * fs))
        x[max(0, a):max(0, b)] = 1.0
    w = max(2, int(round(rise_s * fs)))
    x = np.convolve(x, np.ones(w) / w, mode="same")
    return x.astype(np.complex64)


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

@dataclass
class Template:
    name: str                       # matches a PROTOCOLS entry in iq_protocol_id
    kind: str                       # 'gfsk' | 'css' | 'ook'
    symbol_rate: float
    params: dict
    band: list[tuple[float, float]]
    spec: str
    verified: bool = True
    bits: Optional[str] = None
    cfo_span: Optional[float] = None   # +/- Hz to search; default 0.3*symbol_rate

    def duration(self) -> float:
        if self.kind == "ook":
            return float(self.params["total_s"])
        if self.kind == "css":
            p = self.params
            return p["n_chirps"] * (2 ** p["sf"]) / p["bw"]
        return len(_bits(self.bits)) / self.symbol_rate

    def synth(self, fs: float) -> np.ndarray:
        if self.kind == "gfsk":
            return synth_gfsk(_bits(self.bits), fs, self.symbol_rate,
                              self.params["deviation"], self.params.get("bt", 0.5))
        if self.kind == "css":
            p = self.params
            return synth_css_upchirps(fs, p["bw"], p["sf"], p["n_chirps"])
        if self.kind == "ook":
            return synth_ook_pulses(fs, self.params["pulses"],
                                    self.params["total_s"])
        raise ValueError(f"unknown template kind {self.kind!r}")


MHz, kHz = 1e6, 1e3

# --- Bluetooth LE advertising ------------------------------------------------
# The strongest template in the bank because nothing about it is negotiated:
# the preamble is 0xAA for the 1M PHY and the advertising access address is the
# fixed constant 0x8E89BED6. Whitening starts *after* the access address, so all
# 40 bits are deterministic. Transmitted LSB-first per octet, and the access
# address goes out least-significant-octet first.
_BLE_AA = "01101011011111011001000101110001"   # 0x8E89BED6, LSB-first per octet
_BLE_PREAMBLE_1M = "01010101"                  # 0xAA

BLE_ADV_1M = Template(
    name="Bluetooth LE 1M (GFSK)",
    kind="gfsk", symbol_rate=1e6,
    params={"deviation": 250 * kHz, "bt": 0.5},
    bits=_BLE_PREAMBLE_1M + _BLE_AA,
    band=[(2.402e9, 2.480e9)],
    spec="Bluetooth Core 5.4, Vol 6 Part B 2.1.2 (access address 0x8E89BED6)",
    cfo_span=250 * kHz,
)

BLE_ADV_2M = Template(
    name="Bluetooth LE 2M (GFSK)",
    kind="gfsk", symbol_rate=2e6,
    params={"deviation": 500 * kHz, "bt": 0.5},
    bits="01010101" * 2 + _BLE_AA,             # 2M PHY uses a 16-bit preamble
    band=[(2.402e9, 2.480e9)],
    spec="Bluetooth Core 5.4, Vol 6 Part B 2.1.2",
    cfo_span=400 * kHz,
)

# --- ADS-B 1090ES ------------------------------------------------------------
# Four 0.5 us pulses at 0, 1.0, 3.5 and 4.5 us inside an 8 us window. Entirely
# fixed by the standard and, at 1 Mchip/s, short — so this template is chosen
# for *robustness of identification* rather than deep sensitivity.
ADSB_PREAMBLE = Template(
    name="ADS-B 1090ES (PPM/OOK)",
    kind="ook", symbol_rate=1e6,
    params={"pulses": [(0.0, 0.5e-6), (1.0e-6, 0.5e-6),
                       (3.5e-6, 0.5e-6), (4.5e-6, 0.5e-6)],
            "total_s": 8.0e-6},
    band=[(1089e6, 1091e6)],
    spec="RTCA DO-260B 2.2.3.2.1.1 / ICAO Annex 10 Vol IV",
    cfo_span=0.0,                              # OOK envelope: no carrier phase
)

# --- LoRa preamble -----------------------------------------------------------
# 8 base upchirps. Structurally fixed by the spec even though the PHY itself is
# Semtech IP, and by far the longest template here, hence the lowest floor.
def lora_preamble(bw: float = 125 * kHz, sf: int = 7, n_chirps: int = 8) -> Template:
    return Template(
        name="LoRa / CSS (LoRaWAN, Meshtastic, ELRS)",
        kind="css", symbol_rate=bw / (2 ** sf),
        params={"bw": bw, "sf": sf, "n_chirps": n_chirps},
        band=[(150e6, 960e6), (2.400e9, 2.485e9)],
        spec="LoRa Alliance regional params; preamble = 8 base upchirps",
        cfo_span=0.25 * bw,
    )

# --- SiK / MAVLink telemetry -------------------------------------------------
# Si4432/Si1000-class radios: alternating preamble then a sync word. 0x2DD4 is
# the silicon default and what stock SiK builds use, but both the preamble
# length and the sync word are register-configurable, so a deployment that has
# changed them will not match. Flagged unverified for that reason, not because
# the constant is doubtful.
def sik_preamble(symbol_rate: float = 250e3, deviation: float = 60 * kHz,
                 lead_bits: int = 12, sync_hex: str = "2DD4") -> Template:
    return Template(
        name="SiK / MAVLink telemetry (2-FSK/GFSK)",
        kind="gfsk", symbol_rate=symbol_rate,
        params={"deviation": deviation, "bt": 0.5},
        bits="10" * (lead_bits // 2) + "".join(
            f"{int(sync_hex, 16):0{len(sync_hex) * 4}b}"),
        band=[(410e6, 480e6), (863e6, 928e6), (2.400e9, 2.485e9)],
        spec="SiK firmware / Si4432 datasheet (sync word is configurable)",
        verified=False,
        cfo_span=0.3 * symbol_rate,
    )

# --- POCSAG ------------------------------------------------------------------
# 576 bits of alternating 1/0. We take a 128-bit window: a long template at a
# low symbol rate, which is a very deep floor for very little compute.
def pocsag_preamble(symbol_rate: float = 1200.0, deviation: float = 4.5 * kHz,
                    n_bits: int = 128) -> Template:
    return Template(
        name="POCSAG paging (2-FSK)",
        kind="gfsk", symbol_rate=symbol_rate,
        params={"deviation": deviation, "bt": 1.0},
        bits="10" * (n_bits // 2),
        band=[(130e6, 960e6)],
        spec="ITU-R M.584-2 (576-bit alternating preamble)",
        cfo_span=2.0 * symbol_rate,
    )


def default_templates() -> list[Template]:
    return [
        BLE_ADV_1M,
        BLE_ADV_2M,
        ADSB_PREAMBLE,
        lora_preamble(125 * kHz, 7),
        lora_preamble(125 * kHz, 9),
        lora_preamble(250 * kHz, 7),
        sik_preamble(250e3, 60 * kHz),
        sik_preamble(64e3, 30 * kHz),
        pocsag_preamble(1200.0),
    ]


# ---------------------------------------------------------------------------
# Normalised matched filter
# ---------------------------------------------------------------------------

def threshold_for_pfa(n_template: int, n_trials: int, p_fa: float = 1e-6) -> float:
    """Threshold from equation (3). Never returns more than 0.999."""
    n_eff = max(2, n_template) - 1
    t = math.sqrt(max(0.0, math.log(max(n_trials, 1) / max(p_fa, 1e-300))) / n_eff)
    return float(min(t, 0.999))


def snr_floor_db(n_template: int, n_trials: int, p_fa: float = 1e-6) -> float:
    """Per-sample SNR floor from equation (4), in dB."""
    g = math.log(max(n_trials, 1) / max(p_fa, 1e-300)) / max(n_template, 1)
    g = min(g, 0.999)                     # gamma/(1+gamma) cannot exceed 1
    return 10.0 * math.log10(g / max(1.0 - g, 1e-12))


def _next_fast_len(n: int) -> int:
    return int(sp_fft.next_fast_len(n))


def normalized_correlate(x: np.ndarray, r: np.ndarray,
                         cfo_hz: Sequence[float] = (0.0,),
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Normalised correlation of ``x`` against ``r`` over a frequency grid.

    Returns ``(rho, best_cfo)``, both length ``len(x) - len(r) + 1``, where
    ``rho[n]`` is the best normalised correlation at offset ``n`` across the
    grid and ``best_cfo[n]`` is the grid point that achieved it.

    The frequency search reuses one template transform. Rotating the reference
    by ``+df`` shifts its spectrum; because ``rho`` takes a magnitude, the
    constant phase that shift introduces cancels, so each grid point costs one
    inverse FFT instead of a fresh correlation.
    """
    x = np.asarray(x, dtype=np.complex128)
    r = np.asarray(r, dtype=np.complex128)
    n, m = x.size, r.size
    if n < m:
        return np.zeros(0), np.zeros(0)

    out_len = n - m + 1
    nfft = _next_fast_len(n + m)

    # sliding energy of x over an m-sample window
    p = np.abs(x) ** 2
    cs = np.concatenate(([0.0], np.cumsum(p)))
    e_x = cs[m:] - cs[:-m]                       # length out_len
    e_r = float(np.sum(np.abs(r) ** 2))
    denom = np.sqrt(np.maximum(e_x, 1e-30) * max(e_r, 1e-30))

    X = np.fft.fft(x, nfft)
    h = np.conj(r[::-1])                         # correlation as convolution
    H = np.fft.fft(h, nfft)

    # frequency grid quantised to FFT bins so the shift is exact
    bin_hz = fs / nfft
    shifts = sorted({int(round(f / bin_hz)) for f in cfo_hz})

    # h_f[j] = C * h[j] * exp(+j2*pi*f*j/fs), and multiplying a sequence by that
    # exponential shifts its DFT *up* by k = f*nfft/fs bins. The leftover
    # constant C is a pure phase and dies in the |.|.
    rho = np.zeros(out_len)
    best = np.zeros(out_len)
    for k in shifts:
        c = np.fft.ifft(X * np.roll(H, k))[m - 1 : m - 1 + out_len]
        cand = np.abs(c) / denom
        upd = cand > rho
        rho[upd] = cand[upd]
        best[upd] = k * bin_hz
    return rho, best


def _cfo_grid(span_hz: float, duration_s: float) -> np.ndarray:
    """Grid spaced at 0.8/T, which holds worst-case scalloping under ~1 dB."""
    if span_hz <= 0 or duration_s <= 0:
        return np.array([0.0])
    step = 0.8 / duration_s
    k = int(math.ceil(span_hz / step))
    return np.arange(-k, k + 1) * step


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------

@dataclass
class PreambleHit:
    name: str
    rho: float
    threshold: float
    t0: float                    # seconds into the capture
    sample: int
    cfo_hz: float
    n_template: int
    template_duration: float
    processing_gain_db: float
    est_snr_db: float
    spec: str
    verified: bool = True
    decimation: int = 1
    shadowed: list = field(default_factory=list)   # cross-talk, suppressed

    @property
    def margin(self) -> float:
        return self.rho / max(self.threshold, 1e-9)

    def __str__(self) -> str:
        flag = "" if self.verified else "  [template constants configurable — verify]"
        shad = (f", shadowed {len(self.shadowed)}" if self.shadowed else "")
        return (f"{self.name}: rho={self.rho:.3f} vs threshold {self.threshold:.3f} "
                f"(margin {self.margin:.2f}x) at t={self.t0*1e6:.1f} us, "
                f"CFO={self.cfo_hz/1e3:+.1f} kHz, N={self.n_template}, "
                f"gain={self.processing_gain_db:.1f} dB, "
                f"est SNR={self.est_snr_db:+.1f} dB{shad}{flag}")


class PreambleBank:
    """Matched-filter bank over documented preambles and sync words."""

    def __init__(self, templates: Optional[Sequence[Template]] = None,
                 p_fa: float = 1e-6, sps_target: float = 8.0,
                 max_block: int = 1 << 18, min_margin: float = 1.0,
                 suppress_overlap: bool = True):
        self.templates = list(templates) if templates is not None else default_templates()
        self.p_fa = p_fa
        self.sps_target = sps_target
        self.max_block = max_block
        self.min_margin = min_margin
        self.suppress_overlap = suppress_overlap

    # -- helpers ------------------------------------------------------------
    def _bandwidth_of(self, tpl: Template) -> float:
        """Rough occupied bandwidth, used only to choose a decimation factor."""
        if tpl.kind == "css":
            return float(tpl.params["bw"])
        if tpl.kind == "ook":
            return 4.0 * tpl.symbol_rate
        dev = float(tpl.params["deviation"])
        return 2.0 * (dev + tpl.symbol_rate / 2.0)      # Carson

    def _decimate(self, x: np.ndarray, fs: float, tpl: Template):
        """Channel-filter and decimate for speed only.

        This buys no sensitivity — the matched filter is already the optimal
        filter, and in the language of equation (4) decimating by D raises
        gamma by ~D while shrinking N by the same factor. It buys compute.
        """
        bw = self._bandwidth_of(tpl)
        target = max(self.sps_target * tpl.symbol_rate, 2.5 * bw)
        d = int(max(1, math.floor(fs / target)))
        if d < 2 or x.size // d < 256:
            return x, fs, 1
        taps = min(193, (x.size // 4) | 1)
        if taps < 9:
            return x, fs, 1
        h = sps_signal.firwin(taps, 1.0 / d, window=("kaiser", 6.0))
        y = sps_signal.lfilter(h, 1.0, x)[::d].astype(np.complex64)
        return y, fs / d, d

    # -- main ---------------------------------------------------------------
    def scan(self, x: np.ndarray, fs: float,
             rf_center_hz: Optional[float] = None,
             derotate_hz: float = 0.0) -> list[PreambleHit]:
        """Correlate ``x`` against every in-band template.

        Runs on the *raw capture*, deliberately not on energy-detected bursts:
        below roughly 6 dB the energy detector finds nothing at all, and
        requiring it to fire first would reintroduce the floor we are removing.
        """
        x = np.asarray(x, dtype=np.complex64)
        if derotate_hz:
            n = np.arange(x.size)
            x = (x * np.exp(-2j * np.pi * derotate_hz * n / fs)).astype(np.complex64)

        hits: list[PreambleHit] = []
        for tpl in self.templates:
            if rf_center_hz is not None and not any(
                    lo <= rf_center_hz <= hi for lo, hi in tpl.band):
                continue

            xd, fsd, decim = self._decimate(x, fs, tpl)
            r = tpl.synth(fsd)
            if r.size < 16 or r.size > xd.size:
                continue

            span = tpl.cfo_span if tpl.cfo_span is not None else 0.3 * tpl.symbol_rate
            grid = _cfo_grid(span, tpl.duration())

            # trials: independent positions x frequency bins. Adjacent samples
            # are correlated over a template length, so the effective number of
            # independent positions is the record length in template durations,
            # inflated by 4 for safety.
            n_pos = max(1, int(4 * xd.size / r.size))
            n_trials = n_pos * grid.size * max(1, len(self.templates))
            thr = threshold_for_pfa(r.size, n_trials, self.p_fa)

            rho, cfo = self._scan_blocks(xd, r, grid, fsd)
            if rho.size == 0:
                continue
            k = int(np.argmax(rho))
            if rho[k] < thr * self.min_margin:
                continue

            # invert equation (2) for a post-detection SNR estimate
            p = min(float(rho[k]) ** 2, 0.999999)
            gamma = p / (1.0 - p)
            hits.append(PreambleHit(
                name=tpl.name, rho=float(rho[k]), threshold=thr,
                t0=k / fsd, sample=k * decim, cfo_hz=float(cfo[k]),
                n_template=int(r.size), template_duration=tpl.duration(),
                processing_gain_db=10.0 * math.log10(
                    r.size / math.log(max(n_trials, 2) / self.p_fa)),
                est_snr_db=10.0 * math.log10(max(gamma, 1e-12)),
                spec=tpl.spec, verified=tpl.verified, decimation=decim,
            ))

        hits.sort(key=lambda h: -(h.rho / max(h.threshold, 1e-9)))
        return self._suppress(hits) if self.suppress_overlap else hits

    @staticmethod
    def _suppress(hits: list[PreambleHit]) -> list[PreambleHit]:
        """Non-maximum suppression in time.

        Equation (1) bounds the false-alarm rate against *noise*. It says
        nothing about one template correlating with a different real signal, and
        several of these templates genuinely resemble each other — BLE 1M, BLE
        2M and SiK all open with an alternating GFSK pattern, so a BLE frame
        will light up all three. That is template cross-talk, not a false alarm,
        and no threshold can fix it because the correlation really is there.

        What does fix it: a frame is one protocol, so among hits whose template
        spans overlap in time only the largest margin survives. The suppressed
        hits are returned on ``.shadowed`` for inspection.

        Limitation: two genuinely concurrent signals on different channels
        within one capture will overlap in time, and the weaker one is lost.
        Pass ``suppress_overlap=False`` and separate by frequency first if that
        case matters.
        """
        kept: list[PreambleHit] = []
        for h in hits:                       # already sorted by descending margin
            a0, a1 = h.t0, h.t0 + h.template_duration
            clash = next((k for k in kept
                          if a0 < k.t0 + k.template_duration and k.t0 < a1), None)
            if clash is None:
                kept.append(h)
            else:
                clash.shadowed.append(h)
        return kept

    def _scan_blocks(self, x: np.ndarray, r: np.ndarray,
                     grid: np.ndarray, fs: float):
        """Overlap-save so memory stays bounded on long captures."""
        m = r.size
        if x.size <= self.max_block:
            return normalized_correlate(x, r, grid, fs)

        step = self.max_block - (m - 1)
        rho_parts, cfo_parts = [], []
        for start in range(0, x.size - m + 1, step):
            blk = x[start : start + step + m - 1]
            if blk.size < m:
                break
            a, b = normalized_correlate(blk, r, grid, fs)
            rho_parts.append(a)
            cfo_parts.append(b)
        if not rho_parts:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(rho_parts), np.concatenate(cfo_parts)
