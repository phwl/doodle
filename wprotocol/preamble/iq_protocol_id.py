"""
iq_protocol_id.py — heuristic protocol identification from baseband I/Q.

Pipeline
--------
    raw IQ  ->  burst segmentation  ->  per-burst feature extraction
            ->  modulation-family decision  ->  scored match against protocol DB

The classifier is deliberately *feature-based and interpretable* rather than a
black-box NN: every score comes with the reason it was assigned, so a wrong
answer tells you which measurement disagreed.

Features measured per burst
---------------------------
    occupied bandwidth (99% / RMS)     center frequency offset
    burst duration, rise time          envelope coefficient of variation
    PAPR                               spectral flatness + edge sharpness
    cyclic-prefix autocorrelation      chirp rate (linear FM / CSS)
    FSK tone count + deviation         symbol rate (spectral-line method)

Author's note: bandwidth and symbol rate are the two discriminators that carry
most of the information. Everything else mainly separates modulation families.

Usage
-----
    python iq_protocol_id.py capture.cf32 --fs 4e6 --center 2.44e9
    python iq_protocol_id.py capture.sc16 --fs 2e6 --format sc16 --top 5

    # or as a library
    from iq_protocol_id import identify
    report = identify(iq, fs=4e6, rf_center_hz=2.44e9)
    print(report)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_DTYPES = {
    "cf32": (np.complex64, False),
    "cf64": (np.complex128, False),
    "sc16": (np.int16, True),
    "sc8": (np.int8, True),
    "cu8": (np.uint8, True),   # RTL-SDR
}


def load_iq(path: str, fmt: str = "cf32", max_samples: Optional[int] = None) -> np.ndarray:
    """Read an interleaved IQ file into complex64."""
    if fmt not in _DTYPES:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(_DTYPES)}")
    dtype, interleaved = _DTYPES[fmt]
    count = -1 if max_samples is None else (max_samples * 2 if interleaved else max_samples)
    raw = np.fromfile(path, dtype=dtype, count=count)
    if interleaved:
        if raw.size % 2:
            raw = raw[:-1]
        raw = raw.astype(np.float32)
        if dtype is np.uint8:
            raw -= 127.5
        x = raw[0::2] + 1j * raw[1::2]
    else:
        x = raw
    x = np.asarray(x, dtype=np.complex64)
    # normalise to unit RMS so thresholds are scale-free
    rms = np.sqrt(np.mean(np.abs(x) ** 2)) or 1.0
    return (x / rms).astype(np.complex64)


# ---------------------------------------------------------------------------
# Burst segmentation
# ---------------------------------------------------------------------------

@dataclass
class Burst:
    i0: int
    i1: int
    fs: float

    @property
    def duration(self) -> float:
        return (self.i1 - self.i0) / self.fs

    @property
    def t0(self) -> float:
        return self.i0 / self.fs


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n, dtype=np.float64) / n
    return np.convolve(x, k, mode="same")


def _noise_floor_db(p_db: np.ndarray, n_blocks: int = 200) -> float:
    """Noise floor = the quietest block median.

    A global low percentile fails whenever the duty cycle is high (an 85%-on
    capture puts the 20th percentile inside the signal). Taking the minimum of
    per-block medians only needs *one* quiet window anywhere in the record, and
    degrades gracefully to the signal level for a truly continuous carrier —
    which is exactly what makes the continuous fallback fire.
    """
    n = p_db.size
    blk = max(16, n // n_blocks)
    m = (n // blk) * blk
    if m < blk:
        return float(np.percentile(p_db, 10.0))
    med = np.median(p_db[:m].reshape(-1, blk), axis=1)
    return float(np.percentile(med, 2.0)) if med.size >= 20 else float(med.min())


def detect_bursts(
    x: np.ndarray,
    fs: float,
    win_s: float = 2e-6,
    on_db: float = 8.0,
    off_db: float = 5.0,
    min_dur_s: float = 4e-6,
    max_gap_s: float = 8e-6,
    guard_frac: float = 0.05,
) -> tuple[list[Burst], float]:
    """Energy-based burst detector with hysteresis.

    Returns (bursts, noise_floor_db). If the signal turns out to be
    continuous, a single burst spanning the whole record is returned.
    """
    # Floors in *samples*, not just seconds. These time constants were chosen
    # for MHz-rate captures; at 48 ksps a 2 us window is 0.1 samples, leaving the
    # power envelope unsmoothed and gap-merging disabled, which fragments one
    # burst into pieces. The largest fragment then has too few symbols for the
    # eye histogram in fsk_tones, and narrowband 4-FSK silently degrades to
    # 2 tones. Floors keep the detector meaningful at any sample rate.
    n_win = max(4, int(round(win_s * fs)))
    p = _smooth(np.abs(x).astype(np.float64) ** 2, n_win)
    p_db = 10.0 * np.log10(p + 1e-20)

    noise_db = _noise_floor_db(p_db)
    peak_db = float(np.percentile(p_db, 99.9))
    if peak_db - noise_db < 3.0:
        # no discernible on/off structure -> treat as continuous
        return [Burst(0, len(x), fs)], noise_db

    hi = p_db > (noise_db + on_db)
    lo = p_db > (noise_db + off_db)

    # hysteresis: grow every hi region outward while lo holds
    state = np.zeros(len(p_db), dtype=bool)
    idx = np.flatnonzero(hi)
    if idx.size == 0:
        return [Burst(0, len(x), fs)], noise_db
    edges = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[edges + 1]]
    stops = np.r_[idx[edges], idx[-1]]
    for a, b in zip(starts, stops):
        while a > 0 and lo[a - 1]:
            a -= 1
        while b < len(lo) - 1 and lo[b + 1]:
            b += 1
        state[a : b + 1] = True

    # runs -> bursts, merging short gaps
    d = np.diff(state.astype(np.int8))
    ons = list(np.flatnonzero(d == 1) + 1)
    offs = list(np.flatnonzero(d == -1) + 1)
    if state[0]:
        ons.insert(0, 0)
    if state[-1]:
        offs.append(len(state))

    max_gap = max(8, int(round(max_gap_s * fs)))
    merged: list[list[int]] = []
    for a, b in zip(ons, offs):
        if merged and a - merged[-1][1] <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    min_len = max(16, int(round(min_dur_s * fs)))
    bursts = []
    for a, b in merged:
        if b - a < min_len:
            continue
        g = int((b - a) * guard_frac)  # trim ramps
        bursts.append(Burst(a + g, b - g, fs))
    if not bursts:
        return [Burst(0, len(x), fs)], noise_db

    # many fragments covering nearly everything => continuous carrier that the
    # hysteresis chopped up; a single burst with quiet margins is left alone.
    covered = sum(b.i1 - b.i0 for b in bursts)
    if len(bursts) > 2 and covered > 0.85 * len(x):
        return [Burst(0, len(x), fs)], noise_db
    return bursts, noise_db


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def welch_psd(x: np.ndarray, fs: float, nperseg: Optional[int] = None):
    n = len(x)
    if nperseg is None:
        nperseg = int(2 ** np.floor(np.log2(max(64, min(4096, n // 4 or 64)))))
    nperseg = min(nperseg, n)
    f, pxx = signal.welch(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        return_onesided=False, detrend=False, scaling="density",
    )
    order = np.argsort(f)
    return f[order], pxx[order]


def occupied_bandwidth(f: np.ndarray, pxx: np.ndarray, frac: float = 0.99):
    """Return (bandwidth, center) containing `frac` of total power."""
    total = np.sum(pxx)
    if total <= 0:
        return 0.0, 0.0
    c = np.cumsum(pxx) / total
    tail = (1.0 - frac) / 2.0
    lo = float(np.interp(tail, c, f))
    hi = float(np.interp(1.0 - tail, c, f))
    return hi - lo, 0.5 * (lo + hi)


def rms_bandwidth(f: np.ndarray, pxx: np.ndarray,
                  bw: Optional[float] = None, center: float = 0.0) -> float:
    """RMS (2nd-moment) bandwidth, restricted to the occupied band.

    Integrating over the whole capture would fold the out-of-band noise floor
    into the second moment and report a value several times too large for a
    narrowband burst in a wide capture.
    """
    if bw:
        m = np.abs(f - center) <= bw
        if m.sum() >= 8:
            f, pxx = f[m], pxx[m]
    w = pxx / (np.sum(pxx) + 1e-30)
    mu = float(np.sum(w * f))
    return 2.0 * math.sqrt(max(0.0, float(np.sum(w * (f - mu) ** 2))))


def spectral_flatness(pxx: np.ndarray, f: np.ndarray, bw: float, center: float) -> float:
    """Geometric/arithmetic mean ratio inside the occupied band.

    ~1.0 for OFDM (flat), lower for shaped single-carrier / FSK humps.
    """
    m = np.abs(f - center) <= (bw / 2.0)
    if m.sum() < 8:
        return 0.0
    p = pxx[m] + 1e-30
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def edge_sharpness(pxx: np.ndarray, f: np.ndarray, bw: float, center: float) -> float:
    """dB of roll-off per 10% of BW just outside the band edge.

    High for brick-wall (OFDM, filtered), low for slowly decaying spectra.
    """
    step = bw * 0.10
    def band_db(lo, hi):
        m = (f >= lo) & (f <= hi)
        return 10 * np.log10(np.mean(pxx[m]) + 1e-30) if m.sum() else np.nan
    inner = band_db(center - bw * 0.3, center + bw * 0.3)
    outer = band_db(center + bw / 2 + 0.02 * bw, center + bw / 2 + step)
    outer2 = band_db(center - bw / 2 - step, center - bw / 2 - 0.02 * bw)
    vals = [v for v in (inner - outer, inner - outer2) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Modulation-specific measurements
# ---------------------------------------------------------------------------

def inst_freq(x: np.ndarray, fs: float, mag_gate: float = 0.35) -> np.ndarray:
    """Instantaneous frequency (Hz), gated to samples above `mag_gate` * median |x|."""
    ph = np.unwrap(np.angle(x.astype(np.complex128)))
    f = np.diff(ph) * fs / (2 * np.pi)
    a = np.abs(x[1:])
    gate = mag_gate * np.median(np.abs(x)) if np.median(np.abs(x)) > 0 else 0.0
    return f[a > gate]


def cp_autocorrelation(x: np.ndarray, min_lag: int, max_lag: int):
    """Normalised |R(lag)| via FFT. OFDM cyclic prefixes create an isolated peak
    at lag == FFT size; repeated training symbols also show up here."""
    n = len(x)
    max_lag = int(min(max_lag, n // 3))
    if max_lag <= min_lag:
        return None
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.fft(x.astype(np.complex128), nfft)
    r = np.fft.ifft(X * np.conj(X))[: max_lag + 1]
    r = np.abs(r) / (abs(r[0]) + 1e-30)
    lags = np.arange(min_lag, max_lag + 1)
    vals = r[min_lag : max_lag + 1]
    if vals.size < 8:
        return None
    k = int(np.argmax(vals))
    if k == 0 or k == vals.size - 1:
        return None                      # peak pinned to the search boundary
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-12
    return {
        "lag": int(lags[k]),
        "value": float(vals[k]),
        "score": float((vals[k] - med) / mad),   # robust z-score
        "curve": (lags, vals),
    }


def estimate_cp_length(x: np.ndarray, n_fft: int) -> Optional[int]:
    """Given a candidate FFT size, find the CP length that maximises the
    sliding-window correlation periodicity."""
    best, best_cp = -1.0, None
    for cp in (n_fft // 32, n_fft // 16, n_fft // 8, n_fft // 4):
        if cp < 2 or len(x) < 3 * (n_fft + cp):
            continue
        prod = x[:-n_fft] * np.conj(x[n_fft:])
        w = np.ones(cp) / cp
        g = np.abs(np.convolve(prod, w, mode="valid"))
        if g.size < 4:
            continue
        contrast = float(np.percentile(g, 98) / (np.median(g) + 1e-30))
        if contrast > best:
            best, best_cp = contrast, cp
    return best_cp


LORA_BANDWIDTHS = (7.8e3, 10.4e3, 15.6e3, 20.8e3, 31.25e3, 41.7e3,
                   62.5e3, 125e3, 250e3, 500e3)


def detect_chirp(x: np.ndarray, fs: float, bw_hint: Optional[float] = None):
    """Chirp-spread-spectrum (LoRa) detector by dechirping.

    Pointwise d(f)/dt is useless here: at 25 dB SNR the discriminator noise on
    the slope is the same order as the slope itself. Instead we multiply by a
    conjugate reference chirp for each physically plausible (BW, SF) pair and
    measure how much of the per-symbol FFT energy lands in one bin. A matched
    dechirp collapses each symbol to a tone (two tones if the symbol boundary
    is misaligned, which still concentrates far more than anything non-CSS).
    """
    if bw_hint is None or bw_hint <= 0 or x.size < 512:
        return None

    cands = [b for b in LORA_BANDWIDTHS if 0.55 * bw_hint <= b <= 1.5 * bw_hint]
    cands.append(float(bw_hint))                       # non-standard chirps

    best = None
    for bw in cands:
        for sf in range(5, 13):
            n = int(round(2 ** sf * fs / bw))           # samples per symbol
            if n < 32 or 3 * n > x.size:
                continue
            k = bw * bw / (2 ** sf)                     # chirp rate, Hz/s
            idx = np.arange(x.size)
            tt = (idx % n) / fs                         # sawtooth time base
            y = x * np.exp(-1j * np.pi * k * tt * tt)
            nblk = min(x.size // n, 48)
            if nblk < 3:
                continue
            blocks = y[: nblk * n].reshape(nblk, n)
            S = np.abs(np.fft.fft(blocks, axis=1)) ** 2
            frac = S.max(axis=1) / (S.sum(axis=1) + 1e-30)
            score = float(np.median(frac))
            if best is None or score > best["score"]:
                best = {"score": score, "bw": float(bw), "sf": int(sf),
                        "chirp_rate": float(k), "sym_len": n}
    return best


def _tone_peaks(v: np.ndarray, nbins: Optional[int] = None):
    """Histogram a set of frequency samples and return (peaks, prominences)."""
    v = v - np.median(v)
    if v.size < 40:
        return None
    if nbins is None:
        nbins = int(np.clip(int(np.sqrt(v.size) * 2), 33, 129))
    span = 2.2 * float(np.percentile(np.abs(v), 97)) or 1.0
    h, e = np.histogram(v, bins=np.linspace(-span, span, nbins))
    h = _smooth(h.astype(float), 3)
    if h.max() <= 0:
        return None
    h = h / h.max()
    c = 0.5 * (e[:-1] + e[1:])
    pk, props = signal.find_peaks(h, prominence=0.20, distance=2)
    if pk.size == 0:
        return None
    o = np.argsort(c[pk])
    return c[pk][o], h[pk][o]


def _validate_tones(tones: np.ndarray, heights: np.ndarray) -> bool:
    """M-FSK tones are uniformly spaced and of comparable occupancy."""
    if tones.size < 2:
        return False
    if heights.min() / heights.max() < 0.35:
        return False
    if tones.size >= 3:
        d = np.diff(tones)
        if np.std(d) / (np.mean(d) + 1e-30) > 0.30:
            return False
    return True


def fsk_tones(x: np.ndarray, fs: float, bw_hint: Optional[float] = None,
              symbol_rate: Optional[float] = None):
    """Count instantaneous-frequency modes -> 2-FSK / 4-FSK / M-FSK.

    Blanket-averaging the discriminator straddles symbol transitions and merges
    the inner levels of a 4-FSK signal back together. With a symbol rate in
    hand we can instead sample at the eye's decision points: the phase that
    maximises the sampled variance is the open eye, and the histogram of those
    samples separates all M levels cleanly. The blanket histogram is retained
    as a fallback for when the symbol rate is unknown or unreliable.

    A peak set is only accepted as M-FSK if it is uniformly spaced with
    comparable level occupancy — that is what separates real M-FSK from the
    accidental bumps a linear modulation's discriminator produces.
    """
    f_raw = inst_freq(x, fs)
    if f_raw.size < 256:
        return None
    span_hint = bw_hint if bw_hint else 2.0 * np.percentile(np.abs(f_raw), 98)

    cands = []   # (n_tones, tones, heights, method)

    if symbol_rate and symbol_rate > 0:
        sps = fs / symbol_rate
        if 2.0 <= sps <= 4096.0:
            # Sweep the pre-smoothing length rather than deriving one value. A
            # single derived window is fragile: on narrowband 4-FSK, w=round(sps/4)
            # resolved only 2 of the 4 levels while w=3 resolved all four, and
            # which one wins shifts with the noise realisation. Widening the set
            # is safe because selection below only accepts peak sets that are
            # uniformly spaced with comparable occupancy — a bad window yields an
            # invalid set and is discarded rather than competing.
            cand_w = {1, 2, 3, int(round(sps / 6)), int(round(sps / 4)),
                      int(round(sps / 3)), int(round(sps / 2))}
            for w in sorted(x for x in cand_w if 1 <= x <= min(16, max(1, sps - 1))):
                f = _smooth(f_raw, w)
                best = None
                for ph in range(max(1, int(round(sps)))):
                    idx = np.round(np.arange(ph, f.size - 1, sps)).astype(int)
                    idx = idx[idx < f.size]
                    if idx.size < 40:
                        continue
                    v = f[idx]
                    if best is None or v.var() > best[0]:
                        best = (v.var(), v)
                if best is None:
                    continue
                got = _tone_peaks(best[1])
                if got:
                    cands.append((got[0].size, got[0], got[1], f"eye(w={w})"))

    # blanket histogram fallback
    w = int(np.clip(round(0.5 * (fs / symbol_rate if symbol_rate else fs / max(span_hint, 1.0))), 1, 64))
    got = _tone_peaks(_smooth(f_raw, w), nbins=129)
    blanket = (got[0].size, got[0], got[1], f"blanket(w={w})") if got else None
    if blanket:
        cands.append(blanket)

    valid = [c for c in cands if c[0] in (2, 4, 8) and _validate_tones(c[1], c[2])]

    def _rank(c):
        """Prefer more levels, but only when those levels are well occupied.

        Sweeping windows means a noisy one can occasionally fake a uniform
        4-tone set on a genuine 2-FSK signal, and 'prefer larger M' alone would
        promote it. Requiring a min/max occupancy above 0.5 before a higher M
        outranks a clean 2-tone answer costs nothing on real M-ary FSK, whose
        levels are equiprobable by construction.
        """
        n, tones, heights, _ = c
        occ = float(heights.min() / heights.max())
        return (n if (n == 2 or occ > 0.5) else 0, occ)

    pick = max(valid, key=_rank) if valid else (blanket or (cands[0] if cands else None))
    if pick is None:
        return {"n_tones": 0, "tone_freqs": [], "deviation": float(np.std(f_raw)),
                "uniform": False, "method": "none"}

    n, tones, heights, method = pick
    # Re-derive validity rather than testing `pick in valid`: these candidates
    # are tuples containing ndarrays, so `in` falls back to element-wise `==`
    # and raises "truth value of an array is ambiguous" as soon as the list holds
    # more than one entry. It only ever appeared to work because `valid` was
    # usually a single item.
    pick_valid = bool(n in (2, 4, 8) and _validate_tones(tones, heights))
    d = np.diff(tones)
    dev = float(tones[1] - tones[0]) if n == 2 else (float(np.median(d)) if d.size else 0.0)
    return {"n_tones": int(n), "tone_freqs": tones.tolist(), "deviation": dev,
            "uniform": pick_valid, "method": method}


def estimate_pulse_rate(x: np.ndarray, fs: float):
    """Chip/pulse rate of an on-off keyed signal from envelope run lengths.

    Far more reliable than a spectral line for short pulsed frames: a 112-bit
    ADS-B burst is only ~1000 samples, which is too few bins for the
    cyclostationary method to beat its own noise.
    """
    a = np.abs(x).astype(np.float64)
    thr = 0.5 * float(np.percentile(a, 95))
    if thr <= 0:
        return None
    b = a > thr
    if b.all() or not b.any():
        return None
    idx = np.flatnonzero(np.diff(b.astype(np.int8)) != 0)
    if idx.size < 6:
        return None
    runs = np.diff(idx).astype(float)
    r = float(np.percentile(runs, 10))
    if r < 2.0:
        return None
    return {"symbol_rate": fs / r, "line_snr": 8.0, "method": "envelope-run-length"}


def _spectral_lines(s: np.ndarray, fs: float, f_lo: float, f_hi: float):
    """FFT of a cyclostationary probe; return (freqs, magnitudes, noise level)."""
    n = 1 << int(np.floor(np.log2(max(s.size, 2))))
    if n < 256:
        return None
    s = s[:n].astype(np.float64)
    s = (s - s.mean()) * np.hanning(n)
    S = np.abs(np.fft.rfft(s))
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    m = (fr >= max(f_lo, fs / n * 6)) & (fr <= f_hi)
    if m.sum() < 32:
        return None
    return fr[m], S[m], float(np.median(S[m])) + 1e-30


def _fundamental(fr, S, noise, snr_min: float = 3.0):
    """Strongest spectral line, walked down to its fundamental.

    The walk is *chained*: to accept Rs/4 we must first have accepted Rs/2.
    Allowing a direct jump lets a single noise bin several octaves down hijack
    the estimate, which is how a 1 Mchip/s PPM frame ends up reported as
    250 kHz.
    """
    k = int(np.argmax(S))
    f0, best_snr = float(fr[k]), float(S[k] / noise)
    if best_snr < snr_min:
        return None
    need = max(6.0, 0.5 * best_snr)
    for _ in range(4):                          # at most Rs/16
        moved = False
        for div in (2, 3):
            f_try = f0 / div
            if f_try < fr[0]:
                continue
            j = int(np.argmin(np.abs(fr - f_try)))
            w = max(1, int(0.02 * j))
            lo = max(0, j - w)
            seg = S[lo : j + w + 1]
            if seg.size and float(seg.max() / noise) > need:
                f0 = float(fr[lo + int(np.argmax(seg))])
                moved = True
                break
        if not moved:
            break
    return f0, best_snr


def estimate_symbol_rate(x: np.ndarray, fs: float, bw_hint: Optional[float] = None):
    """Spectral-line symbol-rate estimator.

    Three cyclostationary probes are tried and the strongest line wins:
      (a) sign transitions of the FM discriminator -> FSK / GFSK / MSK
      (b) magnitude of the discriminator derivative -> shaped FSK
      (c) squared envelope -> linear mods, OOK, pulsed
    Sub-harmonics are checked so a strong 2*Rs line does not double the answer.
    """
    f_lo = (bw_hint / 60.0) if bw_hint else fs / 5000.0
    f_hi = (bw_hint * 1.6) if bw_hint else fs / 2.0

    probes = []
    f = inst_freq(x, fs)
    if f.size > 512:
        d = np.sign(f - np.median(f))
        probes.append(("fm-sign-transitions", np.abs(np.diff(d))))
        probes.append(("fm-derivative", np.abs(np.diff(f))))
    a2 = np.abs(x).astype(np.float64) ** 2
    if a2.size > 512:
        probes.append(("squared-envelope", a2))

    best = None
    for name, probe in probes:
        got = _spectral_lines(probe, fs, f_lo, f_hi)
        if got is None:
            continue
        fr, S, noise = got
        fund = _fundamental(fr, S, noise)
        if fund is None:
            continue
        rate, snr = fund
        if best is None or snr > best["line_snr"]:
            best = {"symbol_rate": rate, "line_snr": snr, "method": name}
    return best


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------

@dataclass
class Features:
    t0: float
    duration: float
    snr_db: float
    bw99: float
    bw_rms: float
    cfo: float
    envelope_cv: float
    papr_db: float
    flatness: float
    edge_db: float
    modclass: str = "unknown"
    symbol_rate: Optional[float] = None
    symbol_rate_snr: Optional[float] = None
    n_tones: Optional[int] = None
    fsk_deviation: Optional[float] = None
    tones_uniform: Optional[bool] = None
    decimation: int = 1
    ofdm_fft: Optional[int] = None
    ofdm_cp: Optional[int] = None
    ofdm_score: Optional[float] = None
    subcarrier_spacing: Optional[float] = None
    chirp_rate: Optional[float] = None
    chirp_score: Optional[float] = None
    css_bw: Optional[float] = None
    css_sf: Optional[int] = None
    notes: list[str] = field(default_factory=list)


def extract_features(x: np.ndarray, fs: float, burst: Burst, noise_db: float) -> Features:
    seg = x[burst.i0 : burst.i1]
    if seg.size < 64:
        raise ValueError("burst too short for analysis")

    f, pxx = welch_psd(seg, fs)
    bw99, center = occupied_bandwidth(f, pxx, 0.99)
    bwr = rms_bandwidth(f, pxx, bw=bw99, center=center)

    # De-rotate to the measured centre, then channel-filter and decimate to
    # ~4x the occupied bandwidth. Discarding out-of-band noise is the single
    # cheapest SNR win available (often ~9 dB on a narrowband burst in a wide
    # capture) and it makes sample-count-based smoothing windows meaningful.
    n = np.arange(seg.size)
    seg_c = seg * np.exp(-2j * np.pi * center * n / fs)
    fs_a = fs
    # Target ~6x the occupied bandwidth. Going tighter than that starves the
    # discriminator of samples per symbol, which costs more than the filtering
    # gains: at 4 samples/symbol a GFSK eye stops resolving reliably.
    decim = int(max(1, np.floor(fs / (6.0 * max(bw99, 1.0)))))
    if decim >= 3 and seg_c.size // decim >= 512:
        taps = min(129, (seg_c.size // 4) | 1)
        h = signal.firwin(taps, 1.0 / decim, window=("kaiser", 6.0))
        seg_c = signal.lfilter(h, 1.0, seg_c)[::decim].astype(np.complex64)
        fs_a = fs / decim
        ft_notes_decim = f"channel-filtered and decimated {decim}x -> {fs_a/1e6:.4g} Msps"
    else:
        decim, ft_notes_decim = 1, None

    a = np.abs(seg).astype(np.float64)
    cv = float(np.std(a) / (np.mean(a) + 1e-30))
    papr = 10.0 * math.log10((np.max(a) ** 2) / (np.mean(a**2) + 1e-30) + 1e-30)
    sig_db = 10.0 * math.log10(np.mean(a**2) + 1e-30)

    ft = Features(
        t0=burst.t0,
        duration=burst.duration,
        snr_db=sig_db - noise_db,
        bw99=bw99,
        bw_rms=bwr,
        cfo=center,
        envelope_cv=cv,
        papr_db=papr,
        flatness=spectral_flatness(pxx, f, bw99, center),
        edge_db=edge_sharpness(pxx, f, bw99, center),
    )

    # --- OFDM ---
    min_lag = max(8, int(fs_a / max(bw99, 1.0) * 4))
    ac = cp_autocorrelation(seg_c, min_lag=min_lag, max_lag=4096)
    if ac:
        ft.ofdm_score = ac["score"]
        if ac["score"] > 10.0 and ac["value"] > 0.03 and ac["lag"] >= 16:
            ft.ofdm_fft = ac["lag"]
            ft.ofdm_cp = estimate_cp_length(seg_c, ac["lag"])
            ft.subcarrier_spacing = fs_a / ac["lag"]

    # --- chirp / CSS ---
    ch = detect_chirp(seg_c, fs_a, bw_hint=bw99)
    if ch:
        ft.chirp_rate = ch["chirp_rate"]
        ft.chirp_score = ch["score"]
        ft.css_bw = ch["bw"]
        ft.css_sf = ch["sf"]

    # --- symbol / chip rate (needed before tone counting) ---
    sr = estimate_symbol_rate(seg_c, fs_a, bw_hint=bw99)
    if cv > 0.60:                                   # pulsed: trust run lengths
        pr = estimate_pulse_rate(seg_c, fs_a)
        if pr and (sr is None or pr["line_snr"] >= (sr["line_snr"] or 0) * 0.5):
            sr = pr
    if sr:
        ft.symbol_rate = sr["symbol_rate"]
        ft.symbol_rate_snr = sr["line_snr"]
        ft.notes.append(f"symbol rate from {sr['method']}")

    # --- FSK tones ---
    tn = fsk_tones(seg_c, fs_a, bw_hint=bw99, symbol_rate=ft.symbol_rate)
    if tn:
        ft.n_tones = tn["n_tones"]
        ft.fsk_deviation = tn["deviation"]
        ft.tones_uniform = tn.get("uniform")
        ft.notes.append(f"tone count via {tn.get('method')}")

    if ft_notes_decim:
        ft.notes.append(ft_notes_decim)
    ft.decimation = decim
    ft.modclass = _decide_modclass(ft)
    return ft


def _decide_modclass(ft: Features) -> str:
    """Coarse modulation family from the measured features."""
    # CSS first: a matched dechirp is close to unambiguous
    if ft.chirp_score is not None and ft.chirp_score > 0.15 and ft.envelope_cv < 0.45:
        return "css"

    # OFDM: cyclic-prefix autocorrelation + varying envelope + flat spectrum
    if ft.ofdm_fft and ft.envelope_cv > 0.30 and ft.flatness > 0.25:
        return "ofdm"

    # Pulsed / OOK: large envelope swing with no CP structure
    if ft.envelope_cv > 0.70 and ft.papr_db > 2.0 and not ft.ofdm_fft:
        return "ook-pulsed"

    const_env = ft.envelope_cv < 0.22
    tones_ok = bool(ft.tones_uniform) and ft.n_tones is not None

    if const_env or (ft.envelope_cv < 0.25 and tones_ok):
        if tones_ok and ft.n_tones == 2:
            return "fsk2"
        if tones_ok and ft.n_tones in (3, 4):
            return "fsk4"
        if tones_ok and ft.n_tones and ft.n_tones > 4:
            return "fsk-multi"
        if const_env:
            return "const-env-phase"        # PSK / MSK / GMSK / OQPSK / analog FM

    return "linear-shaped"                  # PSK / QAM with pulse shaping


# ---------------------------------------------------------------------------
# Protocol database
# ---------------------------------------------------------------------------
# bw / rate / dur are (min, max) inclusive ranges; None = don't care.
# band entries are RF centre-frequency windows in Hz.

MHz = 1e6
kHz = 1e3

PROTOCOLS: list[dict] = [
    # ---- 2.4 GHz ISM ----
    dict(name="Wi-Fi 802.11b/g (DSSS/CCK)", mod=["dsss-psk", "linear-shaped", "const-env-phase"],
         bw=(14 * MHz, 24 * MHz), rate=(0.9e6, 1.5e6), dur=(20e-6, 5e-3),
         band=[(2.40e9, 2.50e9)], spec="IEEE 802.11-2020 cl.16/17"),
    dict(name="Wi-Fi 802.11g/n/ax 20 MHz (OFDM)", mod=["ofdm"],
         bw=(15 * MHz, 20.5 * MHz), rate=None, dur=(20e-6, 6e-3),
         band=[(2.40e9, 2.50e9), (5.15e9, 5.90e9), (5.925e9, 7.125e9)],
         scs=(300 * kHz, 330 * kHz), spec="IEEE 802.11-2020 cl.17/19/27"),
    dict(name="Wi-Fi 802.11n/ac/ax 40 MHz", mod=["ofdm"],
         bw=(33 * MHz, 41 * MHz), rate=None, dur=(20e-6, 6e-3),
         band=[(2.40e9, 2.50e9), (5.15e9, 7.125e9)],
         scs=(300 * kHz, 330 * kHz), spec="IEEE 802.11-2020"),
    dict(name="Wi-Fi 802.11ac/ax 80 MHz", mod=["ofdm"],
         bw=(70 * MHz, 82 * MHz), rate=None, dur=(20e-6, 6e-3),
         band=[(5.15e9, 7.125e9)], scs=(300 * kHz, 330 * kHz), spec="IEEE 802.11-2020"),
    dict(name="Bluetooth Classic (GFSK, BR)", mod=["fsk2", "const-env-phase"],
         bw=(0.7 * MHz, 1.3 * MHz), rate=(0.85e6, 1.15e6), dur=(100e-6, 3.0e-3),
         band=[(2.402e9, 2.480e9)], dev=(140 * kHz, 175 * kHz),
         spec="Bluetooth Core 5.4, Vol 6"),
    dict(name="Bluetooth LE 1M (GFSK)", mod=["fsk2", "const-env-phase"],
         bw=(0.8 * MHz, 1.4 * MHz), rate=(0.9e6, 1.1e6), dur=(40e-6, 400e-6),
         band=[(2.402e9, 2.480e9)], dev=(225 * kHz, 275 * kHz),
         spec="Bluetooth Core 5.4, Vol 6 Part B"),
    dict(name="Bluetooth LE 2M (GFSK)", mod=["fsk2", "const-env-phase"],
         bw=(1.6 * MHz, 2.6 * MHz), rate=(1.8e6, 2.2e6), dur=(25e-6, 250e-6),
         band=[(2.402e9, 2.480e9)], dev=(450 * kHz, 550 * kHz),
         spec="Bluetooth Core 5.4"),
    dict(name="Zigbee / 802.15.4 O-QPSK DSSS", mod=["const-env-phase", "linear-shaped"],
         bw=(1.8 * MHz, 3.0 * MHz), rate=(1.8e6, 2.2e6), dur=(200e-6, 4.5e-3),
         band=[(2.405e9, 2.485e9)], spec="IEEE 802.15.4-2020 cl.12"),
    dict(name="Nordic/ANT-class 1 Mbps GFSK", mod=["fsk2"],
         bw=(0.8 * MHz, 1.4 * MHz), rate=(0.9e6, 1.1e6), dur=(30e-6, 500e-6),
         band=[(2.400e9, 2.485e9)], dev=(140 * kHz, 200 * kHz),
         spec="ANT Message Protocol / nRF24 datasheet"),

    # ---- sub-GHz ISM ----
    dict(name="LoRa / CSS (LoRaWAN, Meshtastic, ELRS)", mod=["css"],
         bw=(100 * kHz, 550 * kHz), rate=None, dur=(1e-3, 3.0),
         band=[(150e6, 960e6), (2.400e9, 2.485e9)],
         spec="LoRa Alliance regional params; PHY is Semtech proprietary"),
    dict(name="SiK / MAVLink telemetry (2-FSK/GFSK)", mod=["fsk2"],
         bw=(60 * kHz, 400 * kHz), rate=(30e3, 260e3), dur=(1e-3, 60e-3),
         band=[(410e6, 480e6), (863e6, 928e6), (2.400e9, 2.485e9)],
         dev=(20 * kHz, 130 * kHz), spec="SiK firmware (Si1000/Si4432), MAVLink v2"),
    dict(name="Z-Wave (G.9959 GFSK)", mod=["fsk2"],
         bw=(30 * kHz, 400 * kHz), rate=(9e3, 110e3), dur=(1e-3, 40e-3),
         band=[(865e6, 927e6)], spec="ITU-T G.9959"),
    dict(name="Wi-SUN / 802.15.4g 2-FSK", mod=["fsk2"],
         bw=(100 * kHz, 800 * kHz), rate=(45e3, 310e3), dur=(1e-3, 50e-3),
         band=[(470e6, 960e6)], spec="IEEE 802.15.4g / Wi-SUN FAN 1.1"),
    dict(name="Generic ISM OOK remote (rtl_433 class)", mod=["ook-pulsed"],
         bw=(10 * kHz, 400 * kHz), rate=(200.0, 20e3), dur=(0.5e-3, 200e-3),
         band=[(300e6, 450e6), (860e6, 930e6)], spec="device-specific; see rtl_433"),
    dict(name="Sigfox uplink (DBPSK, UNB)", mod=["linear-shaped", "const-env-phase"],
         bw=(50.0, 400.0), rate=(90.0, 650.0), dur=(1.0, 3.0),
         band=[(862e6, 928e6)], spec="Sigfox radio specification"),
    dict(name="mioty (TS-UNB telegram splitting)", mod=["fsk2", "const-env-phase"],
         bw=(1 * kHz, 30 * kHz), rate=(2e3, 6e3), dur=(0.3, 2.0),
         band=[(862e6, 928e6)], spec="ETSI TS 103 357"),

    # ---- aviation / maritime ----
    # PPM encodes each 1 us chip as a 0.5 us pulse, so a run-length estimator
    # legitimately reports ~2 MHz rather than the 1 Mchip/s frame rate. The window
    # spans both readings; burst duration is the feature that actually separates a
    # 120 us extended squitter from a 20 us Mode A/C/S reply.
    dict(name="ADS-B 1090ES (PPM/OOK)", mod=["ook-pulsed"],
         bw=(1.0 * MHz, 8.0 * MHz), rate=(0.9e6, 2.6e6), dur=(50e-6, 130e-6),
         band=[(1089e6, 1091e6)], spec="RTCA DO-260B / ICAO Annex 10"),
    dict(name="Mode A/C/S interrogation & reply", mod=["ook-pulsed"],
         bw=(1.0 * MHz, 8.0 * MHz), rate=(0.5e6, 2.2e6), dur=(15e-6, 50e-6),
         band=[(1029e6, 1091e6)], spec="ICAO Annex 10 Vol IV"),
    dict(name="AIS (GMSK 9.6 kbps)", mod=["const-env-phase", "fsk2"],
         bw=(9 * kHz, 25 * kHz), rate=(9e3, 10.2e3), dur=(20e-3, 30e-3),
         band=[(161.9e6, 162.1e6), (156e6, 163e6)], spec="ITU-R M.1371-5"),
    dict(name="FLARM / OGN (GFSK 100 kbps)", mod=["fsk2"],
         bw=(150 * kHz, 400 * kHz), rate=(90e3, 110e3), dur=(0.3e-3, 2e-3),
         band=[(868e6, 869e6)], spec="FLARM DDL (partly reverse-engineered)"),

    # ---- LMR / paging ----
    dict(name="DMR (4-FSK, 12.5 kHz)", mod=["fsk4"],
         bw=(8 * kHz, 14 * kHz), rate=(4.5e3, 5.1e3), dur=(20e-3, 0.5),
         band=[(130e6, 950e6)], spec="ETSI TS 102 361"),
    dict(name="P25 Phase 1 (C4FM)", mod=["fsk4"],
         bw=(8 * kHz, 14 * kHz), rate=(4.5e3, 5.1e3), dur=(20e-3, 5.0),
         band=[(130e6, 900e6)], spec="TIA-102.BAAA"),
    dict(name="TETRA (pi/4-DQPSK)", mod=["linear-shaped"],
         bw=(20 * kHz, 30 * kHz), rate=(17e3, 19e3), dur=(10e-3, 5.0),
         band=[(380e6, 470e6)], spec="ETSI EN 300 392-2"),
    dict(name="POCSAG paging (2-FSK)", mod=["fsk2"],
         bw=(6 * kHz, 25 * kHz), rate=(480.0, 2500.0), dur=(50e-3, 5.0),
         band=[(130e6, 960e6)], spec="ITU-R M.584"),
    dict(name="Analog FM voice / NBFM", mod=["fsk-multi", "const-env-phase"],
         bw=(8 * kHz, 20 * kHz), rate=None, dur=(50e-3, 60.0),
         band=[(25e6, 1300e6)], spec="n/a (analog)"),

    # ---- cellular / cordless ----
    dict(name="LTE / 5G NR downlink (OFDM, 15 kHz SCS)", mod=["ofdm"],
         bw=(1.2 * MHz, 100 * MHz), rate=None, dur=(0.5e-3, 60.0),
         band=[(400e6, 6.0e9)], scs=(13 * kHz, 17 * kHz),
         spec="3GPP TS 36.211 / 38.211"),
    dict(name="5G NR (OFDM, 30 kHz SCS)", mod=["ofdm"],
         bw=(5 * MHz, 100 * MHz), rate=None, dur=(0.25e-3, 60.0),
         band=[(400e6, 7.2e9)], scs=(27 * kHz, 33 * kHz), spec="3GPP TS 38.211"),
    dict(name="GSM / GMSK burst", mod=["const-env-phase", "fsk2"],
         bw=(180 * kHz, 300 * kHz), rate=(265e3, 275e3), dur=(400e-6, 700e-6),
         band=[(380e6, 1.99e9)], spec="3GPP TS 45.004"),
    dict(name="DECT (GFSK 1.152 Msym)", mod=["fsk2", "const-env-phase"],
         bw=(1.0 * MHz, 1.9 * MHz), rate=(1.05e6, 1.25e6), dur=(300e-6, 500e-6),
         band=[(1.88e9, 1.93e9)], spec="ETSI EN 300 175-2"),

    # ---- broadcast ----
    dict(name="DVB-T/T2 (OFDM)", mod=["ofdm"],
         bw=(6.5 * MHz, 8.2 * MHz), rate=None, dur=(1e-3, 60.0),
         band=[(470e6, 862e6)], scs=(0.9e3, 4.5e3), spec="ETSI EN 300 744 / 302 755"),
    dict(name="DAB+ (OFDM, 1 kHz SCS)", mod=["ofdm"],
         bw=(1.4 * MHz, 1.6 * MHz), rate=None, dur=(1e-3, 60.0),
         band=[(174e6, 240e6)], scs=(0.8e3, 1.2e3), spec="ETSI EN 300 401"),
    dict(name="FM broadcast (WBFM)", mod=["fsk-multi", "const-env-phase"],
         bw=(150 * kHz, 260 * kHz), rate=None, dur=(50e-3, 60.0),
         band=[(87e6, 109e6)], spec="ITU-R BS.450"),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

MOD_COMPAT = {
    # detected -> {db family: score}
    "ofdm": {"ofdm": 1.0, "linear-shaped": 0.15},
    "css": {"css": 1.0},
    "fsk2": {"fsk2": 1.0, "const-env-phase": 0.55, "fsk4": 0.15, "fsk-multi": 0.3},
    "fsk4": {"fsk4": 1.0, "fsk-multi": 0.6, "fsk2": 0.25, "const-env-phase": 0.4},
    "fsk-multi": {"fsk-multi": 1.0, "fsk4": 0.5, "fsk2": 0.3, "const-env-phase": 0.5},
    "const-env-phase": {"const-env-phase": 1.0, "fsk2": 0.55, "fsk4": 0.4,
                        "linear-shaped": 0.5, "dsss-psk": 0.6, "fsk-multi": 0.4},
    "linear-shaped": {"linear-shaped": 1.0, "dsss-psk": 0.8,
                      "const-env-phase": 0.5, "ofdm": 0.2},
    "ook-pulsed": {"ook-pulsed": 1.0, "linear-shaped": 0.15},
    "unknown": {},
}


def _range_score(value: Optional[float], rng, tol_octaves: float = 1.0) -> Optional[float]:
    """1.0 inside the range, decaying log-linearly outside it."""
    if value is None or rng is None:
        return None
    lo, hi = rng
    if lo <= value <= hi:
        return 1.0
    ref = lo if value < lo else hi
    if value <= 0 or ref <= 0:
        return 0.0
    octaves = abs(math.log2(value / ref))
    return float(max(0.0, 1.0 - octaves / tol_octaves))


def score_protocol(ft: Features, proto: dict, rf_center_hz: Optional[float]) -> dict:
    reasons: list[str] = []
    terms: list[tuple[float, float]] = []   # (score, weight)

    # band gate — hard filter when the tuner frequency is known
    if rf_center_hz is not None:
        ok = any(lo <= rf_center_hz <= hi for lo, hi in proto["band"])
        if not ok:
            return dict(name=proto["name"], score=0.0,
                        reasons=["RF centre outside allocated band"], spec=proto["spec"])

    m = MOD_COMPAT.get(ft.modclass, {})
    mod_gate = max((m.get(k, 0.0) for k in proto["mod"]), default=0.0)
    reasons.append(f"modulation {ft.modclass} vs {'/'.join(proto['mod'])}: "
                   f"gate x{mod_gate:.2f}")
    if mod_gate <= 0.0:
        return dict(name=proto["name"], score=0.0, reasons=reasons, spec=proto["spec"])

    bw_s = _range_score(ft.bw99, proto["bw"], tol_octaves=1.0)
    if bw_s is not None:
        terms.append((bw_s, 3.0))
        reasons.append(f"BW99 {_hz(ft.bw99)} vs {_hz(proto['bw'][0])}–{_hz(proto['bw'][1])}: {bw_s:.2f}")

    if proto.get("rate") and ft.symbol_rate and (ft.symbol_rate_snr or 0) > 4.0:
        rs = _range_score(ft.symbol_rate, proto["rate"], tol_octaves=1.2)
        terms.append((rs, 2.0))
        reasons.append(f"symbol rate {_hz(ft.symbol_rate)} vs "
                       f"{_hz(proto['rate'][0])}–{_hz(proto['rate'][1])}: {rs:.2f}")

    if proto.get("dur"):
        # Weight 1.5, above its original 1.0. Frame durations are tightly
        # specified (an ADS-B extended squitter is exactly 112 bits = 120 us)
        # whereas a run-length chip-rate estimate on a pulsed frame is noisy, so
        # duration deserves more say than rate when the two disagree. This is
        # what separates ADS-B from a Mode A/C/S reply: same band, same
        # modulation, same bandwidth, different frame length.
        ds = _range_score(ft.duration, proto["dur"], tol_octaves=2.5)
        terms.append((ds, 1.5))
        reasons.append(f"burst {ft.duration*1e3:.3f} ms: {ds:.2f}")

    if proto.get("scs") and ft.subcarrier_spacing:
        ss = _range_score(ft.subcarrier_spacing, proto["scs"], tol_octaves=0.6)
        terms.append((ss, 2.5))
        reasons.append(f"subcarrier spacing {_hz(ft.subcarrier_spacing)}: {ss:.2f}")

    if proto.get("dev") and ft.fsk_deviation and ft.modclass.startswith("fsk"):
        vs = _range_score(ft.fsk_deviation, proto["dev"], tol_octaves=1.0)
        terms.append((vs, 1.5))
        reasons.append(f"FSK deviation {_hz(ft.fsk_deviation)}: {vs:.2f}")

    # weighted geometric mean -> one bad feature is punished hard
    num = sum(w * math.log(max(s, 1e-3)) for s, w in terms)
    den = sum(w for _, w in terms) or 1.0
    score = math.exp(num / den) * mod_gate
    return dict(name=proto["name"], score=float(score),
                reasons=reasons, spec=proto["spec"])


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR = 0.35


@dataclass
class BurstResult:
    features: Features
    candidates: list[dict]
    preamble: Optional[object] = None      # a PreambleHit, if one matched

    @property
    def best(self) -> Optional[dict]:
        return self.candidates[0] if self.candidates else None

    @property
    def confident(self) -> bool:
        """True only when the leader clears the floor. A 20% leader means
        'nothing in the database fits', not 'probably this one'.

        A preamble correlation is confident by construction: the threshold was
        chosen for a specific false-alarm probability, so clearing it already
        *is* the confidence statement.
        """
        if self.preamble is not None:
            return True
        return bool(self.best and self.best["score"] >= CONFIDENCE_FLOOR)


def identify(
    x: np.ndarray,
    fs: float,
    rf_center_hz: Optional[float] = None,
    max_bursts: int = 12,
    top: int = 3,
    preamble_bank: Optional[object] = None,
    preamble_snr_tol_db: float = 12.0,
    **burst_kw,
) -> list[BurstResult]:
    """Segment ``x`` into bursts and rank protocol candidates for each.

    ``preamble_bank`` is any object exposing ``scan(x, fs, rf_center_hz)`` and
    returning hits with ``.name``, ``.t0``, ``.rho``, ``.threshold``,
    ``.n_template``, ``.spec`` and ``.est_snr_db`` — see ``iq_preamble``. It is
    duck-typed rather than imported so the blind classifier stays standalone.

    A preamble hit overrides the blind ranking rather than merging with it. The
    two are not comparable evidence: the blind score is a similarity heuristic,
    while a correlation above threshold is a statement about false-alarm
    probability. Where the correlator fires, its answer wins; the blind features
    are kept alongside as corroboration and as the source of the measurements
    (bandwidth, symbol rate, deviation) that the correlator does not produce.
    """
    x = np.asarray(x, dtype=np.complex64)
    bursts, noise_db = detect_bursts(x, fs, **burst_kw)

    # analyse the strongest / longest bursts first
    bursts = sorted(bursts, key=lambda b: -(b.i1 - b.i0))[:max_bursts]
    bursts = sorted(bursts, key=lambda b: b.i0)

    out = []
    for b in bursts:
        try:
            ft = extract_features(x, fs, b, noise_db)
        except ValueError:
            continue
        cands = [score_protocol(ft, p, rf_center_hz) for p in PROTOCOLS]
        cands = [c for c in cands if c["score"] > 0.02]
        cands.sort(key=lambda c: -c["score"])
        out.append(BurstResult(features=ft, candidates=cands[:top]))

    if preamble_bank is None:
        return out

    for hit in preamble_bank.scan(x, fs, rf_center_hz=rf_center_hz):
        target = None
        for r in out:
            t0, t1 = r.features.t0, r.features.t0 + r.features.duration
            if t0 - 200e-6 <= hit.t0 <= t1 + 200e-6:
                target = r
                break

        # Consistency gate. Equation (2) says a genuinely matched template must
        # return rho ~= sqrt(gamma/(1+gamma)), so its implied SNR should track
        # the SNR we measured on the burst. A template that correlates with a
        # frame's random payload rather than its preamble clears the noise-only
        # threshold but implies an SNR far below what the burst actually has —
        # which is how a BLE 2M template finds a 'preamble' inside a 30 dB BLE
        # 1M payload. One-sided on purpose: a hit is only ever rejected for
        # implying *less* signal than is measurably present.
        # Only applied to bursts whose SNR is high enough for the measurement
        # itself to mean anything. Near the detector's own floor the energy
        # detector reports whatever a noise fluctuation happened to do, and
        # letting that veto a correlation that cleared its threshold throws away
        # the entire point of the matched filter — it cost ~12 dB of POCSAG
        # sensitivity before this guard was added.
        if (target is not None and np.isfinite(target.features.snr_db)
                and target.features.snr_db > 10.0):
            if hit.est_snr_db < target.features.snr_db - preamble_snr_tol_db:
                target.features.notes.append(
                    f"rejected {hit.name} preamble hit: implied SNR "
                    f"{hit.est_snr_db:+.1f} dB inconsistent with measured "
                    f"{target.features.snr_db:+.1f} dB")
                continue
        if target is None:
            # Below roughly 6 dB the energy detector finds nothing, so a hit
            # with no matching burst is the normal low-SNR case, not an error.
            target = BurstResult(
                features=Features(
                    t0=hit.t0, duration=hit.template_duration,
                    snr_db=hit.est_snr_db, bw99=0.0, bw_rms=0.0, cfo=hit.cfo_hz,
                    envelope_cv=float("nan"), papr_db=float("nan"),
                    flatness=float("nan"), edge_db=float("nan"),
                    modclass="(preamble only — no burst detected)",
                    notes=["no energy-detected burst; identified by correlation alone"],
                ),
                candidates=[],
            )
            out.append(target)
        # keep the strongest hit per burst; hits arrive sorted by margin, so a
        # later one only wins if it is actually better
        if target.preamble is not None and \
                getattr(hit, "margin", 0) <= getattr(target.preamble, "margin", 0):
            continue
        target.preamble = hit
        target.candidates = [dict(
            name=hit.name, score=1.0, spec=hit.spec,
            reasons=[f"preamble correlation rho={hit.rho:.3f} vs threshold "
                     f"{hit.threshold:.3f} over N={hit.n_template} samples",
                     f"processing gain {hit.processing_gain_db:.1f} dB, "
                     f"CFO {hit.cfo_hz/1e3:+.1f} kHz",
                     f"implied per-sample SNR {hit.est_snr_db:+.1f} dB"],
        )] + [c for c in target.candidates if c["name"] != hit.name][: max(0, top - 1)]

    out.sort(key=lambda r: r.features.t0)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _hz(v: Optional[float]) -> str:
    if v is None:
        return "—"
    a = abs(v)
    for div, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if a >= div:
            return f"{v/div:.4g} {unit}"
    return f"{v:.4g} Hz"


def format_report(results: Sequence[BurstResult], verbose: bool = False) -> str:
    lines = []
    if not results:
        return "No bursts detected."
    for i, r in enumerate(results):
        ft = r.features
        lines.append(f"\n─── burst {i}  t={ft.t0*1e3:.3f} ms  "
                     f"len={ft.duration*1e3:.3f} ms  SNR≈{ft.snr_db:.1f} dB ───")
        lines.append(f"  class          {ft.modclass}")
        lines.append(f"  BW99 / RMS     {_hz(ft.bw99)} / {_hz(ft.bw_rms)}"
                     f"   offset {_hz(ft.cfo)}")
        lines.append(f"  env CV / PAPR  {ft.envelope_cv:.3f} / {ft.papr_db:.1f} dB"
                     f"   flatness {ft.flatness:.2f}  edge {ft.edge_db:.1f} dB")
        if ft.symbol_rate:
            lines.append(f"  symbol rate    {_hz(ft.symbol_rate)} "
                         f"(line SNR {ft.symbol_rate_snr:.1f}x)")
        if ft.n_tones:
            lines.append(f"  FSK tones      {ft.n_tones}, deviation {_hz(ft.fsk_deviation)}")
        if ft.ofdm_fft:
            lines.append(f"  OFDM           FFT≈{ft.ofdm_fft} samp, CP≈{ft.ofdm_cp}, "
                         f"SCS≈{_hz(ft.subcarrier_spacing)} (z={ft.ofdm_score:.1f})")
        if ft.chirp_score and ft.chirp_score > 0.10:
            lines.append(f"  CSS            BW≈{_hz(ft.css_bw)}, SF≈{ft.css_sf}, "
                         f"{ft.chirp_rate/1e6:.4g} MHz/s, "
                         f"dechirp energy {ft.chirp_score:.2f}")
        if not r.candidates:
            lines.append("  no candidate matched (unknown / not in database)")
            continue
        if r.preamble is not None:
            lines.append(f"  PREAMBLE MATCH  {r.preamble}")
        for note in ft.notes:
            lines.append(f"  · {note}")
        lines.append("  candidates:" if r.confident else
                     "  NO CONFIDENT MATCH — closest entries, all weak:")
        for c in r.candidates:
            lines.append(f"    {c['score']*100:5.1f}%  {c['name']}")
            lines.append(f"            spec: {c['spec']}")
            if verbose:
                for why in c["reasons"]:
                    lines.append(f"            · {why}")
    return "\n".join(lines)


def to_json(results: Sequence[BurstResult]) -> str:
    payload = []
    for r in results:
        d = asdict(r.features)
        d["candidates"] = r.candidates
        payload.append(d)
    return json.dumps(payload, indent=2, default=float)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Identify a wireless protocol from baseband IQ.")
    ap.add_argument("path")
    ap.add_argument("--fs", type=float, required=True, help="sample rate in Hz")
    ap.add_argument("--format", default="cf32", choices=sorted(_DTYPES))
    ap.add_argument("--center", type=float, default=None,
                    help="RF tuner centre frequency in Hz (greatly improves accuracy)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--max-bursts", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="show per-feature scoring")
    a = ap.parse_args(argv)

    x = load_iq(a.path, a.format, a.max_samples)
    res = identify(x, a.fs, rf_center_hz=a.center, top=a.top, max_bursts=a.max_bursts)
    print(to_json(res) if a.json else format_report(res, a.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
