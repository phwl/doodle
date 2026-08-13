"""Synthetic-signal validation for iq_protocol_id.

Generates textbook waveforms for several protocol families and checks that
the classifier lands on the right modulation family and a sensible top
protocol candidate. Not a substitute for real captures, but it catches
regressions in the feature extractors.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig

from iq_protocol_id import identify, format_report

RNG = np.random.default_rng(0xC0FFEE)


def awgn(x, snr_db):
    p = np.mean(np.abs(x) ** 2)
    n = np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return x + n * (RNG.standard_normal(x.size) + 1j * RNG.standard_normal(x.size))


def pad(x, fs, lead=200e-6, tail=200e-6, snr_db=25):
    """Frame a burst in noise so the burst detector has something to find."""
    nl, nt = int(lead * fs), int(tail * fs)
    p = np.mean(np.abs(x) ** 2)
    nfloor = np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    noise = lambda n: nfloor * (RNG.standard_normal(n) + 1j * RNG.standard_normal(n))
    # cosine ramps to avoid spectral splatter at the edges
    r = int(min(64, x.size // 20)) or 1
    w = np.ones(x.size)
    w[:r] = np.sin(np.linspace(0, np.pi / 2, r)) ** 2
    w[-r:] = np.cos(np.linspace(0, np.pi / 2, r)) ** 2
    return np.concatenate([noise(nl), awgn(x * w, snr_db), noise(nt)]).astype(np.complex64)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

def gen_gfsk(fs, rate, dev, nbits, bt=0.5, m=2):
    sps = int(round(fs / rate))
    syms = RNG.integers(0, m, nbits)
    levels = (2 * syms - (m - 1)) / (m - 1)          # -1..+1
    up = np.repeat(levels, sps)
    # Gaussian pulse shaping
    span = 4 * sps
    t = (np.arange(-span, span + 1)) / sps
    a = np.sqrt(np.log(2) / 2) / bt
    g = np.exp(-(np.pi**2) * (t**2) / (2 * a**2))
    g /= g.sum()
    up = np.convolve(up, g, mode="same")
    ph = 2 * np.pi * dev * np.cumsum(up) / fs
    return np.exp(1j * ph).astype(np.complex64)


def gen_ofdm(fs, n_fft, n_cp, n_active, n_sym):
    out = []
    idx = np.r_[np.arange(1, n_active // 2 + 1), np.arange(n_fft - n_active // 2, n_fft)]
    for _ in range(n_sym):
        X = np.zeros(n_fft, complex)
        X[idx] = (RNG.choice([-1, 1], idx.size) + 1j * RNG.choice([-1, 1], idx.size)) / np.sqrt(2)
        s = np.fft.ifft(X) * np.sqrt(n_fft)
        out.append(np.r_[s[-n_cp:], s])
    return np.concatenate(out).astype(np.complex64)


def gen_lora(fs, bw, sf, n_sym):
    n = int(round(2**sf * fs / bw))
    k = bw / (2**sf / bw)                            # Hz/s
    t = np.arange(n) / fs
    out = []
    for _ in range(n_sym):
        off = RNG.integers(0, 2**sf) / 2**sf
        f = (-bw / 2 + bw * ((t / (n / fs) + off) % 1.0))
        out.append(np.exp(1j * 2 * np.pi * np.cumsum(f) / fs))
    return np.concatenate(out).astype(np.complex64)


def gen_ppm_ook(fs, chip_rate, nchips):
    """ADS-B-style pulse-position modulation: 0.5 us pulse in each 1 us chip."""
    sps = int(round(fs / chip_rate))
    half = sps // 2
    out = np.zeros(nchips * sps)
    for i in range(nchips):
        o = 0 if RNG.integers(0, 2) else half
        out[i * sps + o : i * sps + o + half] = 1.0
    out = sig.lfilter(np.ones(max(2, sps // 8)) / max(2, sps // 8), 1, out)
    return out.astype(np.complex64)


def gen_linear(fs, rate, nsym, order=4, alpha=0.35):
    sps = int(round(fs / rate))
    ang = 2 * np.pi * RNG.integers(0, order, nsym) / order
    syms = np.exp(1j * ang)
    up = np.zeros(nsym * sps, complex)
    up[::sps] = syms
    span = 8
    t = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sinc(t) * np.cos(np.pi * alpha * t) / (1 - (2 * alpha * t) ** 2)
    h[~np.isfinite(h)] = 0
    h[np.abs(1 - (2 * alpha * t) ** 2) < 1e-9] = alpha / 2 * np.sin(np.pi / (2 * alpha))
    h /= np.sqrt(np.sum(h**2))
    return np.convolve(up, h, mode="same").astype(np.complex64)


# ---------------------------------------------------------------------------
# cases: (label, fs, rf_center, iq, expected modclass, substring of expected name)
# ---------------------------------------------------------------------------

def cases():
    yield ("SiK/MAVLink 250 kbps GFSK, dev 60 kHz", 2e6, 915e6,
           pad(gen_gfsk(2e6, 250e3, 60e3, 700), 2e6, 1e-3, 1e-3),
           "fsk2", "SiK")

    yield ("BLE 1M GFSK, dev 250 kHz", 8e6, 2.44e9,
           pad(gen_gfsk(8e6, 1e6, 250e3, 300), 8e6),
           "fsk2", "Bluetooth")

    yield ("Wi-Fi 20 MHz OFDM (N=64, CP=16, 52 active)", 20e6, 2.437e9,
           pad(gen_ofdm(20e6, 64, 16, 52, 400), 20e6, 40e-6, 40e-6),
           "ofdm", "Wi-Fi")

    yield ("LTE-like OFDM, 15 kHz SCS, 10 MHz", 15.36e6, 1.85e9,
           pad(gen_ofdm(15.36e6, 1024, 72, 600, 60), 15.36e6, 200e-6, 200e-6),
           "ofdm", "LTE")

    yield ("LoRa BW=125 kHz SF7", 500e3, 868e6,
           pad(gen_lora(500e3, 125e3, 7, 40), 500e3, 4e-3, 4e-3),
           "css", "LoRa")

    yield ("ADS-B 1090ES PPM (112-bit frame)", 8e6, 1090e6,
           pad(gen_ppm_ook(8e6, 1e6, 120), 8e6, 60e-6, 60e-6),
           "ook-pulsed", "ADS-B")

    yield ("DMR 4-FSK 4.8 ksym, dev 1.944 kHz", 48e3, 446e6,
           pad(gen_gfsk(48e3, 4800, 1944, 900, bt=0.6, m=4), 48e3, 20e-3, 20e-3),
           "fsk4", "DMR")

    yield ("QPSK 5 Msym RRC a=0.35", 20e6, 2.4e9,
           pad(gen_linear(20e6, 5e6, 2000), 20e6, 40e-6, 40e-6),
           "linear-shaped", None)


def main():
    passed = failed = 0
    for label, fs, fc, iq, want_mod, want_name in cases():
        res = identify(iq, fs, rf_center_hz=fc, top=3)
        print("=" * 78)
        print(f"{label}\n  fs={fs/1e6:g} MHz  fc={fc/1e6:g} MHz  "
              f"{len(iq)} samples  ->  {len(res)} burst(s)")
        if not res:
            print("  FAIL: no burst detected")
            failed += 1
            continue
        # take the longest burst as the verdict
        r = max(res, key=lambda q: q.features.duration)
        ft, best = r.features, r.best
        got_mod = ft.modclass
        got_name = best["name"] if best else "(no candidate)"
        ok_mod = got_mod == want_mod
        ok_name = want_name is None or (best and want_name.lower() in got_name.lower())
        print(f"  modclass  : {got_mod:16s} expected {want_mod:16s} {'OK' if ok_mod else 'MISMATCH'}")
        sc = best["score"] * 100 if best else 0.0
        print(f"  top match : {got_name}  ({sc:.1f}%)   {'OK' if ok_name else 'MISMATCH'}")
        print(f"  BW99={ft.bw99/1e3:.1f} kHz  Rs="
              f"{(ft.symbol_rate or 0)/1e3:.1f} kHz (snr {ft.symbol_rate_snr or 0:.1f})"
              f"  cv={ft.envelope_cv:.3f}  tones={ft.n_tones}"
              f"  fft={ft.ofdm_fft} cp={ft.ofdm_cp} scs="
              f"{(ft.subcarrier_spacing or 0)/1e3:.1f} kHz"
              f"  chirp={ft.chirp_score} sf={ft.css_sf}")
        print("  runners-up:", ", ".join(f"{c['name']} {c['score']*100:.0f}%"
                                        for c in r.candidates[1:]))
        if ok_mod and ok_name:
            passed += 1
        else:
            failed += 1
    print("=" * 78)
    print(f"passed {passed} / {passed + failed}")
    return 0 if failed == 0 else 1




def extra_checks():
    """Multi-burst segmentation and an SNR sweep."""
    print("\n" + "=" * 78)
    print("MULTI-BURST CAPTURE (3 protocols interleaved at fs=8 MHz)")
    fs = 8e6
    # one global noise floor; splicing separately-padded chunks would create
    # zero-power gaps and give the noise-floor estimator a false reference
    def gap(us):
        return np.zeros(int(us * 1e-6 * fs), np.complex64)

    cores = [
        (gap(300), None),
        (gen_gfsk(fs, 1e6, 250e3, 300), "BLE-ish"),
        (gap(400), None),
        (gen_ppm_ook(fs, 1e6, 120), "ADS-B-ish"),
        (gap(400), None),
        (gen_gfsk(fs, 250e3, 60e3, 400), "SiK-ish"),
        (gap(300), None),
    ]
    body = np.concatenate([c for c, _ in cores])
    ref = np.mean(np.abs(np.concatenate([c for c, n in cores if n])) ** 2)
    nf = np.sqrt(ref / (2 * 10 ** (25 / 10)))
    parts = [body + nf * (RNG.standard_normal(body.size)
                          + 1j * RNG.standard_normal(body.size))]
    iq = np.concatenate(parts).astype(np.complex64)
    res = identify(iq, fs, rf_center_hz=None, top=1)
    print(f"  {len(iq)} samples -> {len(res)} bursts")
    for i, r in enumerate(res):
        nm = r.best["name"] if r.best else "(none)"
        print(f"   {i}: t={r.features.t0*1e6:7.1f} us  {r.features.duration*1e6:7.1f} us  "
              f"{r.features.modclass:15s} {nm}"
              f"{'' if r.confident else '   [low confidence]'}")

    print("\n" + "=" * 78)
    print("SNR SWEEP — SiK-class 250 kbps GFSK at fs=2 MHz, fc=915 MHz")
    for snr in (30, 20, 15, 10, 6, 3, 0):
        iq = pad(gen_gfsk(2e6, 250e3, 60e3, 700), 2e6, 1e-3, 1e-3, snr_db=snr)
        res = identify(iq, 2e6, rf_center_hz=915e6, top=1)
        if not res:
            print(f"   {snr:3d} dB: no burst detected")
            continue
        r = max(res, key=lambda q: q.features.duration)
        nm = r.best["name"] if r.best else "(none)"
        print(f"   {snr:3d} dB: {r.features.modclass:12s} Rs="
              f"{(r.features.symbol_rate or 0)/1e3:7.1f} kHz  tones={r.features.n_tones}"
              f"  -> {nm[:40]}{'' if r.confident else ' [low conf]'}")


if __name__ == "__main__":
    rc = main()
    extra_checks()
    raise SystemExit(rc)
