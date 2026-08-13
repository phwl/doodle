"""Validation for iq_preamble: correctness, false-alarm rate, and SNR floor.

The headline number is in `snr_floor_sweep`: the lowest SNR at which each
detector identifies the protocol on 10 out of 10 independent trials.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import fft as sp_fft

from iq_protocol_id import identify
from iq_preamble import (
    BLE_ADV_1M, ADSB_PREAMBLE, PreambleBank, Template, _bits, _cfo_grid,
    default_templates, lora_preamble, normalized_correlate, pocsag_preamble,
    sik_preamble, snr_floor_db, synth_css_upchirps, synth_gfsk,
    threshold_for_pfa,
)

RNG = np.random.default_rng(0xBEEF)


# ---------------------------------------------------------------------------
# frame builders: a real preamble followed by real payload
# ---------------------------------------------------------------------------

def awgn_at(x: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise scaled to the *signal* power, so SNR is per-sample in-band."""
    p = float(np.mean(np.abs(x) ** 2))
    n = math.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return (x + n * (RNG.standard_normal(x.size)
                     + 1j * RNG.standard_normal(x.size))).astype(np.complex64)


def frame_gfsk(tpl: Template, fs: float, payload_bits: int,
               snr_db: float, lead_s: float, cfo_hz: float = 0.0) -> np.ndarray:
    """Preamble + random payload, embedded in a longer noise-only record."""
    bits = np.concatenate([_bits(tpl.bits),
                           RNG.integers(0, 2, payload_bits).astype(np.int8)])
    core = synth_gfsk(bits, fs, tpl.symbol_rate,
                      tpl.params["deviation"], tpl.params.get("bt", 0.5))
    if cfo_hz:
        core = core * np.exp(2j * np.pi * cfo_hz * np.arange(core.size) / fs)

    lead = int(round(lead_s * fs))
    tail = lead
    p = float(np.mean(np.abs(core) ** 2))
    nf = math.sqrt(p / (2 * 10 ** (snr_db / 10)))
    noise = lambda n: nf * (RNG.standard_normal(n) + 1j * RNG.standard_normal(n))
    return np.concatenate([noise(lead), awgn_at(core, snr_db),
                           noise(tail)]).astype(np.complex64)


def frame_lora(fs: float, bw: float, sf: int, n_data: int,
               snr_db: float, lead_s: float) -> np.ndarray:
    pre = synth_css_upchirps(fs, bw, sf, 8)
    n = int(round(2 ** sf * fs / bw))
    t = np.arange(n) / fs
    k = bw * bw / (2 ** sf)
    data = []
    for _ in range(n_data):
        off = RNG.integers(0, 2 ** sf) / 2 ** sf
        f = -bw / 2 + bw * ((t / (n / fs) + off) % 1.0)
        data.append(np.exp(1j * 2 * np.pi * np.cumsum(f) / fs))
    core = np.concatenate([pre] + data).astype(np.complex64)
    lead = int(round(lead_s * fs))
    p = float(np.mean(np.abs(core) ** 2))
    nf = math.sqrt(p / (2 * 10 ** (snr_db / 10)))
    noise = lambda m: nf * (RNG.standard_normal(m) + 1j * RNG.standard_normal(m))
    return np.concatenate([noise(lead), awgn_at(core, snr_db),
                           noise(lead)]).astype(np.complex64)


# ---------------------------------------------------------------------------
# 1. correctness of the FFT + spectral-shift correlator
# ---------------------------------------------------------------------------

def test_correlator_matches_brute_force():
    print("=" * 78)
    print("1. FFT/spectral-shift correlator vs direct evaluation")
    fs = 8e6
    r = BLE_ADV_1M.synth(fs)
    grid = _cfo_grid(BLE_ADV_1M.cfo_span, BLE_ADV_1M.duration())
    x = (RNG.standard_normal(1500) + 1j * RNG.standard_normal(1500)) * 0.3
    x = x.astype(np.complex64)
    x[400:400 + r.size] += r * np.exp(2j * np.pi * 37e3 * np.arange(r.size) / fs)

    nfft = sp_fft.next_fast_len(x.size + r.size)
    gq = sorted({round(f / (fs / nfft)) * (fs / nfft) for f in grid})

    er = np.sum(np.abs(r) ** 2)
    ref = np.zeros(x.size - r.size + 1)
    for i in range(ref.size):
        seg = x[i:i + r.size]
        ex = np.sum(np.abs(seg) ** 2)
        best = 0.0
        for f in gq:
            rr = r * np.exp(2j * np.pi * f * np.arange(r.size) / fs)
            best = max(best, abs(np.vdot(rr, seg)) / math.sqrt(ex * er))
        ref[i] = best

    rho, cfo = normalized_correlate(x, r, grid, fs)
    err = float(np.max(np.abs(rho - ref)))
    k = int(np.argmax(rho))
    print(f"   N={r.size}, {grid.size} frequency bins spaced "
          f"{grid[1]-grid[0]:.0f} Hz")
    print(f"   max |rho_fft - rho_direct| = {err:.2e}")
    print(f"   peak at sample {k} (true 400), CFO {cfo[k]/1e3:+.1f} kHz (true +37.0)")
    ok = err < 1e-6 and k == 400 and abs(cfo[k] - 37e3) < 25e3
    print(f"   -> {'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# 2. equation (2): E[rho] = sqrt(gamma/(1+gamma)), independent of N
# ---------------------------------------------------------------------------

def test_rho_vs_snr():
    print("=" * 78)
    print("2. Peak correlation vs SNR — equation (2), and its independence of N")
    fs = 8e6
    templates = {"BLE 1M (N=320)": BLE_ADV_1M.synth(fs),
                 "POCSAG (N=%d)" % pocsag_preamble().synth(64e3).size:
                     pocsag_preamble().synth(64e3)}
    print(f"   {'SNR':>6}  {'theory':>8}   " +
          "   ".join(f"{k:>16}" for k in templates))
    ok = True
    for snr in (20, 10, 0, -6, -10, -15):
        g = 10 ** (snr / 10)
        theory = math.sqrt(g / (1 + g))
        row = []
        for r in templates.values():
            acc = []
            for _ in range(16):
                nf = math.sqrt(1 / (2 * g))
                y = r + nf * (RNG.standard_normal(r.size)
                              + 1j * RNG.standard_normal(r.size))
                acc.append(abs(np.vdot(r, y))
                           / (np.linalg.norm(r) * np.linalg.norm(y)))
            m = float(np.mean(acc))
            row.append(m)
            if abs(m - theory) > 0.05:
                ok = False
        print(f"   {snr:4d}dB  {theory:8.4f}   " +
              "   ".join(f"{v:16.4f}" for v in row))
    print(f"   -> {'OK' if ok else 'FAIL'} (both template lengths track the same "
          f"curve; N affects the threshold, not the peak)")
    return ok


# ---------------------------------------------------------------------------
# 3. false alarms on noise only
# ---------------------------------------------------------------------------

def test_false_alarms():
    print("=" * 78)
    print("3. False-alarm rate on noise-only captures")
    bank = PreambleBank(p_fa=1e-6)
    total = 0
    trials = 12
    for band, fs, n in ((2.44e9, 8e6, 200_000),
                        (915e6, 2e6, 200_000),
                        (1090e6, 8e6, 200_000)):
        fired = 0
        for _ in range(trials):
            x = (RNG.standard_normal(n) + 1j * RNG.standard_normal(n))
            hits = bank.scan(x.astype(np.complex64), fs, rf_center_hz=band)
            fired += len(hits)
        total += fired
        print(f"   fc={band/1e6:7.1f} MHz  fs={fs/1e6:g} MHz  "
              f"{trials} x {n} samples -> {fired} hits")
    print(f"   -> {'OK' if total == 0 else 'FAIL'} ({total} total false alarms)")
    return total == 0


# ---------------------------------------------------------------------------
# 4. the headline: SNR floor, blind vs preamble
# ---------------------------------------------------------------------------

def _blind_ok(iq, fs, fc, want) -> bool:
    res = identify(iq, fs, rf_center_hz=fc, top=1)
    if not res:
        return False
    r = max(res, key=lambda q: q.features.duration)
    return bool(r.confident and r.best and want.lower() in r.best["name"].lower())


def _preamble_ok(iq, fs, fc, want, bank) -> bool:
    res = identify(iq, fs, rf_center_hz=fc, top=1, preamble_bank=bank)
    return any(r.preamble is not None and want.lower() in r.preamble.name.lower()
               for r in res)


def snr_floor_sweep():
    print("=" * 78)
    print("4. SNR floor — lowest SNR passing 10/10 trials")
    bank = PreambleBank(p_fa=1e-6)
    grid = [30, 25, 20, 15, 10, 6, 3, 0, -3, -6, -9, -12, -15, -18, -21, -24]

    cases = []

    sik = sik_preamble(250e3, 60e3)
    cases.append(("SiK/MAVLink 250 kbps GFSK", 2e6, 915e6, "SiK",
                  lambda s: frame_gfsk(sik, 2e6, 600, s, 1e-3), sik))

    cases.append(("BLE 1M advertising", 8e6, 2.44e9, "Bluetooth LE 1M",
                  lambda s: frame_gfsk(BLE_ADV_1M, 8e6, 300, s, 200e-6),
                  BLE_ADV_1M))

    lora = lora_preamble(125e3, 7)
    cases.append(("LoRa BW125 SF7", 500e3, 868e6, "LoRa",
                  lambda s: frame_lora(500e3, 125e3, 7, 20, s, 4e-3), lora))

    poc = pocsag_preamble(1200.0, 4.5e3)
    cases.append(("POCSAG 1200 bps", 64e3, 148e6, "POCSAG",
                  lambda s: frame_gfsk(poc, 64e3, 400, s, 30e-3), poc))

    rows = []
    for label, fs, fc, want, make, tpl in cases:
        floors = {}
        for mode in ("blind", "preamble"):
            floor = None
            for snr in grid:
                good = 0
                for _ in range(10):
                    iq = make(snr)
                    ok = (_blind_ok(iq, fs, fc, want) if mode == "blind"
                          else _preamble_ok(iq, fs, fc, want, bank))
                    good += bool(ok)
                if good == 10:
                    floor = snr
                else:
                    break
            floors[mode] = floor
        _, fsd, _d = bank._decimate(np.zeros(4096, np.complex64), fs, tpl)
        n = tpl.synth(fsd).size
        pred = snr_floor_db(n, 10_000)
        rows.append((label, floors["blind"], floors["preamble"], n, pred))
        b = f"{floors['blind']:+d} dB" if floors["blind"] is not None else "  none"
        p = f"{floors['preamble']:+d} dB" if floors["preamble"] is not None else "  none"
        gain = (f"{floors['blind'] - floors['preamble']:+d} dB"
                if None not in (floors["blind"], floors["preamble"]) else "   —")
        print(f"   {label:28s} blind {b:>8}   preamble {p:>8}   "
              f"gain {gain:>7}   (N={n}, theory {pred:+.1f} dB)")
    return rows


# ---------------------------------------------------------------------------
# 5. carrier offset robustness
# ---------------------------------------------------------------------------

def test_cfo_robustness():
    print("=" * 78)
    print("5. Carrier-offset robustness at 0 dB SNR (SiK, 250 kbps)")
    bank = PreambleBank(p_fa=1e-6)
    tpl = sik_preamble(250e3, 60e3)
    ok = True
    for ppm in (0, 2, 5, 10, 20, 40):
        cfo = ppm * 915e6 / 1e6
        good = 0
        for _ in range(6):
            iq = frame_gfsk(tpl, 2e6, 600, 0.0, 1e-3, cfo_hz=cfo)
            good += bool(_preamble_ok(iq, 2e6, 915e6, "SiK", bank))
        span = tpl.cfo_span
        inside = "in grid" if abs(cfo) <= span else "OUTSIDE grid"
        print(f"   {ppm:3d} ppm = {cfo/1e3:+7.1f} kHz  ({inside:12s})  {good}/6 detected")
        if abs(cfo) <= span and good < 6:
            ok = False
    print(f"   -> {'OK' if ok else 'FAIL'} (grid spans +/-{tpl.cfo_span/1e3:.0f} kHz)")
    return ok


# ---------------------------------------------------------------------------
# 6. end-to-end report at an SNR where blind analysis has no chance
# ---------------------------------------------------------------------------

def test_end_to_end():
    print("=" * 78)
    print("6. End-to-end identify() at -9 dB SNR")
    from iq_protocol_id import format_report
    bank = PreambleBank(p_fa=1e-6)
    tpl = sik_preamble(250e3, 60e3)
    iq = frame_gfsk(tpl, 2e6, 600, -9.0, 1e-3, cfo_hz=4.2e3)
    print("   --- without the bank ---")
    print(format_report(identify(iq, 2e6, rf_center_hz=915e6, top=2)))
    print("   --- with the bank ---")
    print(format_report(identify(iq, 2e6, rf_center_hz=915e6, top=2,
                                 preamble_bank=bank)))
    return True


def main():
    results = [
        test_correlator_matches_brute_force(),
        test_rho_vs_snr(),
        test_false_alarms(),
        test_cfo_robustness(),
    ]
    snr_floor_sweep()
    test_end_to_end()
    print("=" * 78)
    print(f"unit checks passed {sum(bool(r) for r in results)} / {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
