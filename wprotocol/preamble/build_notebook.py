"""Assemble the tutorial notebook.

Every code cell is lifted verbatim out of the tested modules by AST lookup, so
the notebook and the modules cannot drift apart.
"""
from __future__ import annotations

import ast
import json
import pathlib

ROOT = pathlib.Path("/home/claude")


def load(path: str) -> dict[str, str]:
    src = (ROOT / path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    out: dict[str, str] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Tuple):      # MHz, kHz = 1e6, 1e3
                    names += [e.id for e in t.elts if isinstance(e, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if not names:
            continue
        start = min([d.lineno for d in getattr(node, "decorator_list", [])]
                    + [node.lineno])
        seg = "\n".join(lines[start - 1: node.end_lineno])
        for n in names:
            out[n] = seg
    return out


PID = load("iq_protocol_id.py")
PRE = load("iq_preamble.py")
TPI = load("test_protocol_id.py")
TPR = load("test_preamble.py")


def S(table: dict[str, str], *names: str) -> str:
    missing = [n for n in names if n not in table]
    if missing:
        raise KeyError(f"not found in module: {missing}")
    return "\n\n\n".join(table[n] for n in names)


CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ===========================================================================
md(r"""
# Identifying Wireless Protocols from Baseband I/Q

### Blind feature classification, and why it needs a matched filter underneath

This notebook builds a working protocol identifier from raw complex baseband
samples, derives the mathematics behind every measurement it makes, and then
measures how well it actually performs.

It is organised around one finding. A classifier built purely from **blind
features** — bandwidths, envelope statistics, histograms of the frequency
discriminator — works well but stops working below roughly **20 dB SNR**. That
floor is not a coding deficiency. Every blind feature is a *second-order*
statistic, and second-order statistics of a signal in noise converge at a rate
no amount of cleverness improves. Adding a **preamble matched filter** — a
*first-order* statistic — moves the floor to between **&minus;6 and &minus;18 dB**
depending on preamble length, a gain of 31 to 38 dB.

## Contents

**Part I — Foundations** &nbsp;(§1–4)
signal model, synthetic ground truth, file I/O

**Part II — Blind feature extraction** &nbsp;(§5–14)
burst segmentation, spectral estimation, amplitude statistics, the FM
discriminator and its noise threshold, OFDM cyclic prefixes, chirp spread
spectrum, cyclostationary symbol-rate estimation, eye-diagram tone counting

**Part III — Classification** &nbsp;(§15–18)
feature assembly, modulation-family decision, protocol database, scoring

**Part IV — Validating the blind path** &nbsp;(§19–20)
eight synthetic protocols, multi-burst segmentation, and the SNR floor

**Part V — The preamble matched filter** &nbsp;(§21–26)
why the floor exists, detection theory, template bank, carrier-offset search,
template cross-talk, and the measured floors

**Part VI — Limitations** &nbsp;(§27)

## A note on provenance

The code cells are extracted programmatically from two tested modules,
`iq_protocol_id.py` and `iq_preamble.py`, so what you read here is what was
measured. Numerical claims in the prose are recomputed by the cells around
them — several of the constants below were originally *wrong* and the
verification cells are what caught them.

Where I cite a standard, I am working from memory of the specification rather
than from the document itself, and I have flagged the places where that
matters. In particular §22 discusses one template constant I deliberately
excluded because I do not trust my recall of it.
""")

md(r"""
## 1. Signal model and notation

The input is a stream of complex samples from a quadrature receiver. A single
real passband signal at carrier $f_c$ has been mixed down and low-pass filtered
into its in-phase and quadrature components:

$$x[n] \;=\; I[n] + jQ[n] \;=\; A[n]\,e^{j\phi[n]} \;+\; w[n], \qquad t = n/f_s$$

where $f_s$ is the sample rate. The noise $w[n]$ is modelled as circularly
symmetric complex white Gaussian: real and imaginary parts independent, each
of variance $\sigma^2/2$, so $\mathbb{E}|w|^2 = \sigma^2$. Per-sample SNR is

$$\gamma \;=\; \frac{\mathbb{E}|s[n]|^2}{\mathbb{E}|w[n]|^2}.$$

Because mixing to baseband is not perfect, a residual **carrier frequency
offset** $\Delta f$ remains, appearing as a slow rotation $e^{j2\pi\Delta f t}$
across the whole record. This turns out to be the dominant practical nuisance
in Part V.

The quantities we want to recover, and the section that derives each:

| Quantity | Symbol | Section |
|---|---|---|
| Occupied bandwidth | $B$ | §6 |
| Symbol (or chip) rate | $R_s = 1/T$ | §11 |
| Modulation family | — | §16 |
| Frequency deviation, tone count | $\Delta f_{\text{dev}}$, $M$ | §12 |
| OFDM FFT size, cyclic prefix | $N$, $L$ | §9 |
| Chirp bandwidth, spreading factor | $B$, $\mathrm{SF}$ | §10 |

Two conventions used throughout. **Modulation index** for FSK is
$h = 2\Delta f_{\text{dev}}/R_s$; BLE uses $h=0.5$. **Carson's rule** gives a
rough occupied bandwidth for angle modulation,
$B \approx 2(\Delta f_{\text{dev}} + R_s/2)$, and is used only to size filters,
never as a measurement.
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Setup

`QUICK` reduces trial counts in the statistical sections. The full sweeps in
Part V take roughly half an hour; the quick versions take a couple of minutes
and show the same structure with noisier floor estimates.
""")

code(r"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft as sp_fft
from scipy import signal as sps_signal
from scipy import signal as sig          # alias used by the generator module

QUICK = True          # set False to reproduce the published floor numbers

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({
    "figure.figsize": (11, 3.6),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

RNG = np.random.default_rng(0xC0FFEE)
print(f"numpy {np.__version__}   QUICK={QUICK}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Synthetic waveforms: ground truth first

Building the generators before the analysers is deliberate. Every measurement
in Parts II–III can then be checked against a waveform whose parameters we
chose, which is the only way to tell a working estimator from one that merely
produces plausible-looking numbers.

Each generator produces the **core** waveform; `pad` frames it in noise with
cosine ramps at the edges, so the burst detector has something realistic to
find and the spectrum has no splatter from a hard on/off transition.

The GFSK generator is worth reading closely. It builds an NRZ symbol stream,
shapes it with a Gaussian pulse, then integrates to phase — frequency
modulation is *integration* of the shaped symbol stream, which is why
`np.cumsum` appears where a naive implementation would put a multiply.
""")

code(S(TPI, "awgn", "pad"))
code(S(TPI, "gen_gfsk", "gen_ofdm", "gen_lora", "gen_ppm_ook", "gen_linear"))

md(r"""
A quick look at what these produce. Note how visually distinct the four
modulation families already are in the time-frequency plane — that separability
is what the blind features are trying to capture numerically.
""")

code(r"""
fig, ax = plt.subplots(1, 4, figsize=(13, 2.9))

demos = [
    ("GFSK 250 kbps\n(SiK class)", gen_gfsk(2e6, 250e3, 60e3, 120), 2e6),
    ("OFDM N=64 CP=16\n(Wi-Fi class)", gen_ofdm(20e6, 64, 16, 52, 40), 20e6),
    ("LoRa BW125 SF7\n(CSS)", gen_lora(500e3, 125e3, 7, 8), 500e3),
    ("PPM/OOK\n(ADS-B class)", gen_ppm_ook(8e6, 1e6, 40), 8e6),
]
for a, (title, x, fs) in zip(ax, demos):
    nper = max(32, min(256, len(x) // 24))
    a.specgram(x, NFFT=nper, Fs=fs / 1e6, noverlap=nper // 2, cmap="magma")
    a.set_title(title, fontsize=8)
    a.set_xlabel("time (us)" if fs > 1e6 else "time (s)")
    a.grid(False)
ax[0].set_ylabel("MHz")
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Reading captures from disk

Nothing subtle, but two details cause most real-world grief. **Interleaving:**
integer formats store $I_0 Q_0 I_1 Q_1 \dots$, so an odd sample count means a
truncated file. **RTL-SDR's `cu8`** is unsigned with a 127.5 offset; forgetting
to remove it puts a large DC spike at bin zero that corrupts every bandwidth
measurement downstream.

Normalising to unit RMS makes every threshold in the rest of the notebook
scale-free, which matters because SDR sample scaling is arbitrary and varies
with gain settings.
""")

code(S(PID, "_DTYPES", "load_iq"))

# ---------------------------------------------------------------------------
md(r"""
# Part II — Blind feature extraction

## 5. Burst segmentation

Most captures are mostly silence. Before measuring anything we split the record
into bursts, because averaging a 200 µs frame together with 5 ms of noise
destroys every statistic we care about.

### Energy detection

Smooth the instantaneous power over a short window and compare against the
noise floor:

$$P[n] = \frac{1}{W}\sum_{k=0}^{W-1}\bigl|x[n-k]\bigr|^2, \qquad
\text{declare signal where } 10\log_{10} P[n] > \eta_{\text{noise}} + \tau.$$

We use **hysteresis** — a high threshold (8 dB) to open a burst and a lower one
(5 dB) to close it. A single threshold chatters on and off through the burst's
own amplitude fluctuations, fragmenting one frame into dozens of pieces.

### Estimating the noise floor is the hard part

The obvious choice, a low percentile of $P$, fails whenever the duty cycle is
high: in a capture that is 85% signal, the 20th percentile lies *inside the
signal*, so the estimated floor is the signal level, the threshold is never
exceeded, and nothing is detected. This is not hypothetical — it was a real bug,
and it silently broke the LoRa and DMR test cases while the code appeared to
work.

The fix is the **minimum of per-block medians**. Split the record into ~200
blocks and take a low percentile of the block medians. This needs only *one*
quiet window anywhere in the record, regardless of overall duty cycle. It also
degrades gracefully: for a genuinely continuous carrier every block median sits
at the signal level, the peak-to-floor difference collapses, and that condition
is exactly what triggers the continuous-signal fallback.
""")

code(S(PID, "_smooth", "_noise_floor_db"))
code(S(PID, "Burst", "detect_bursts"))

md(r"""
Demonstrating both the failure and the fix on a high-duty-cycle record.
""")

code(r"""
fs = 500e3
core = gen_lora(fs, 125e3, 7, 40)
iq = pad(core, fs, 4e-3, 4e-3)          # ~84% duty cycle
p_db = 10 * np.log10(_smooth(np.abs(iq) ** 2, 4) + 1e-20)

naive = np.percentile(p_db, 20.0)
robust = _noise_floor_db(p_db)
true_floor = 10 * np.log10(np.mean(np.abs(iq[:1500]) ** 2))

t = np.arange(iq.size) / fs * 1e3
plt.plot(t, p_db, lw=0.4, color="0.6", label="smoothed power")
for v, lab, c in ((naive, "20th percentile", "tab:red"),
                  (robust, "min block median", "tab:green"),
                  (true_floor, "true noise floor", "k")):
    plt.axhline(v, color=c, ls="--", lw=1.2, label=f"{lab}: {v:.1f} dB")
plt.xlabel("time (ms)"); plt.ylabel("dB"); plt.legend(fontsize=7, ncol=2)
plt.title("Noise-floor estimation at 84% duty cycle")
plt.show()

print(f"20th percentile overestimates the floor by {naive - true_floor:5.1f} dB")
print(f"min block median is off by         {robust - true_floor:5.1f} dB")
b, nd = detect_bursts(iq, fs)
print(f"\nbursts found: {len(b)}, spanning "
      f"{[(round(x.t0*1e3,2), round(x.duration*1e3,2)) for x in b]} (ms)")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Spectral estimation and bandwidth

### Welch's method

A single periodogram of an $M$-sample record has variance equal to the square of
its own mean — it is a consistent estimator of nothing. Welch averages $K$
overlapping windowed periodograms, trading frequency resolution for variance
reduction by roughly $1/K$:

$$\hat{S}(f) = \frac{1}{K}\sum_{i=1}^{K}
\frac{1}{\|w\|^2}\Bigl|\sum_{n} x_i[n]\,w[n]\,e^{-j2\pi fn/f_s}\Bigr|^2$$

We keep the two-sided spectrum, because a baseband burst sitting off-centre in
the capture has a frequency offset we need to measure and remove.

### Occupied bandwidth

The standard regulatory definition: the band containing a fraction $\alpha$
(here 99%) of total power, split symmetrically about the tails. With the
cumulative distribution $C(f) = \int_{-f_s/2}^{f} \hat S / \int \hat S$,

$$B_{99} = C^{-1}\!\left(1 - \tfrac{1-\alpha}{2}\right) - C^{-1}\!\left(\tfrac{1-\alpha}{2}\right),
\qquad f_{\text{centre}} = \tfrac{1}{2}\bigl(\text{upper} + \text{lower}\bigr).$$

Interpolating $C^{-1}$ rather than snapping to bins matters: at coarse
resolution, bin snapping quantises the bandwidth badly enough to push a
narrowband signal into the wrong protocol's range.

### RMS bandwidth

The second-moment bandwidth $B_{\text{rms}} = 2\sqrt{\int (f-\mu)^2 \hat S / \int \hat S}$
weights energy far from the centre quadratically. That makes it a useful
companion to $B_{99}$ — but only if it is computed **inside the occupied band**.
Integrating over the whole capture folds the out-of-band noise floor into the
second moment; before this was restricted, a 320 kHz burst in a 20 MHz capture
reported an RMS bandwidth of 1.14 MHz, over three times too large.

### Shape features

**Spectral flatness** (Wiener entropy) is the ratio of geometric to arithmetic
mean of the PSD inside the band:

$$\mathcal{F} = \frac{\exp\bigl(\frac{1}{K}\sum_k \ln S_k\bigr)}{\frac{1}{K}\sum_k S_k} \in (0, 1]$$

It approaches 1 for OFDM, whose many independent subcarriers fill the band
uniformly, and drops for the rounded spectrum of a shaped single carrier.

**Edge sharpness** measures dB of roll-off just outside the band edge,
separating brick-wall filtered signals from slowly decaying ones.
""")

code(S(PID, "welch_psd", "occupied_bandwidth", "rms_bandwidth",
       "spectral_flatness", "edge_sharpness"))

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.1))
cases = [("GFSK 250 kbps", gen_gfsk(2e6, 250e3, 60e3, 400), 2e6),
         ("OFDM 52/64 carriers", gen_ofdm(20e6, 64, 16, 52, 200), 20e6),
         ("QPSK RRC a=0.35", gen_linear(20e6, 5e6, 800), 20e6)]

for a, (title, x, fs) in zip(ax, cases):
    f, pxx = welch_psd(x, fs)
    bw, c = occupied_bandwidth(f, pxx, 0.99)
    a.plot(f / 1e6, 10 * np.log10(pxx / pxx.max() + 1e-20), lw=0.7)
    a.axvspan((c - bw / 2) / 1e6, (c + bw / 2) / 1e6, alpha=0.15, color="tab:green")
    a.set_title(f"{title}\nB99={bw/1e3:.0f} kHz  "
                f"flatness={spectral_flatness(pxx, f, bw, c):.2f}  "
                f"edge={edge_sharpness(pxx, f, bw, c):.0f} dB", fontsize=8)
    a.set_xlabel("MHz"); a.set_ylim(-60, 3)
ax[0].set_ylabel("dB (normalised)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Amplitude statistics, and two numbers we can predict exactly

Two cheap scalars separate constant-envelope modulations (FSK, MSK, GMSK, CSS)
from those carrying information in amplitude (OFDM, QAM, OOK).

### Envelope coefficient of variation

$$\mathrm{CV} = \frac{\operatorname{std}|x|}{\operatorname{mean}|x|}$$

For an ideal constant-envelope signal, $\mathrm{CV} \to 0$. For OFDM we can
derive the value in closed form. With many independent subcarriers the central
limit theorem makes each time-domain sample circularly complex Gaussian, so
$|x|$ is **Rayleigh** with parameter $\sigma$:

$$\mathbb{E}|x| = \sigma\sqrt{\tfrac{\pi}{2}}, \qquad \mathbb{E}|x|^2 = 2\sigma^2
\;\Rightarrow\; \operatorname{Var}|x| = \sigma^2\!\left(2 - \tfrac{\pi}{2}\right)$$

$$\boxed{\;\mathrm{CV}_{\text{Rayleigh}} = \frac{\sigma\sqrt{2 - \pi/2}}{\sigma\sqrt{\pi/2}}
= \sqrt{\frac{4}{\pi} - 1} \approx 0.5227\;}$$

This is a genuinely sharp prediction with no free parameters, and it is the
single most reliable OFDM indicator in the whole feature set.

### Peak-to-average power ratio

For $M$ i.i.d. exponential power samples, the expected maximum grows
logarithmically:

$$\mathbb{E}\!\left[\frac{\max_m |x_m|^2}{\mathbb{E}|x|^2}\right] \approx \ln M + \gamma_E,
\qquad \gamma_E \approx 0.5772$$

so PAPR in dB is $\approx 10\log_{10}(\ln M + 0.577)$ — slowly growing, and
therefore weak evidence on its own. It is genuinely useful only for pulsed
signals, where a 50%-duty OOK waveform gives almost exactly 3 dB.

Let us check both predictions.
""")

code(r"""
def env_stats(x):
    a = np.abs(x).astype(float)
    cv = a.std() / a.mean()
    papr = 10 * np.log10(a.max() ** 2 / np.mean(a ** 2))
    return cv, papr

rows = [
    ("GFSK (constant envelope)", gen_gfsk(2e6, 250e3, 60e3, 800), 0.0),
    ("OFDM 52/64 carriers", gen_ofdm(20e6, 64, 16, 52, 400), math.sqrt(4/math.pi - 1)),
    ("QPSK RRC a=0.35", gen_linear(20e6, 5e6, 2000), None),
    ("PPM/OOK 50% duty", gen_ppm_ook(8e6, 1e6, 400), 1.0),
]
print(f"{'waveform':30s} {'CV':>8} {'predicted':>10} {'PAPR dB':>9} {'ln M+g':>8}")
for name, x, pred in rows:
    cv, papr = env_stats(x)
    p = f"{pred:.4f}" if pred is not None else "—"
    print(f"{name:30s} {cv:8.4f} {p:>10} {papr:9.2f} "
          f"{10*math.log10(math.log(len(x)) + 0.5772):8.2f}")

print(f"\nRayleigh CV prediction: sqrt(4/pi - 1) = {math.sqrt(4/math.pi - 1):.4f}")
print("The OFDM row matches to ~0.01 with no fitted parameters.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Channel filtering and decimation

A 250 kHz burst in a 20 MHz capture is 80 samples per symbol of mostly noise.
Filtering to the occupied band and decimating discards the out-of-band noise —
the largest cheap SNR win available, worth roughly $10\log_{10}(f_s/2B)$, about
9 dB in this example.

It also makes sample-count-based smoothing windows *meaningful*: a window of
"5 samples" means something entirely different at 80 samples/symbol than at 6.

The target rate is a compromise with a real cost on each side. Too little
decimation and the discriminator smoothing in §12 cannot be tuned; too much and
we starve the eye of samples per symbol. Targeting $\approx 6B$ works; targeting
$4B$ **broke the BLE test case**, because at 4 samples/symbol a GFSK eye stops
resolving reliably. That is why the guard is `decim >= 3` rather than `>= 2`.
""")

code(r"""
fs, bw = 20e6, 320e3
for target_mult in (4.0, 6.0, 10.0):
    d = int(max(1, np.floor(fs / (target_mult * bw))))
    print(f"target {target_mult:4.1f}xB -> decimate {d:3d}x  "
          f"-> fs={fs/d/1e6:6.3f} MHz, {fs/d/250e3:5.2f} samples/symbol at 250 kbps, "
          f"noise rejected ~{10*math.log10(max(d,1)):4.1f} dB")
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. The FM discriminator and its noise threshold

Almost every narrowband feature we extract comes from the **instantaneous
frequency**, obtained by differencing unwrapped phase:

$$\hat f[n] = \frac{f_s}{2\pi}\Bigl(\phi[n+1] - \phi[n]\Bigr)$$

This is the single noisiest operation in the pipeline, and understanding exactly
*how* noisy is what makes the rest of Part II work.

### Phase noise to frequency noise

Write the received sample as signal plus noise, $x = A e^{j\phi} + w$. For small
noise, the phase error is the noise component in quadrature with the signal,
divided by the amplitude:

$$\theta \approx \frac{\operatorname{Im}\{w e^{-j\phi}\}}{A}
\qquad\Longrightarrow\qquad
\sigma_\theta^2 = \frac{\sigma^2/2}{A^2} = \frac{1}{2\gamma}$$

Differencing two *independent* phase errors doubles the variance, so

$$\sigma_{\hat f}^2 = 2\sigma_\theta^2\left(\frac{f_s}{2\pi}\right)^2
= \frac{1}{\gamma}\left(\frac{f_s}{2\pi}\right)^2
\qquad\Longrightarrow\qquad
\boxed{\;\sigma_{\hat f} = \frac{f_s}{2\pi\sqrt{\gamma}}\;}$$

Put numbers in: at $f_s = 48$ kHz and 25 dB SNR, $\sigma_{\hat f} \approx 430$ Hz.
A DMR 4-FSK signal has adjacent tones **1296 Hz** apart. The noise is a third of
the tone spacing, which is why raw discriminator histograms of narrowband 4-FSK
collapse into a single blob.

### Smoothing telescopes

Averaging $W$ consecutive difference estimates does something better than
$\sqrt{W}$, because the sum telescopes:

$$\frac{1}{W}\sum_{k=0}^{W-1}\bigl(\phi[n{+}k{+}1]-\phi[n{+}k]\bigr)
= \frac{\phi[n{+}W]-\phi[n]}{W}$$

Only the two endpoint phase errors survive, so

$$\sigma_{\hat f, W} = \frac{f_s}{2\pi W \sqrt{\gamma}} \quad\text{— falls as } 1/W,\ \text{not } 1/\sqrt{W}.$$

### The constraint that closes the argument

Smoothing is only valid while the frequency is constant, so $W$ cannot exceed
the symbol period, $W < f_s/R_s$. Requiring $\pm 3\sigma$ separation between
adjacent tones, $\sigma_{\hat f,W} \le \Delta f/6$, gives

$$W \;\ge\; \frac{3 f_s}{\pi \sqrt{\gamma}\, \Delta f}
\qquad\text{and combining with } W < f_s/R_s:$$

$$\boxed{\;\gamma \;>\; \left(\frac{3 R_s}{\pi \Delta f}\right)^{2}\;}$$

A hard SNR floor for tone separation that depends only on the ratio of symbol
rate to tone spacing. For DMR ($R_s = 4800$, $\Delta f = 1296$) it is **11 dB**;
for BLE ($R_s = 10^6$, $\Delta f = 500$ kHz) it is **5.6 dB**. Narrowband M-ary
FSK is intrinsically the hardest case, and no estimator design escapes this.
""")

code(S(PID, "inst_freq"))

code(r"""
fs, snr_db = 48e3, 25.0
clean = gen_gfsk(fs, 4800, 1944, 400, bt=0.6, m=4)
noisy = awgn(clean, snr_db)

gamma = 10 ** (snr_db / 10)
pred = fs / (2 * np.pi * math.sqrt(gamma))
meas = np.std(inst_freq(noisy, fs) - inst_freq(clean, fs)[:inst_freq(noisy, fs).size])

print(f"predicted sigma_f = fs/(2*pi*sqrt(gamma)) = {pred:7.1f} Hz")
print(f"measured                                  = {meas:7.1f} Hz")
print(f"\nDMR 4-FSK adjacent tone spacing           = {2*1944/3:7.1f} Hz")

print("\nTone-separation SNR floor, gamma > (3*Rs/(pi*df))^2:")
for nm, rs, df in (("DMR 4-FSK", 4800, 2*1944/3), ("BLE 1M 2-FSK", 1e6, 500e3),
                   ("SiK 2-FSK", 250e3, 120e3), ("POCSAG 2-FSK", 1200, 9000)):
    g = (3 * rs / (math.pi * df)) ** 2
    print(f"   {nm:14s} Rs={rs:8.0f}  df={df:8.0f} -> {10*math.log10(g):+6.1f} dB")

W = np.arange(1, 33)
plt.semilogy(W, fs / (2 * np.pi * W * math.sqrt(gamma)), label=r"telescoping: $\propto 1/W$")
plt.semilogy(W, fs / (2 * np.pi * np.sqrt(W) * math.sqrt(gamma)), ls="--",
             label=r"naive averaging: $\propto 1/\sqrt{W}$")
plt.axhline(2 * 1944 / 3 / 6, color="k", ls=":", label=r"required: $\Delta f/6$")
plt.axvline(fs / 4800, color="tab:red", ls=":", label="one symbol (upper limit on W)")
plt.xlabel("smoothing window W (samples)"); plt.ylabel("$\\sigma_f$ (Hz)")
plt.legend(fontsize=7); plt.title("Discriminator noise vs smoothing, DMR at 25 dB")
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. OFDM: detecting the cyclic prefix

An OFDM symbol is an IFFT of $N$ subcarriers, prefixed by a copy of its own
last $L$ samples:

$$x[n] = x[n+N] \qquad \text{for the } L \text{ samples of each cyclic prefix.}$$

That exact repetition is a strong, protocol-agnostic signature. Correlate the
signal against a delayed copy of itself:

$$\hat R(\tau) = \sum_n x[n]\,x^*[n+\tau]$$

computed via FFT in $O(M\log M)$. At $\tau = N$ the CP samples add
**coherently** while everything else adds incoherently, so with $M$ total
samples:

$$\frac{|\hat R(N)|}{\hat R(0)} \;\approx\; \frac{L}{N+L},
\qquad \text{noise floor of the estimate} \;\sim\; \frac{1}{\sqrt{M}}$$

Requiring a detection SNR of about 5 gives the record-length condition

$$\boxed{\;M \;\gtrsim\; 25\left(\frac{N+L}{L}\right)^{2}\;}$$

For Wi-Fi ($N{=}64$, $L{=}16$) that is only 625 samples. For LTE ($N{=}1024$,
$L{=}72$) it is 5800. **This is why the threshold matters:** LTE's normal cyclic
prefix is 6.6% of the symbol, so a hard threshold of 0.08 on the normalised peak
rejects LTE outright — which it did, until the threshold was dropped to 0.03 and
the decision moved onto the robust $z$-score instead.

Recovering $N$ gives the **subcarrier spacing** $\Delta f_{sc} = f_s/N$
directly, and that single number separates Wi-Fi (312.5 kHz) from LTE (15 kHz)
from 5G NR (30 kHz) from DAB (1 kHz) — the most decisive measurement in the
whole classifier.

Two guards worth noting. A peak pinned to either end of the search range is
rejected: a chirp has high short-lag autocorrelation and will otherwise
masquerade as OFDM with a tiny FFT size. And repeated training symbols (802.11's
L-STF is ten repeats of 16 samples) also produce peaks here — harmless, since
they still indicate OFDM, but it means the recovered "FFT size" is occasionally
a training-sequence period.
""")

code(S(PID, "cp_autocorrelation", "estimate_cp_length"))

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
for a, (title, x, fs, Nt, Lt) in zip(ax, [
        ("Wi-Fi class: N=64, CP=16", gen_ofdm(20e6, 64, 16, 52, 400), 20e6, 64, 16),
        ("LTE class: N=1024, CP=72", gen_ofdm(15.36e6, 1024, 72, 600, 60), 15.36e6, 1024, 72)]):
    x = awgn(x, 25)
    ac = cp_autocorrelation(x, min_lag=8, max_lag=2048)
    lags, vals = ac["curve"]
    a.plot(lags, vals, lw=0.6)
    a.axvline(Nt, color="tab:red", ls="--", lw=1, label=f"true N={Nt}")
    a.axhline(Lt / (Nt + Lt), color="tab:green", ls=":", label=f"L/(N+L)={Lt/(Nt+Lt):.3f}")
    cp = estimate_cp_length(x, ac["lag"])
    a.set_title(f"{title}\nfound N={ac['lag']}, CP={cp}, z={ac['score']:.0f}, "
                f"SCS={fs/ac['lag']/1e3:.1f} kHz", fontsize=8)
    a.set_xlabel("lag (samples)"); a.legend(fontsize=7)
    print(f"{title}: predicted peak {Lt/(Nt+Lt):.4f}, measured {ac['value']:.4f}, "
          f"min record 25((N+L)/L)^2 = {25*((Nt+Lt)/Lt)**2:.0f} samples")
ax[0].set_ylabel(r"$|\hat R(\tau)|/\hat R(0)$")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 11. Chirp spread spectrum (LoRa): detect by dechirping

A LoRa symbol sweeps the entire bandwidth $B$ exactly once per symbol. With
spreading factor $\mathrm{SF}$, the symbol period and chirp rate are

$$T_s = \frac{2^{\mathrm{SF}}}{B}, \qquad
k = \frac{B}{T_s} = \frac{B^2}{2^{\mathrm{SF}}}\ \ [\text{Hz/s}],
\qquad f(t) = -\frac{B}{2} + kt \ (\mathrm{mod}\ B)$$

### The approach that failed

The natural idea is to test whether $\mathrm{d}f/\mathrm{d}t$ is constant. It
does not work, and §9 tells us exactly why. Differentiating the discriminator a
*second* time compounds the noise: at 25 dB SNR the noise on the estimated slope
came out the same order as the slope itself, and the linearity score for a true
LoRa signal was **0.27** where it needed to exceed 0.8.

### Dechirping

Multiply by the conjugate of a reference chirp. The quadratic phase cancels and
each symbol collapses to a **single tone** whose frequency is the symbol value:

$$y(t) = x(t)\,e^{-j\pi k t^2} \;\longrightarrow\; \text{tone at } f_0 = \text{symbol offset}$$

An FFT over one symbol has resolution $1/T_s = B/2^{\mathrm{SF}}$, giving exactly
$2^{\mathrm{SF}}$ resolvable bins — which is the LoRa symbol alphabet, as it must
be. We score the fraction of per-symbol FFT energy landing in the peak bin:
$\approx 0.9$ for a matched dechirp, $\approx 0.45$ if the symbol boundary is
misaligned (a cyclic chirp splits into two tones), and $\approx 1/N$ for
anything else.

### Why we search the standard bandwidth grid

A chirp-rate error $\Delta k$ leaves residual sweep, spreading the tone across

$$\Delta k \cdot T_s^2 = \frac{\Delta k}{k}\,k\,T_s^2
= \frac{\Delta k}{k}\,2^{\mathrm{SF}} \ \text{bins}
\qquad(\text{since } k T_s^2 = 2^{\mathrm{SF}})$$

Holding the spread under ~6 bins needs $\Delta k/k \le 6/2^{\mathrm{SF}}$, and
since $k \propto B^2$ we have $\Delta k/k = 2\Delta B/B$, so

$$\boxed{\;\frac{\Delta B}{B} \;\le\; \frac{3}{2^{\mathrm{SF}}}\;}$$

That is 2.3% at SF7 and **0.07%** at SF12 — far tighter than any measured $B_{99}$.
The measured bandwidth is unusable as the chirp parameter. Instead we search the
grid of standardised LoRa bandwidths, which is exact when the signal really is
LoRa. The measured $B_{99}$ (135.6 kHz for a true 125 kHz signal, 8.5% high) only
selects *which* grid entries to try.
""")

code(S(PID, "LORA_BANDWIDTHS", "detect_chirp"))

code(r"""
fs, bw, sf = 500e3, 125e3, 7
x = awgn(gen_lora(fs, bw, sf, 24), 20)
res = detect_chirp(x, fs, bw_hint=136e3)
print(f"detected: BW={res['bw']/1e3:.1f} kHz, SF={res['sf']}, "
      f"chirp rate={res['chirp_rate']/1e6:.1f} MHz/s, energy fraction={res['score']:.3f}")
print(f"truth:    BW={bw/1e3:.1f} kHz, SF={sf}, "
      f"k=B^2/2^SF={bw**2/2**sf/1e6:.1f} MHz/s")
print(f"\nbandwidth tolerance 3/2^SF at SF{sf}: "
      f"{3/2**sf*100:.2f}%  (vs 8.5% error in the measured B99)")

n = int(round(2 ** sf * fs / bw))
k = bw * bw / 2 ** sf
tt = (np.arange(x.size) % n) / fs
y = x * np.exp(-1j * np.pi * k * tt ** 2)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.0))
ax[0].specgram(x, NFFT=128, Fs=fs / 1e3, noverlap=96, cmap="magma")
ax[0].set_title("raw LoRa: sawtooth chirps", fontsize=8); ax[0].grid(False)
ax[0].set_ylabel("kHz")
ax[1].specgram(y, NFFT=128, Fs=fs / 1e3, noverlap=96, cmap="magma")
ax[1].set_title("after matched dechirp: tones", fontsize=8); ax[1].grid(False)

for sfx in range(5, 13):
    r = detect_chirp(x, fs, bw_hint=136e3)
    pass
scores = []
sfs = list(range(5, 13))
for s in sfs:
    nn = int(round(2 ** s * fs / bw))
    kk = bw * bw / 2 ** s
    t2 = (np.arange(x.size) % nn) / fs
    yy = x * np.exp(-1j * np.pi * kk * t2 ** 2)
    nb = min(x.size // nn, 24)
    blk = yy[:nb * nn].reshape(nb, nn)
    S = np.abs(np.fft.fft(blk, axis=1)) ** 2
    scores.append(np.median(S.max(axis=1) / S.sum(axis=1)))
ax[2].stem(sfs, scores)
ax[2].axvline(sf, color="tab:red", ls="--", lw=1)
ax[2].set_xlabel("trial SF"); ax[2].set_title("peak energy fraction vs trial SF", fontsize=8)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 12. Symbol rate from cyclostationarity

A digitally modulated signal is not wide-sense stationary — its statistics are
**periodic** with the symbol period $T$. For a linearly modulated stream
$s(t) = \sum_k a_k\,g(t-kT)$ with i.i.d. symbols,

$$\mathbb{E}\bigl|s(t)\bigr|^2 = \sigma_a^2 \sum_k \bigl|g(t-kT)\bigr|^2$$

which is periodic in $t$ with period $T$. Its Fourier series therefore contains
a **spectral line** at $1/T$, with coefficient proportional to

$$\int G(f)\,G^*\!\left(f - \tfrac{1}{T}\right)\mathrm{d}f$$

This integral is the crux. For a root-Nyquist pulse with excess bandwidth
$\alpha$, $G(f)$ and its shift by $1/T$ overlap only in the roll-off region, so
the line strength grows with $\alpha$ — and at $\alpha = 0$ (ideal Nyquist) the
line **vanishes entirely**. A perfectly shaped signal has no recoverable symbol
rate by this method. That is a real limitation, not an implementation gap.

### Three probes

Different modulations expose the periodicity in different places, so we try
three and keep the strongest line:

1. **Sign transitions of the discriminator**, $\bigl|\Delta\,\mathrm{sgn}(\hat f - \mathrm{med})\bigr|$ — an impulse train at symbol boundaries. Best for FSK/GFSK/MSK.
2. **Discriminator derivative magnitude** $|\Delta \hat f|$ — for heavily shaped FSK.
3. **Squared envelope** $|x|^2$ — for linear modulations, OOK, and pulsed signals.

### The harmonic trap

The lines appear at *every* multiple $m/T$, and the strongest is often not the
fundamental. Walking down to the fundamental needs care: an unconstrained search
over divisors let a single noise bin several octaves down hijack the estimate,
reporting a 1 Mchip/s ADS-B frame as **250 kHz**. The fix is to make the walk
**chained** — accepting $R_s/4$ requires having first accepted $R_s/2$ — and to
demand the sub-harmonic reach at least half the strength of the peak.
""")

code(S(PID, "_spectral_lines", "_fundamental", "estimate_symbol_rate"))

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.0))
demo = [("GFSK 250 kbps", gen_gfsk(2e6, 250e3, 60e3, 800), 2e6, 250e3, 246e3),
        ("QPSK RRC 5 Msym", gen_linear(20e6, 5e6, 2000), 20e6, 5e6, 5.3e6),
        ("PPM/OOK 1 Mchip", gen_ppm_ook(8e6, 1e6, 300), 8e6, 1e6, 5.1e6)]
for a, (title, x, fs, true_rs, bwh) in zip(ax, demo):
    x = awgn(x, 25)
    est = estimate_symbol_rate(x, fs, bw_hint=bwh)
    f = inst_freq(x, fs)
    probe = np.abs(np.diff(np.sign(f - np.median(f)))) if f.size > 512 else np.abs(x) ** 2
    got = _spectral_lines(probe, fs, bwh / 60, bwh * 1.6)
    if got:
        fr, S, nz = got
        a.semilogy(fr / 1e3, S / nz, lw=0.5)
        a.axvline(true_rs / 1e3, color="tab:red", ls="--", lw=1, label="true $R_s$")
        for m in (2, 3):
            a.axvline(true_rs * m / 1e3, color="tab:orange", ls=":", lw=0.8)
    a.set_title(f"{title}\nestimated {est['symbol_rate']/1e3:.0f} kHz "
                f"(true {true_rs/1e3:.0f}), SNR {est['line_snr']:.0f}x\n{est['method']}",
                fontsize=8)
    a.set_xlabel("cycle frequency (kHz)"); a.legend(fontsize=7)
ax[0].set_ylabel("line strength / median")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 13. Counting FSK tones with an eye diagram

Given the symbol rate we can count modulation levels. The obvious approach —
histogram the smoothed discriminator — fails for narrowband 4-FSK, and §9
explains why: the smoothing needed to beat the noise is comparable to a symbol
period, so the window **straddles transitions** and averages the inner levels
back into each other.

Measured on DMR 4-FSK at 25 dB, sweeping the blanket window:

| window $W$ | peaks found | levels |
|---|---|---|
| 1 | 1 | noise-dominated blob |
| 3 | 3 | marginal |
| 5 | 2 | inner levels merged |
| 7, 10, 15 | 1 | fully merged |

There is no good choice. Maximising the peak count instead just rewards noise —
that variant reported 3, 4, and 5 tones on 2-FSK and QPSK signals.

### Sample the decision points instead

With $R_s$ known, $\mathrm{sps} = f_s/R_s$, and we sample the discriminator once
per symbol at phase $p$:

$$v_p[m] = \hat f\bigl[\,\mathrm{round}(p + m \cdot \mathrm{sps})\,\bigr],
\qquad p^\star = \arg\max_p \operatorname{Var}\bigl(v_p\bigr)$$

Maximum variance identifies the **open eye**: at the symbol centre the levels
are maximally separated, while at a transition they collapse toward the mean.
This is the classic timing-recovery criterion, used here for a different
purpose. The histogram of $v_{p^\star}$ separates all $M$ levels cleanly, and on
the DMR case it recovers all four tones where every blanket window failed.

### Validating a peak set

Peaks alone are not enough — a QPSK discriminator throws up accidental bumps
that formed a "uniform triplet" and were classified as 4-FSK. M-ary FSK tones
are *uniformly spaced* with *comparable occupancy*, so we require spacing
CV < 0.30 and a min/max peak-height ratio > 0.35, and additionally require a
near-constant envelope (CV < 0.25) before any FSK verdict.
""")

code(S(PID, "_tone_peaks", "_validate_tones", "fsk_tones", "estimate_pulse_rate"))

code(r"""
fs, Rs, dev = 48e3, 4800.0, 1944.0
x = awgn(gen_gfsk(fs, Rs, dev, 900, bt=0.6, m=4), 25)
f = inst_freq(x, fs)
sps = fs / Rs

fig, ax = plt.subplots(1, 3, figsize=(13, 3.1))

for W in (1, 3, 5, 9, 15):
    fh = _smooth(f, W); fh = fh - np.median(fh)
    h, e = np.histogram(fh, bins=np.linspace(-6100, 6100, 129))
    ax[0].plot(0.5 * (e[:-1] + e[1:]), _smooth(h.astype(float), 5) / h.max(),
               lw=0.8, label=f"W={W}")
ax[0].set_title("blanket smoothing: no window works", fontsize=8)
ax[0].set_xlabel("Hz"); ax[0].legend(fontsize=7)

best = None
for ph in range(int(round(sps))):
    idx = np.round(np.arange(ph, f.size - 1, sps)).astype(int)
    idx = idx[idx < f.size]
    v = f[idx]
    if best is None or v.var() > best[0]:
        best = (v.var(), ph, v)
_, ph, v = best
tp = _tone_peaks(v)
ax[1].hist(v - np.median(v), bins=64, color="tab:blue", alpha=0.8)
for tone in tp[0]:
    ax[1].axvline(tone, color="tab:red", ls="--", lw=1)
for lvl in (-dev, -dev / 3, dev / 3, dev):
    ax[1].axvline(lvl, color="tab:green", ls=":", lw=1)
ax[1].set_title(f"eye sampling at phase {ph}: {tp[0].size} tones\n"
                "red=found, green=true levels", fontsize=8)
ax[1].set_xlabel("Hz")

varz = []
for p2 in range(int(round(sps))):
    idx = np.round(np.arange(p2, f.size - 1, sps)).astype(int)
    varz.append(f[idx[idx < f.size]].var())
ax[2].stem(range(len(varz)), varz)
ax[2].axvline(ph, color="tab:red", ls="--", lw=1, label=f"chosen p*={ph}")
ax[2].set_xlabel("sampling phase"); ax[2].set_title("variance identifies the open eye",
                                                    fontsize=8)
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.show()

r = fsk_tones(x, fs, bw_hint=6100, symbol_rate=Rs)
print(f"fsk_tones -> {r['n_tones']} tones via {r['method']}, "
      f"spacing {r['deviation']:.0f} Hz (true {2*dev/3:.0f} Hz), uniform={r['uniform']}")
""")

# ---------------------------------------------------------------------------
md(r"""
# Part III — Classification

## 14. Assembling the feature vector

`extract_features` runs the whole measurement chain on one burst. Two ordering
decisions in it are load-bearing:

1. **Amplitude statistics are taken before filtering.** CV and PAPR describe the
   signal as received; measuring them after the anti-alias filter would report
   the filter's envelope, not the modulation's.
2. **Symbol rate is estimated before tone counting**, because §13's eye sampling
   needs $\mathrm{sps}$. This dependency is why the pipeline is not a simple
   left-to-right sequence of independent features.
""")

code(S(PID, "Features", "extract_features"))

md(r"""
## 15. Deciding the modulation family

An ordered decision list rather than a trained classifier — each test is
something we derived, so a wrong answer is traceable to a specific measurement.
Order matters: CSS is tested first because a matched dechirp is nearly
unambiguous, and OFDM before OOK because a large envelope swing alone does not
distinguish them.
""")

code(S(PID, "_decide_modclass"))

md(r"""
## 16. The protocol database

Roughly 35 entries, each with the bandwidth, symbol rate, burst duration,
subcarrier spacing, and deviation ranges its standard permits — plus the
specification reference, so any classification can be checked against the
document.

`band` is the most powerful field. Given a tuner centre frequency it acts as a
hard gate, and without it several protocols are simply not separable: DMR and
P25 Phase 1 share bandwidth *and* symbol rate *and* modulation, differing only
in allocation and vocoder.
""")

code(S(PID, "MHz", "kHz", "PROTOCOLS"))

md(r"""
## 17. Scoring

### Per-feature range score

Measurements decay smoothly outside the permitted range, in octaves:

$$s_i = \max\!\left(0,\; 1 - \frac{\bigl|\log_2(v/v_{\text{ref}})\bigr|}{\text{tol}}\right)$$

Log-domain because RF quantities span decades: being 50 kHz off means something
entirely different at 100 kHz than at 20 MHz.

### Combination: gate times weighted geometric mean

$$\text{score} \;=\; g_{\text{mod}} \cdot \exp\!\left(\frac{\sum_i w_i \ln s_i}{\sum_i w_i}\right)$$

The **geometric** mean is the important choice. An arithmetic mean lets a
candidate survive one decisive contradiction by scoring well elsewhere; the
geometric mean sends the product toward zero if any single $s_i$ does, which
matches how the reasoning should work — a signal whose bandwidth is wrong by a
factor of eight is not that protocol, however well everything else fits. It is
also equivalent to a log-likelihood sum under a naive-Bayes-style independence
assumption.

Modulation family enters as a **multiplicative gate**, not a term in the mean.
As one term among several, a mismatched family was survivable: an OFDM-only
entry (5G NR) scored 50% on a QPSK signal. As a gate the same case scores 20%,
below the confidence floor.

### Confidence floor

A leader at 20% means *nothing in the database fits*, not *probably this one*.
Below `CONFIDENCE_FLOOR = 0.35` the report says so explicitly rather than naming
a protocol.
""")

code(S(PID, "MOD_COMPAT", "_range_score", "score_protocol"))
code(S(PID, "CONFIDENCE_FLOOR", "BurstResult", "identify"))
code(S(PID, "_hz", "format_report", "to_json"))

# ---------------------------------------------------------------------------
md(r"""
# Part IV — Validating the blind path

## 18. Eight synthetic protocols, over many noise realisations

Ground truth for both the modulation family and the protocol name.

**A methodological point that matters more than the code.** This suite was
originally run at one fixed seed and reported 8/8. Re-running it across twelve
independent noise realisations showed the DMR 4-FSK case passing at *only that
seed* and failing at six others. A single-seed pass rate is not a measurement —
it is one sample of a random variable, and on a marginal case it is roughly a
coin flip.

Widening to a multi-seed sweep exposed four defects that the single-seed run had
concealed:

1. **Burst-detector time constants were absolute, not relative.** `win_s=2e-6`
   is 0.1 samples at 48 ksps, so the power envelope went unsmoothed and gap
   merging was disabled, fragmenting one burst into pieces. The largest fragment
   then had too few symbols for the eye histogram. Fixed with floors in samples
   (§5).
2. **A single derived smoothing window is fragile.** For narrowband 4-FSK,
   `round(sps/4)` resolved 2 of 4 levels while `w=3` resolved all four — and
   which one won shifted with the noise. Fixed by sweeping windows and letting
   the validation rules discard bad ones (§13).
3. **A latent `numpy` identity-vs-equality bug.** `pick in valid`, where the
   candidates are tuples containing arrays, falls back to element-wise `==` and
   raises *"truth value of an array is ambiguous"*. It had only ever appeared to
   work because `valid` usually held a single entry; widening the window sweep
   made multiple valid candidates common and it began throwing on six of eight
   cases. The exception was being swallowed by an `except ValueError` intended
   for short bursts — so the symptom was silently empty results, not a crash.
4. **The DMR generator was unrealistic.** C4FM is not Gaussian-filtered; BT=0.6
   over a 4-symbol span smeared the four levels well beyond what the standard
   implies. The test was measuring the generator, not the detector.

Current result: **94/96** across 12 seeds. The residual failures are the ADS-B
case being classified as a Mode A/C/S reply on 2 of 12 seeds — see §26.
""")

code(S(TPI, "cases"))
code(S(TPI, "main").replace("def main(", "def run_blind_validation("))

md(r"""
A single realisation first, at a fixed seed, so the per-case measurements are
visible.
""")

code(r"""
RNG = np.random.default_rng(0xC0FFEE)      # reproducible
status = run_blind_validation()
print(f"\nexit status {status}")
""")

md(r"""
Then the honest version: the same suite over independent noise realisations.
""")

code(r"""
def multiseed_validation(seeds):
    global RNG
    fails, total = {}, 0
    for seed in seeds:
        RNG = np.random.default_rng(seed)
        for label, fs, fc, iq, want_mod, want_name in cases():
            total += 1
            res = identify(iq, fs, rf_center_hz=fc, top=3)
            if not res:
                fails.setdefault(label, []).append((seed, "no burst", ""))
                continue
            r = max(res, key=lambda q: q.features.duration)
            ok = (r.features.modclass == want_mod
                  and (want_name is None
                       or (r.best and want_name.lower() in r.best["name"].lower())))
            if not ok:
                fails.setdefault(label, []).append(
                    (seed, r.features.modclass, (r.best or {}).get("name", "")[:26]))
    return total, fails

seeds = range(1, 5) if QUICK else range(1, 13)
total, fails = multiseed_validation(seeds)
nf = sum(len(v) for v in fails.values())
print(f"passed {total - nf} / {total}  ({len(list(seeds))} seeds x 8 cases)\n")
for k, v in sorted(fails.items(), key=lambda kv: -len(kv[1])):
    print(f"  {k}: {len(v)}/{len(list(seeds))} failed -> {v[:2]}")
if not fails:
    print("  no failures")
RNG = np.random.default_rng(0xC0FFEE)
""")

md(r"""
## 19. Segmentation, and the blind SNR floor

The SNR sweep is the motivation for Part V. Classification is solid at 20–30 dB,
degrades through 15, and by 6 dB the modulation family is wrong. At 0 dB the
energy detector finds **no burst at all** — the deeper problem, because a
pipeline gated on burst detection cannot recover below that point however good
its features are.

The low-confidence flag fires correctly throughout the degraded region: the
classifier fails *honestly*, which is the minimum acceptable behaviour.
""")

code(S(TPI, "extra_checks"))
code("extra_checks()")

# ---------------------------------------------------------------------------
md(r"""
# Part V — The preamble matched filter

## 20. Why the floor exists

The 20 dB floor is structural. Every feature in Part II is a **second-order
statistic** — a bandwidth, an envelope variance, a histogram of a
nonlinearly-derived quantity. Two consequences:

1. Second-order statistics of a signal in noise converge at a rate no estimator
   design improves.
2. The frequency discriminator has a **threshold effect**. The small-noise
   approximation $\theta \approx \operatorname{Im}\{we^{-j\phi}\}/A$ underlying
   §9 requires $\sigma_\theta \ll 1$; below about 10 dB it fails and the
   discriminator produces clicks rather than a mildly noisy estimate.

A **matched filter** is a first-order statistic and is the provably optimal
detector for a *known* waveform in AWGN. The key observation is that the
waveform genuinely is known: most protocols open every frame with a fixed,
publicly documented preamble and sync word.
""")

md(r"""
## 21. Detection theory

Correlate against a reference $r$ of $N$ samples, normalised so the statistic is
scale-free:

$$\rho[n] \;=\; \frac{\bigl|\sum_{k=0}^{N-1} x[n+k]\,r^*[k]\bigr|}
{\bigl\|x[n{:}n{+}N]\bigr\|\;\bigl\|r\bigr\|} \;\in\; [0, 1]$$

### Noise only

For circular complex Gaussian noise, $\rho^2 \sim \mathrm{Beta}(1, N-1)$, so

$$\boxed{\;P(\rho > t) \;=\; \bigl(1 - t^2\bigr)^{N-1} \;\approx\; e^{-(N-1)t^2}\;} \tag{1}$$

The false-alarm probability falls **exponentially in template length**. This is
the entire mechanism: the threshold needed for a fixed $P_{fa}$ shrinks as
$1/\sqrt{N}$.

### Signal present

With a matched, constant-amplitude reference, write $x = Ar + w$. The numerator
accumulates coherently, $\sum x_k r_k^* = AN + \sum w_k r_k^*$, where the second
term is complex Gaussian of variance $N\sigma^2$ — growing as $\sqrt{N}$ against
the signal's $N$. The denominator is $\sqrt{N(A^2+\sigma^2)}\cdot\sqrt{N}$. Hence

$$\boxed{\;\mathbb{E}[\rho] \;\approx\; \sqrt{\frac{\gamma}{1+\gamma}}\;} \tag{2}$$

**Note what (2) does not contain: $N$.** The peak correlation is set purely by
SNR and saturates at 1. Longer templates do not raise the peak — all of their
benefit lives in (1), in the threshold we are permitted to use. This is a
genuinely counter-intuitive point and §25 verifies it directly across a 21×
range of template lengths.

### Threshold and floor

With $M$ effective trials (positions × frequency bins × templates), invert (1):

$$\boxed{\;t \;=\; \sqrt{\frac{\ln(M/P_{fa})}{N-1}}\;} \tag{3}$$

Setting (2) above (3) and using $\gamma/(1+\gamma)\approx\gamma$ for small
$\gamma$:

$$\boxed{\;\gamma_{\min} \;\approx\; \frac{\ln(M/P_{fa})}{N}\;} \tag{4}$$

The processing gain is $N/\ln(M/P_{fa})$ — template length, discounted by the
logarithm of the search space. Searching harder costs remarkably little.

| Template | $N$ | threshold $t$ | $\gamma_{\min}$ |
|---|---|---|---|
| SiK, 28 bits @ 8 sps | 224 | 0.32 | −9.4 dB |
| BLE advertising, 40 bits @ 8 sps | 320 | 0.27 | −11.1 dB |
| POCSAG, 128 bits @ 27 sps (after 2x decimation) | 3413 | 0.08 | −21.7 dB |
| LoRa SF7, 8 upchirps @ 4 sps | 4096 | 0.08 | −22.5 dB |

This also explains something about LoRa that is otherwise mysterious: its
ability to demodulate below the noise floor is not exotic, it is equation (4)
with a very long preamble.
""")

code(S(PRE, "threshold_for_pfa", "snr_floor_db"))

md(r"""
## 22. Reference waveform synthesis and the template bank

All synthesis supports **fractional** samples-per-symbol, because the capture
rate is whatever the SDR provided and is rarely an integer multiple of the
protocol's symbol rate.

### How much of each template is actually determined by its standard

This distinction matters more than any code here, so it is recorded per template
in a `verified` flag:

- **BLE advertising** — fully determined. The 1M preamble is `0xAA` and the
  advertising access address is the fixed constant `0x8E89BED6`; whitening
  begins *after* the access address, so all 40 bits are deterministic. Bit
  order is the subtlety: octets transmit LSB-first and the address goes out
  least-significant-octet first.
- **ADS-B 1090ES** — fully determined. Four 0.5 µs pulses at 0, 1.0, 3.5 and
  4.5 µs.
- **LoRa** — structurally determined (8 base upchirps) even though the PHY
  itself is Semtech IP.
- **SiK/MAVLink** — flagged `verified=False`. `0x2DD4` is the Si4432 silicon
  default and what stock builds use, but both sync word and preamble length are
  register-configurable, so a modified deployment will not match. The flag is
  about configurability, not doubt about the constant.

### One template deliberately omitted

Zigbee/802.15.4 would be a good candidate — the O-QPSK symbol-to-chip mapping is
public and the 32-chip sequence for symbol zero would make a strong template. I
left it out because **I do not trust my recall of that sequence**, and the
failure mode is nasty: since the test generator would use the same constant, a
wrong sequence would correlate perfectly with itself and *pass its own test
suite* while never matching a real capture. Self-consistent tests cannot detect
a wrong constant. Adding it requires reading the table out of IEEE 802.15.4
directly.

The same caution applies to everything cited here — these are constants recalled
from specifications, not read out of them, and the BLE bit order in particular is
worth verifying against the document before trusting a real capture.
""")

code(S(PRE, "_bits", "synth_gfsk", "synth_css_upchirps", "synth_ook_pulses"))
code(S(PRE, "Template"))
code(S(PRE, "MHz", "_BLE_AA", "_BLE_PREAMBLE_1M", "BLE_ADV_1M", "BLE_ADV_2M",
       "ADSB_PREAMBLE", "lora_preamble", "sik_preamble", "pocsag_preamble",
       "default_templates"))

code(r"""
# verify the BLE access-address bit order by reconstructing it
def lsb_first(byte):
    return "".join(str((byte >> i) & 1) for i in range(8))

aa = 0x8E89BED6
octets = [(aa >> s) & 0xFF for s in (0, 8, 16, 24)]     # LSO first: D6 BE 89 8E
rebuilt = "".join(lsb_first(o) for o in octets)
print(f"access address 0x{aa:08X} -> octets LSO first: {[f'{o:02X}' for o in octets]}")
print(f"rebuilt : {rebuilt}")
print(f"in module: {_BLE_AA}")
print(f"match: {rebuilt == _BLE_AA}")
""")

md(r"""
## 23. The correlator, and a nearly free frequency search

### Carrier offset is the main fragility

Coherent correlation is why this works and also what breaks it. A residual
offset $\Delta f$ rotates the reference through $\Delta f \cdot T$ cycles over
the template duration, and the coherent sum degrades as $|\mathrm{sinc}(\Delta f T)|$
— one full cycle destroys the gain completely. At 915 MHz a 10 ppm crystal is
9 kHz, and over a 112 µs SiK preamble that *is* one full cycle. So we must
search a frequency grid, spaced at $0.8/T$ to hold worst-case scalloping loss
near 1 dB.

### The grid is almost free

De-rotating the signal by $-\Delta f$ is equivalent to rotating the reference by
$+\Delta f$. Writing the correlation as a convolution with
$h[j] = r^*[N-1-j]$, the rotated kernel is

$$h_{\Delta f}[j] = r^*[N{-}1{-}j]\,e^{-j2\pi\Delta f (N-1-j)/f_s}
= \underbrace{e^{-j2\pi \Delta f (N-1)/f_s}}_{\text{constant phase}}\;
h[j]\,e^{+j2\pi\Delta f j/f_s}$$

Multiplying a sequence by that exponential **shifts its DFT** by
$k = \Delta f \cdot N_{\text{FFT}}/f_s$ bins, and the leading constant phase dies
inside the $|\cdot|$. So we transform the template **once** and circularly shift
its spectrum per frequency bin: one inverse FFT per bin instead of a fresh
correlation.

The shift direction is easy to get backwards — an earlier version used
`roll(H, -k)` and reported every offset with inverted sign while detecting
perfectly, since a symmetric grid hides the error in the peak. §25 checks the
sign against injected offsets for exactly this reason.
""")

code(S(PRE, "_next_fast_len", "normalized_correlate", "_cfo_grid"))

md(r"""
## 24. The bank, and template cross-talk

### A limit of equation (1)

Equation (1) bounds false alarms against **noise**. It says nothing about one
template correlating with a *different real signal*, and several of these
templates genuinely resemble each other — BLE 1M, BLE 2M and SiK all open with
an alternating GFSK pattern, so a BLE frame lights up all three. That is
cross-talk, not a false alarm, and no threshold fixes it because the correlation
really is present.

Two mechanisms handle it:

**Non-maximum suppression in time.** A frame is one protocol, so among hits
whose template spans overlap, only the largest margin $\rho/t$ survives.
Suppressed hits are retained on `.shadowed`. *Limitation:* two genuinely
concurrent signals on different channels overlap in time and the weaker is lost
— separate by frequency first if that case matters.

**A consistency gate, from equation (2).** A truly matched template must return
$\rho \approx \sqrt{\gamma/(1+\gamma)}$, so its *implied* SNR should track the
SNR measured on the burst. A template correlating with a frame's random payload
clears the noise-only threshold but implies far less signal than is present —
this is how a BLE 2M template found a "preamble" inside a 30 dB BLE 1M payload,
reporting an implied −2.0 dB inside a +31.3 dB burst.

The gate carries its own trap, and it is instructive. Applied unconditionally it
**cost 12 dB of POCSAG sensitivity**: near the energy detector's floor the
measured burst SNR is whatever a noise fluctuation happened to do, and a spurious
+2.7 dB burst vetoed a perfectly good correlation at −12 dB. It is now applied
only to bursts above 10 dB, where the measurement means something, and is
one-sided — a hit is only ever rejected for implying *less* signal than is
measurably present.

### The correlator runs on the raw capture

Deliberately not on detected bursts. Below roughly 6 dB the energy detector
finds nothing at all, so requiring it to fire first would reintroduce the exact
floor we are removing.

### Channel filtering buys no sensitivity here

Worth stating precisely, because it contradicts §8. In the language of (4),
decimating by $D$ raises $\gamma$ by $\approx D$ and shrinks $N$ by the same
factor, leaving $\gamma N$ — and therefore the floor — invariant. The matched
filter already *is* the optimal filter. Decimation is retained purely for
compute.
""")

code(S(PRE, "PreambleHit"))
code(S(PRE, "PreambleBank"))

# ---------------------------------------------------------------------------
md(r"""
## 25. Validation

Four checks, then the headline measurement.
""")

code(S(TPR, "awgn_at", "frame_gfsk", "frame_lora"))
code(S(TPR, "test_correlator_matches_brute_force"))
code("_ = test_correlator_matches_brute_force()")

md(r"""
### Equation (2) and its independence of $N$

The claim to test is the surprising one: a template of 320 samples and one of
6827 samples should track the *same* $\rho$ curve. If they do, the processing
gain really does live entirely in the threshold.
""")

code(S(TPR, "test_rho_vs_snr"))
code("_ = test_rho_vs_snr()")

md(r"""
### Sign of the frequency estimate

Detection is insensitive to the shift direction because the grid is symmetric,
so this needs an explicit test against injected offsets. It also confirms the
$0.8/T$ grid spacing holds the scalloping loss to a fraction of a dB across the
full search span.
""")

code(r"""
fs = 8e6
r = BLE_ADV_1M.synth(fs)
grid = _cfo_grid(BLE_ADV_1M.cfo_span, BLE_ADV_1M.duration())
print(f"grid: {grid.size} bins spaced {grid[1]-grid[0]:.0f} Hz "
      f"(0.8/T with T={BLE_ADV_1M.duration()*1e6:.0f} us)")
rhos = []
for true in (-120e3, -80e3, -37e3, 0.0, 37e3, 80e3, 120e3):
    x = ((RNG.standard_normal(1500) + 1j * RNG.standard_normal(1500)) * 0.3).astype(np.complex64)
    x[400:400 + r.size] += r * np.exp(2j * np.pi * true * np.arange(r.size) / fs)
    rho, cfo = normalized_correlate(x, r, grid, fs)
    k = int(np.argmax(rho)); rhos.append(rho[k])
    print(f"  injected {true/1e3:+7.1f} kHz -> reported {cfo[k]/1e3:+7.1f} kHz  "
          f"(sample {k}, rho={rho[k]:.4f})")
print(f"\nscalloping loss across the span: "
      f"{20*math.log10(max(rhos)/min(rhos)):.2f} dB")
""")

md(r"""
### False alarms

Equation (1) predicts essentially none. Verifying on noise-only captures across
three bands checks that the threshold derivation is right, and that the
normalisation really is scale-free.

One caveat on interpretation. The trial count $M$ fed to equation (3) is a
*heuristic*: adjacent correlation samples are dependent over a template length,
so the number of independent positions is estimated as the record length in
template durations, inflated by 4 for safety. That makes the realised $P_{fa}$
approximate rather than exactly the nominal $10^{-6}$. Over 80 independent
noise scans I measured zero hits, but a different seed produced one — consistent
with a rate somewhat above nominal. If you need a calibrated false-alarm rate,
measure it empirically on your own noise rather than trusting the analytic
value.
""")

code(S(TPR, "test_false_alarms"))
code(r"""
RNG = np.random.default_rng(0xBEEF)        # reproducible
_ = test_false_alarms()
""")

md(r"""
### Carrier-offset robustness

Detection should hold across the whole grid span at 0 dB SNR and fail outside
it. The failure outside the span is expected behaviour, not a defect — it marks
where `cfo_span` needs widening for a given oscillator tolerance.
""")

code(S(TPR, "_blind_ok", "_preamble_ok"))
code(S(TPR, "test_cfo_robustness"))
code("_ = test_cfo_robustness()")

md(r"""
### The headline: measured SNR floors

Lowest SNR at which the protocol is identified on every trial. Published values
used 10 trials over a fine grid; with `QUICK = True` this runs 3 over a coarse
one and will land within a few dB.

| Protocol | Blind | Preamble | Gain | $N$ | Theory (4) |
|---|---|---|---|---|---|
| SiK/MAVLink 250 kbps | +25 dB | **−6 dB** | +31 dB | 224 | −9.4 dB |
| BLE 1M advertising | +25 dB | **−6 dB** | +31 dB | 320 | −11.1 dB |
| LoRa BW125 SF7 | +20 dB | **−18 dB** | +38 dB | 4096 | −22.5 dB |
| POCSAG 1200 bps | +25 dB | **−18 dB** | +43 dB | 3413 | −21.7 dB |

Measured floors sit 3–7 dB above theory, and the reasons are known rather than
mysterious: GFSK template mismatch from Gaussian ISI at the preamble boundary,
frequency-grid quantisation, and an all-trials-must-pass criterion that is
stricter than marginal detection. The *ordering* and the *scaling with $N$* both
follow the prediction.
""")

code(r"""
from iq_protocol_id import identify as _ident   # ensure the notebook's identify is used
def measure_floor(make, fs, fc, want, bank, mode, grid, trials):
    floor = None
    for snr in grid:
        good = sum(bool(_blind_ok(make(snr), fs, fc, want) if mode == "blind"
                        else _preamble_ok(make(snr), fs, fc, want, bank))
                   for _ in range(trials))
        if good == trials:
            floor = snr
        else:
            break
    return floor

bank = PreambleBank(p_fa=1e-6)
trials = 3 if QUICK else 10
grid = ([25, 10, 0, -6, -12, -18, -24] if QUICK
        else [30, 25, 20, 15, 10, 6, 3, 0, -3, -6, -9, -12, -15, -18, -21, -24])

sik = sik_preamble(250e3, 60e3)
lora_t = lora_preamble(125e3, 7)
poc = pocsag_preamble(1200.0, 4.5e3)
suite = [
    ("SiK/MAVLink 250 kbps", 2e6, 915e6, "SiK", sik,
     lambda s: frame_gfsk(sik, 2e6, 600, s, 1e-3)),
    ("BLE 1M advertising", 8e6, 2.44e9, "Bluetooth LE 1M", BLE_ADV_1M,
     lambda s: frame_gfsk(BLE_ADV_1M, 8e6, 300, s, 200e-6)),
    ("LoRa BW125 SF7", 500e3, 868e6, "LoRa", lora_t,
     lambda s: frame_lora(500e3, 125e3, 7, 20, s, 4e-3)),
    ("POCSAG 1200 bps", 64e3, 148e6, "POCSAG", poc,
     lambda s: frame_gfsk(poc, 64e3, 300, s, 20e-3)),
]

print(f"{'protocol':24s} {'blind':>8} {'preamble':>10} {'gain':>8} {'N':>7} {'theory':>9}")
for label, fs_, fc_, want, tpl, make in suite:
    fb = measure_floor(make, fs_, fc_, want, bank, "blind", grid, trials)
    fp = measure_floor(make, fs_, fc_, want, bank, "preamble", grid, trials)
    # N as the bank actually sees it, after its own channel decimation
    _, fsd, _dd = bank._decimate(np.zeros(4096, np.complex64), fs_, tpl)
    n = tpl.synth(fsd).size
    g = f"{fb - fp:+d} dB" if None not in (fb, fp) else "—"
    print(f"{label:24s} {str(fb)+' dB':>8} {str(fp)+' dB':>10} {g:>8} {n:7d} "
          f"{snr_floor_db(n, 10000):+8.1f} dB")
""")

md(r"""
### End to end, at an SNR where blind analysis has no chance

The same capture, with and without the bank.
""")

code(S(TPR, "test_end_to_end").replace(
    "    from iq_protocol_id import format_report\n", ""))
code("_ = test_end_to_end()")

# ---------------------------------------------------------------------------
md(r"""
# Part VI

## 26. Limitations

Honest accounting of what this does not do.

### Constants recalled, not read
Every specification constant here — the BLE access address and its bit order,
the ADS-B pulse positions, the Si4432 sync word, the protocol database's
bandwidth and rate ranges — comes from memory of the standards rather than from
the documents. §22 explains why the Zigbee chip sequence was omitted rather than
guessed, and the same skepticism should be applied to everything else before
trusting a real capture. Self-consistent tests **cannot** catch a wrong constant.

### The blind path
- **Band information is close to essential.** Without `rf_center_hz`, DMR and
  P25 are not separable, and SiK / Z-Wave / Wi-SUN 2-FSK overlap heavily — the
  scorer honestly reports several at 100%.
- **Ideal Nyquist shaping ($\alpha = 0$) has no recoverable symbol rate** by the
  cyclostationary method (§12). The spectral line genuinely vanishes.
- **Deviation estimates under-read** for heavily shaped GFSK at low samples per
  symbol: the DMR case recovers 4 tones but spacing biased by ISI.
- **Only AWGN is modelled.** No multipath, no fading, no adjacent-channel
  interference, no phase noise, no IQ imbalance or DC offset. Real captures have
  all of these, and frequency-selective fading in particular will break the
  bandwidth measurements that carry most of the classifier's information.
- **The "FFT size" from §10 is sometimes a training-sequence period.**
- **ADS-B vs Mode A/C/S replies are only partly separable** (2 of 12 seeds
  misclassify). Both sit at 1090 MHz with the same modulation and bandwidth,
  differing essentially in frame length; when the run-length chip-rate estimate
  wanders, duration alone is not always enough.

### Methodology
- **Single-seed test results are not measurements.** §18 documents four defects
  that a fixed seed concealed, including one that made six of eight cases return
  silently empty results. Any pass rate quoted here without a seed count should
  be distrusted, including the Part V floors — those used 10 trials per SNR
  point, which pins the floor to a few dB, not exactly.
- **Swallowing exceptions hides bugs.** The `except ValueError` in `identify`,
  intended for bursts too short to analyse, absorbed a genuine `numpy`
  broadcasting error for some time. Narrow the exception or re-raise unexpected
  ones.

### The preamble path
- **Only covers protocols with fixed preambles.** Anything negotiated
  (Bluetooth Classic access codes derive from the piconet LAP) needs the
  parameter before a template can be built.
- **Concurrent overlapping signals:** NMS keeps the stronger one (§24).
- **Cross-talk is bounded empirically, not analytically.** Equation (1) covers
  noise; the mutual coherence between templates is not characterised, and adding
  templates raises cross-talk in a way this suite does not measure.
- **CFO span is a fixed per-template guess.** Outside it, detection fails
  silently.
- **No Doppler or timing drift.** A long template assumes a stationary offset
  over its whole duration; the POCSAG template spans 107 ms, over which real
  oscillator drift is not negligible.

### Natural extensions
Demodulate after a hit (timing and CFO both fall out of the correlator, which is
most of acquisition); estimate the noise floor per-channel rather than
per-record; add a proper cyclic spectral correlation surface for blind symbol
rate at $\alpha = 0$; and replace the per-template CFO span with an offset
estimate from the coarse spectral centre.

## Summary

The classifier is interpretable by construction: every score decomposes into
per-feature terms that trace back to a derivation, so a wrong answer identifies
which measurement disagreed rather than requiring the model to be retrained.

The result worth carrying away is the division of labour. Blind features are
general — they classify anything, including protocols not in the database — but
are fundamentally limited to about 20 dB because they are second-order
statistics. A matched filter is narrow, requiring a documented preamble, but
converts template length into processing gain at the exponential rate of
equation (1), reaching −18 dB. Neither replaces the other: the correlator
supplies identification where it applies, and the blind features supply the
measurements it cannot produce and the corroboration that catches its
cross-talk.
""")


# ===========================================================================
def build() -> dict:
    cells = []
    for kind, text in CELLS:
        lines = text.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": source})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12",
                              "mimetype": "text/x-python",
                              "file_extension": ".py",
                              "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    out = ROOT / "protocol_id_tutorial.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    nc = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    nm = len(nb["cells"]) - nc
    print(f"wrote {out}  ({len(nb['cells'])} cells: {nm} markdown, {nc} code)")
