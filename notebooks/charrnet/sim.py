"""Simulation substrate for the ChaRRNet reproduction (Brown, Mattei, Draganov 2021).

* 802.11a/g OFDM bursts (legacy STF + LTF + QPSK data symbols), 20 Msps.
* Device fingerprints from IIR low-pass filters with device-specific pole
  perturbations, applied *independently* to I and Q (paper Sec. II-A).
* Multipath channels from a randomised reflector geometry (paper Sec. III-A):
  per-reflector attenuation ~ U[-15, -5] dB, nominal reflector positions fixed
  per configuration and jittered per burst; optional line-of-sight path.
* AWGN and carrier-frequency-offset (CFO) augmentation.
"""
import numpy as np
from scipy import signal as sps

FS = 20e6          # sample rate
FC = 2.437e9       # carrier (channel 6), only used for reflection phases
C0 = 3e8

# ---------------------------------------------------------------- 802.11 bursts
_S = np.sqrt(13 / 6) * np.array(
    [0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0,
     -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0,
     0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0])
_L = np.array(
    [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
     1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1,
     1, -1, 1, -1, 1, 1, 1, 1], dtype=complex)
_PILOT_IDX = np.array([-21, -7, 7, 21])
_DATA_IDX = np.array([k for k in range(-26, 27) if k != 0 and k not in _PILOT_IDX])


def _ifft64(spec53):
    """Map subcarriers -26..26 to a 64-point IFFT."""
    X = np.zeros(64, complex)
    X[np.arange(-26, 27) % 64] = spec53
    return np.fft.ifft(X) * 64 / np.sqrt(52)


def wifi_burst(rng, n_data_syms=4):
    stf = np.tile(_ifft64(_S)[:16], 10)                     # 160 samples
    ltf_sym = _ifft64(_L)
    ltf = np.concatenate([ltf_sym[-32:], ltf_sym, ltf_sym])  # 160 samples
    syms = []
    for _ in range(n_data_syms):
        X = np.zeros(53, complex)
        X[_DATA_IDX + 26] = (rng.choice([-1, 1], 48) + 1j * rng.choice([-1, 1], 48)) / np.sqrt(2)
        X[_PILOT_IDX + 26] = [1, 1, 1, -1]
        s = _ifft64(X)
        syms.append(np.concatenate([s[-16:], s]))
    x = np.concatenate([stf, ltf] + syms)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


# ------------------------------------------------------------ device fingerprints
class DevicePopulation:
    """Each device: independent I and Q IIR low-pass filters (4th-order
    Butterworth prototype) whose poles are perturbed in radius and angle."""

    def __init__(self, n_devices, rng, order=4, cutoff=0.85, pole_sigma=0.04):
        b0, a0 = sps.butter(order, cutoff)
        z0, p0, k0 = sps.tf2zpk(b0, a0)
        self.filters = []
        for _ in range(n_devices):
            pair = []
            for _iq in range(2):
                p = p0.copy()
                # perturb each conjugate pair consistently (keeps filter real)
                upper = np.where(p.imag > 0)[0]
                for i in upper:
                    r = np.clip(np.abs(p[i]) * (1 + pole_sigma * rng.standard_normal()), 0.05, 0.97)
                    th = np.angle(p[i]) * (1 + pole_sigma * rng.standard_normal())
                    p[i] = r * np.exp(1j * th)
                    j = np.argmin(np.abs(p - np.conj(p0[i])))
                    p[j] = np.conj(p[i])
                b, a = sps.zpk2tf(z0, p, 1.0)
                b = b * (np.sum(a) / np.sum(b))   # unity DC gain
                pair.append((np.real(b), np.real(a)))
            self.filters.append(pair)

    def apply(self, x, dev):
        (bi, ai), (bq, aq) = self.filters[dev]
        return sps.lfilter(bi, ai, x.real) + 1j * sps.lfilter(bq, aq, x.imag)


# ---------------------------------------------------------------- channels
class ReflectorChannel:
    """Geometry-based multipath. Tx at origin, Rx at (d_los, 0). Nominal
    reflector positions are drawn once per configuration (seed), then each
    burst jitters them (``jitter_m``) and redraws attenuations."""

    def __init__(self, n_reflectors, los, seed, d_los=15.0, extent=25.0,
                 jitter_m=0.5, n_taps=16):
        g = np.random.default_rng(seed)
        self.nom = g.uniform([-extent, -extent], [d_los + extent, extent], (n_reflectors, 2))
        self.los, self.d_los, self.jitter, self.n_taps = los, d_los, jitter_m, n_taps

    def draw(self, rng):
        pos = self.nom + self.jitter * rng.standard_normal(self.nom.shape)
        L = np.linalg.norm(pos, axis=1) + np.linalg.norm(pos - [self.d_los, 0], axis=1)
        tau = (L - self.d_los) / C0                       # excess delay (s)
        amp = 10 ** (rng.uniform(-15, -5, len(L)) / 20)
        gains = amp * np.exp(-2j * np.pi * FC * L / C0)
        if self.los:
            tau = np.append(tau, 0.0)
            gains = np.append(gains, np.exp(-2j * np.pi * FC * self.d_los / C0))
        n = np.arange(self.n_taps)
        h = (gains[None, :] * np.sinc(n[:, None] - 2 - tau[None, :] * FS)
             * np.hanning(self.n_taps + 2)[1:-1, None]).sum(1)
        return h / np.linalg.norm(h)


def channel_configs():
    """Test configurations of paper Table I (+ the nLOS200 training config)."""
    return {
        "Pristine": None,
        "nLOS 10": ReflectorChannel(10, False, seed=10),
        "LOS 10": ReflectorChannel(10, True, seed=11),
        "nLOS 30": ReflectorChannel(30, False, seed=30),
        "LOS 100": ReflectorChannel(100, True, seed=100),
        "nLOS 400": ReflectorChannel(400, False, seed=400),
        "LOS 400": ReflectorChannel(400, True, seed=401),
    }


TRAIN_NLOS200 = ReflectorChannel(200, False, seed=200)


# ---------------------------------------------------------------- dataset
def make_dataset(pop, n_per_dev, channel, rng, snr_db=(15, 30), max_cfo=2e-3):
    """Returns X (N, T) complex64, y (N,). ``max_cfo`` in cycles/sample
    (2e-3 * 20 MHz = 40 kHz, about 16 ppm at 2.4 GHz)."""
    xs, ys = [], []
    n_dev = len(pop.filters)
    for dev in range(n_dev):
        for _ in range(n_per_dev):
            x = pop.apply(wifi_burst(rng), dev)
            if channel is not None:
                x = np.convolve(x, channel.draw(rng))[: len(x)]
            n = np.arange(len(x))
            x = x * np.exp(1j * (2 * np.pi * rng.uniform(-max_cfo, max_cfo) * n
                                 + rng.uniform(0, 2 * np.pi)))
            snr = rng.uniform(*snr_db)
            p = np.mean(np.abs(x) ** 2)
            x = x + np.sqrt(p / 10 ** (snr / 10) / 2) * (rng.standard_normal(len(x))
                                                         + 1j * rng.standard_normal(len(x)))
            xs.append(x / np.sqrt(np.mean(np.abs(x) ** 2)))
            ys.append(dev)
    return np.array(xs, np.complex64), np.array(ys)
