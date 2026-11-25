"""Baseline complex-valued CNN and ChaRRNet (scaled-down, NumPy)."""
import numpy as np
from scipy.signal.windows import kaiser
import ag
from ag import T


class Model:
    def params(self):
        return [v for v in self.__dict__.values() if isinstance(v, T) and v.param]

    def state(self):
        return {k: v.data.copy() for k, v in self.__dict__.items() if isinstance(v, T) and v.param}


def dense(x, W, b):
    return x @ W + b


# ======================================================================
# Baseline: complex convolutions on raw IQ (in the spirit of Mattei et al. 2019)
# ======================================================================
class BaselineCNN(Model):
    """Three complex conv layers (CReLU + max-pool) -> dense head."""

    def __init__(self, n_cls, rng, L=640, ch=(16, 32, 32), ks=(7, 5, 5), pools=(4, 4, 2), hid=128):
        self.cfg = dict(ch=ch, ks=ks, pools=pools)
        cin = 1
        for i, (c, k) in enumerate(zip(ch, ks)):
            sc = np.sqrt(1.0 / (cin * k))
            setattr(self, f'wr{i}', ag.param((c, cin, k), rng, sc))
            setattr(self, f'wi{i}', ag.param((c, cin, k), rng, sc))
            setattr(self, f'b{i}', ag.zeros((2 * c,)))
            cin = c
        Lf = L
        for p in pools:
            Lf //= p
        self.W1 = ag.param((2 * cin * Lf, hid), rng)
        self.c1 = ag.zeros((hid,))
        self.W2 = ag.param((hid, n_cls), rng, np.sqrt(1.0 / hid))
        self.c2 = ag.zeros((n_cls,))

    @staticmethod
    def prep(X):
        """complex (N,L) -> real (N,2,L) [I;Q] stacked as channels."""
        return np.stack([X.real, X.imag], 1).astype(np.float32)

    def complex_conv(self, h, i):
        wr, wi = getattr(self, f'wr{i}'), getattr(self, f'wi{i}')
        # [[Wr, -Wi], [Wi, Wr]] acting on [re; im]
        W = ag.concat([ag.concat([wr, -wi], 1), ag.concat([wi, wr], 1)], 0)
        k = wr.shape[2]
        return ag.conv1d(h, W, getattr(self, f'b{i}'), k // 2)

    def features(self, x, upto=None):
        h = T(x)
        for i, p in enumerate(self.cfg['pools']):
            h = self.complex_conv(h, i)
            if upto == i:
                return h
            h = ag.maxpool1d(h.relu(), p)
        return h

    def __call__(self, x):
        h = self.features(x)
        h = h.reshape(h.shape[0], -1)
        h = dense(h, self.W1, self.c1).relu()
        return dense(h, self.W2, self.c2)


# ======================================================================
# ChaRRNet
# ======================================================================
NFFT, HOP, KAISER_BETA = 128, 16, 8.0


def stft(X, window='kaiser'):
    """complex (N,L) -> complex (N,T,F) short-time spectra (fftshifted)."""
    w = kaiser(NFFT, KAISER_BETA) if window == 'kaiser' else np.ones(NFFT)
    fr = np.lib.stride_tricks.sliding_window_view(X, NFFT, axis=1)[:, ::HOP, :]
    return np.fft.fftshift(np.fft.fft(fr * w, axis=-1), axes=-1)


def to_manifold(S, eps=1e-4):
    """complex STFT -> (log-magnitude, angle), each (N,1,T,F) float32.
    Each element lives on R+ x U(1); log maps it to the tangent space."""
    lr = np.log(np.abs(S) + eps)[:, None].astype(np.float32)
    th = np.angle(S)[:, None].astype(np.float32)
    return lr, th


def wfm_conv(lr, th, w_raw, K, s):
    """Channel-equivariant layer: weighted Frechet mean on R+ x U(1),
    strided along the STFT-window axis, independently per frequency bin.
      log-mag : sum_i w_i ln r_i          (geometric mean -> scaling equivariant)
      angle   : atan2(sum w_i sin, sum w_i cos)   (rotation equivariant)
    Weights are convex (softmax over channels x kernel taps)."""
    O, C, Kk = w_raw.shape
    w = ag.softmax(w_raw.reshape(O, C * Kk), 1).reshape(O, C, Kk)
    lr_o = ag.tconv(lr, w, K, s)
    th_o = ag.atan2(ag.tconv(th.sin(), w, K, s), ag.tconv(th.cos(), w, K, s))
    return lr_o, th_o


def invariant_layer(lr, th, w_raw, K, s):
    """Channel-invariant layer: learned wFM over (channels x K windows), then the
    manifold distance  sqrt(ln^2(r'/r) + wrap(theta' - theta)^2)  from every input
    in the receptive field to that mean.  Output is real: (B, C*K, To, F)."""
    m_lr, m_th = wfm_conv(lr, th, w_raw, K, s)          # B,1,To,F
    u_lr = ag.unfold_time(lr, K, s)                      # B,C,To,K,F
    u_th = ag.unfold_time(th, K, s)
    B, C, To, Kk, F = u_lr.shape
    d_lr = u_lr - m_lr.reshape(B, 1, To, 1, F)
    d_th = (u_th - m_th.reshape(B, 1, To, 1, F)).wrap()
    d = (d_lr.sq() + d_th.sq()).sqrt()
    return d.transpose(0, 1, 3, 2, 4).reshape(B, C * Kk, To, F)


def g_transport(lr, th, a, p):
    """SurReal's G-transport: act on every value of channel c with the group element
    (e^{a_c}, R(p_c)), i.e. a per-channel translation in the tangent space. It commutes
    with the channel's per-bin translation, so equivariance is preserved."""
    return lr + a, th + p


class ChaRRNet(Model):
    def __init__(self, n_cls, rng, T_in=None, F=None, C1=8, K1=4, S1=2, K2=4, S2=4,
                 ch=(64, 64), hid=128, variant='base', K1b=3, S1b=1):
        """variant: 'base'    one wFM layer (as in the main results)
                   'deep'    two stacked wFM layers
                   'deep_gt' two wFM layers, each followed by G-transport (SurReal)"""
        assert variant in ('base', 'deep', 'deep_gt')
        self.variant, self.K1b, self.S1b = variant, K1b, S1b
        T_in = (640 - NFFT) // HOP + 1 if T_in is None else T_in
        F = NFFT if F is None else F
        self.K1, self.S1, self.K2, self.S2 = K1, S1, K2, S2
        self.weq = T((0.1 * rng.standard_normal((C1, 1, K1))).astype(np.float32), param=True)
        self.winv = T((0.1 * rng.standard_normal((1, C1, K2))).astype(np.float32), param=True)
        T1 = (T_in - K1) // S1 + 1
        if variant != 'base':
            T1 = (T1 - K1b) // S1b + 1
        T2 = (T1 - K2) // S2 + 1
        cin = C1 * K2 * T2
        self.cw0 = ag.param((ch[0], cin, 5), rng); self.cb0 = ag.zeros((ch[0],))
        self.cw1 = ag.param((ch[1], ch[0], 3), rng); self.cb1 = ag.zeros((ch[1],))
        self.W1 = ag.param((ch[1] * (F // 4), hid), rng); self.c1 = ag.zeros((hid,))
        self.W2 = ag.param((hid, n_cls), rng, np.sqrt(1.0 / hid)); self.c2 = ag.zeros((n_cls,))
        # learned per-input-channel normalisation of invariant features
        self.g = T(np.ones((1, cin, 1), np.float32), param=True)
        self.gb = ag.zeros((1, cin, 1))
        if variant != 'base':      # created last, so 'base' keeps its original initialisation
            self.weq2 = T((0.1 * rng.standard_normal((C1, C1, K1b))).astype(np.float32), param=True)
        if variant == 'deep_gt':   # G-transport: one learned (ln-gain, rotation) per channel and layer
            self.gt_a1, self.gt_p1 = ag.zeros((1, C1, 1, 1)), ag.zeros((1, C1, 1, 1))
            self.gt_a2, self.gt_p2 = ag.zeros((1, C1, 1, 1)), ag.zeros((1, C1, 1, 1))

    @staticmethod
    def prep(X, window='kaiser'):
        lr, th = to_manifold(stft(X, window))
        return np.concatenate([lr, th], 1)                # (N,2,T,F) cached input

    def equivariant(self, x):
        lr, th = T(x[:, :1]), T(x[:, 1:])
        lr, th = wfm_conv(lr, th, self.weq, self.K1, self.S1)
        if self.variant == 'base':
            return lr, th
        if self.variant == 'deep_gt':
            lr, th = g_transport(lr, th, self.gt_a1, self.gt_p1)
        lr, th = wfm_conv(lr, th, self.weq2, self.K1b, self.S1b)
        if self.variant == 'deep_gt':
            lr, th = g_transport(lr, th, self.gt_a2, self.gt_p2)
        return lr, th

    def invariant(self, x):
        lr, th = self.equivariant(x)
        return invariant_layer(lr, th, self.winv, self.K2, self.S2)

    def __call__(self, x):
        z = self.invariant(x)                              # B, C1*K2, T2, F
        B, Cz, T2, F = z.shape
        h = z.transpose(0, 1, 2, 3).reshape(B, Cz * T2, F)  # 1D CNN along frequency
        h = h * self.g + self.gb
        h = ag.maxpool1d(ag.conv1d(h, self.cw0, self.cb0, 2).relu(), 2)
        h = ag.maxpool1d(ag.conv1d(h, self.cw1, self.cb1, 1).relu(), 2)
        h = h.reshape(B, -1)
        h = dense(h, self.W1, self.c1).relu()
        return dense(h, self.W2, self.c2)
