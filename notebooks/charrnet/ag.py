"""Minimal reverse-mode autograd on NumPy (PyTorch was not available)."""
import numpy as np


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class T:
    def __init__(self, data, parents=(), back=None, param=False):
        self.data = np.asarray(data, np.float32) if not isinstance(data, np.ndarray) else data
        self.parents, self.back, self.grad, self.param = parents, back, None, param

    shape = property(lambda s: s.data.shape)

    # ---------------------------------------------------------- arithmetic
    def __add__(a, b):
        b = b if isinstance(b, T) else T(np.asarray(b, np.float32))
        return T(a.data + b.data, (a, b), lambda g: (_unbroadcast(g, a.shape), _unbroadcast(g, b.shape)))
    __radd__ = __add__

    def __sub__(a, b):
        b = b if isinstance(b, T) else T(np.asarray(b, np.float32))
        return T(a.data - b.data, (a, b), lambda g: (_unbroadcast(g, a.shape), -_unbroadcast(g, b.shape)))

    def __neg__(a):
        return T(-a.data, (a,), lambda g: (-g,))

    def __mul__(a, b):
        b = b if isinstance(b, T) else T(np.asarray(b, np.float32))
        return T(a.data * b.data, (a, b),
                 lambda g: (_unbroadcast(g * b.data, a.shape), _unbroadcast(g * a.data, b.shape)))
    __rmul__ = __mul__

    def __matmul__(a, b):
        return T(a.data @ b.data, (a, b), lambda g: (g @ b.data.T, a.data.T @ g))

    def sq(a):
        return T(a.data ** 2, (a,), lambda g: (2 * g * a.data,))

    def sqrt(a, eps=1e-6):
        o = np.sqrt(a.data + eps)
        return T(o, (a,), lambda g: (g / (2 * o),))

    def sin(a):
        return T(np.sin(a.data), (a,), lambda g: (g * np.cos(a.data),))

    def cos(a):
        return T(np.cos(a.data), (a,), lambda g: (-g * np.sin(a.data),))

    def relu(a):
        m = a.data > 0
        return T(a.data * m, (a,), lambda g: (g * m,))

    def wrap(a):
        """Wrap angle to [-pi, pi); gradient is identity almost everywhere."""
        return T((a.data + np.pi) % (2 * np.pi) - np.pi, (a,), lambda g: (g,))

    def reshape(a, *s):
        return T(a.data.reshape(*s), (a,), lambda g: (g.reshape(a.shape),))

    def transpose(a, *ax):
        inv = np.argsort(ax)
        return T(a.data.transpose(*ax), (a,), lambda g: (g.transpose(*inv),))

    # ---------------------------------------------------------- backprop
    def backward(self):
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t.parents:
                visit(p)
            order.append(t)
        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t.back is None or t.grad is None:
                continue
            for p, g in zip(t.parents, t.back(t.grad)):
                if g is None:
                    continue
                p.grad = g if p.grad is None else p.grad + g
            if not t.param:
                t.grad = None


def atan2(y, x):
    r2 = x.data ** 2 + y.data ** 2 + 1e-12
    return T(np.arctan2(y.data, x.data), (y, x), lambda g: (g * x.data / r2, -g * y.data / r2))


def concat(ts, axis):
    sizes = np.cumsum([t.shape[axis] for t in ts])[:-1]
    return T(np.concatenate([t.data for t in ts], axis), tuple(ts),
             lambda g: tuple(np.split(g, sizes, axis)))


def softmax(a, axis=-1):
    e = np.exp(a.data - a.data.max(axis, keepdims=True))
    s = e / e.sum(axis, keepdims=True)
    return T(s, (a,), lambda g: (s * (g - (g * s).sum(axis, keepdims=True)),))


# ---------------------------------------------------------------- conv / pool
def conv1d(x, w, b, pad):
    """x (B,C,L), w (O,C,K), b (O,) -> (B,O,L+2pad-K+1)."""
    B, C, L = x.shape
    O, _, K = w.shape
    xp = np.pad(x.data, ((0, 0), (0, 0), (pad, pad)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, K, axis=2)  # B,C,Lo,K
    Lo = cols.shape[2]
    cols = cols.transpose(0, 2, 1, 3).reshape(B * Lo, C * K)
    W = w.data.reshape(O, C * K)
    out = (cols @ W.T + b.data).reshape(B, Lo, O).transpose(0, 2, 1)

    def back(g):
        g2 = g.transpose(0, 2, 1).reshape(B * Lo, O)
        dW = (g2.T @ cols).reshape(O, C, K)
        dcols = (g2 @ W).reshape(B, Lo, C, K)
        dxp = np.zeros_like(xp)
        for k in range(K):
            dxp[:, :, k:k + Lo] += dcols[:, :, :, k].transpose(0, 2, 1)
        return dxp[:, :, pad:pad + L], dW, g2.sum(0)
    return T(np.ascontiguousarray(out), (x, w, b), back)


def maxpool1d(x, k):
    B, C, L = x.shape
    Lo = L // k
    xr = x.data[:, :, :Lo * k].reshape(B, C, Lo, k)
    idx = xr.argmax(-1)
    out = np.take_along_axis(xr, idx[..., None], -1)[..., 0]

    def back(g):
        d = np.zeros_like(xr)
        np.put_along_axis(d, idx[..., None], g[..., None], -1)
        dx = np.zeros_like(x.data)
        dx[:, :, :Lo * k] = d.reshape(B, C, Lo * k)
        return (dx,)
    return T(out, (x,), back)


def tconv(x, w, K, s):
    """Weighted sum over a strided window of the time axis and all channels.
    x (B,C,T,F), w (O,C,K) -> out[b,o,t,f] = sum_{c,k} w[o,c,k] x[b,c,t*s+k,f]."""
    B, C, Tn, F = x.shape
    To = (Tn - K) // s + 1
    idx = np.arange(To)[:, None] * s + np.arange(K)[None, :]       # To,K
    u = x.data[:, :, idx, :]                                          # B,C,To,K,F
    out = np.einsum('bctkf,ock->botf', u, w.data, optimize=True)

    def back(g):
        dw = np.einsum('botf,bctkf->ock', g, u, optimize=True)
        du = np.einsum('botf,ock->bctkf', g, w.data, optimize=True)
        dx = np.zeros_like(x.data)
        for k in range(K):
            dx[:, :, idx[:, k], :] += du[:, :, :, k, :]
        return dx, dw
    return T(out, (x, w), back)


def unfold_time(x, K, s):
    """x (B,C,T,F) -> (B,C,To,K,F) strided windows along time."""
    B, C, Tn, F = x.shape
    To = (Tn - K) // s + 1
    idx = np.arange(To)[:, None] * s + np.arange(K)[None, :]

    def back(g):
        dx = np.zeros_like(x.data)
        for k in range(K):
            dx[:, :, idx[:, k], :] += g[:, :, :, k, :]
        return (dx,)
    return T(x.data[:, :, idx, :], (x,), back)


def cross_entropy(logits, y):
    z = logits.data - logits.data.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()

    def back(g):
        d = p.copy()
        d[np.arange(n), y] -= 1
        return (g * d / n,)
    return T(np.asarray(loss), (logits,), back)


# ---------------------------------------------------------------- optimiser
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = params, lr, b1, b2, wd, 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.p, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.data -= (self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            p.grad = None


def param(shape, rng, scale=None):
    fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
    scale = np.sqrt(2.0 / fan_in) if scale is None else scale
    return T((rng.standard_normal(shape) * scale).astype(np.float32), param=True)


def zeros(shape):
    return T(np.zeros(shape, np.float32), param=True)
