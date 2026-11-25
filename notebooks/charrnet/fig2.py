"""Reproduce Fig. 2: equivariance / invariance of layer outputs under the
ideal channel action (per-window multiplication of the STFT by H(f)), the
physical action (time-domain convolution), and physical action + Kaiser window.

Metric: relative error ||f(T x) - T' f(x)|| / ||f(x)|| (T' = identity for
invariance tests). For scale, the same metric is reported for swapping the
transmitter (same burst, different device fingerprint) -- the signal the
network must keep."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import models, sim
from scipy import signal as sps

rng = np.random.default_rng(7)
pop = sim.DevicePopulation(65, rng, pole_sigma=0.1)
chans = [sim.TRAIN_NLOS200, sim.channel_configs()['LOS 10'], sim.channel_configs()['nLOS 30']]
cr = models.ChaRRNet(65, np.random.default_rng(1))
cr.weq.data = rng.standard_normal(cr.weq.shape).astype(np.float32)   # non-trivial weights
cr.winv.data = rng.standard_normal(cr.winv.shape).astype(np.float32)
bl = models.BaselineCNN(65, np.random.default_rng(1))

N = 200
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
res = {k: [] for k in ['eq_ideal', 'eq_phys_rect', 'eq_phys_kaiser', 'inv_ideal', 'inv_phys_rect',
                       'inv_phys_kaiser', 'bl_phys', 'inv_device', 'bl_device']}


def eq_target(x, Hs):
    """Apply the ideal action to the equivariant layer's output manifold values."""
    lr, th = cr.equivariant(x)
    return lr.data + np.log(np.abs(Hs))[None, None, None, :], th.data + np.angle(Hs)[None, None, None, :]


def eq_err(x_t, x0, Hs):
    lr, th = cr.equivariant(x_t)
    lr0, th0 = eq_target(x0, Hs)
    dth = (th.data - th0 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(np.sum((lr.data - lr0) ** 2 + dth ** 2)) / np.linalg.norm(lr0)


for i in range(N):
    dev, dev2 = rng.integers(0, 65, 2)
    burst = sim.wifi_burst(rng)
    x = pop.apply(burst, dev)[None]
    x2 = pop.apply(burst, dev2 if dev2 != dev else (dev + 1) % 65)[None]
    h = chans[i % 3].draw(rng)
    Hs = np.fft.fftshift(np.fft.fft(h, models.NFFT))
    xc = np.convolve(x[0], h)[: x.shape[1]][None]
    for win, tag in [('rect', 'rect'), ('kaiser', 'kaiser')]:
        p0 = models.ChaRRNet.prep(x, win)
        pc = models.ChaRRNet.prep(xc, win)
        res[f'eq_phys_{tag}'].append(eq_err(pc, p0, Hs))
        res[f'inv_phys_{tag}'].append(rel(cr.invariant(pc).data, cr.invariant(p0).data))
    # ideal action: multiply each Kaiser-windowed short-time spectrum by H(f)
    S = models.stft(x, 'kaiser')
    pi = np.concatenate(models.to_manifold(S * Hs), 1)
    p0 = models.ChaRRNet.prep(x, 'kaiser')
    res['eq_ideal'].append(eq_err(pi, p0, Hs))
    res['inv_ideal'].append(rel(cr.invariant(pi).data, cr.invariant(p0).data))
    res['inv_device'].append(rel(cr.invariant(models.ChaRRNet.prep(x2)).data, cr.invariant(p0).data))
    # baseline: first complex-conv layer + CReLU (its post-nonlinearity features)
    f = lambda z: bl.features(models.BaselineCNN.prep(z), upto=0).relu().data
    res['bl_phys'].append(rel(f(xc), f(x)))
    res['bl_device'].append(rel(f(x2), f(x)))

summ = {k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90))) for k, v in res.items()}
json.dump(summ, open('fig2.json', 'w'), indent=1)
for k, v in summ.items():
    print(f'{k:16s} median={v["median"]:.4f}  p90={v["p90"]:.4f}')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
labels = ['Ideal', 'Physical\n(rect)', 'Physical\n+ Kaiser']
ax[0].boxplot([res['eq_ideal'], res['eq_phys_rect'], res['eq_phys_kaiser']], showfliers=False)
ax[0].set_xticks([1, 2, 3], labels)
ax[0].set_title('ChaRRNet equivariant layer\n' r'$\|f(Tx)-T f(x)\|/\|Tf(x)\|$')
ax[0].set_yscale('log')
ax[1].boxplot([res['inv_ideal'], res['inv_phys_rect'], res['inv_phys_kaiser'], res['bl_phys'],
               res['inv_device'], res['bl_device']], showfliers=False)
ax[1].set_xticks(range(1, 7), ['ChaRR\nideal', 'ChaRR\nphys rect', 'ChaRR\nphys Kaiser',
                               'Baseline\nphys', 'ChaRR\nΔdevice', 'Baseline\nΔdevice'], fontsize=8)
ax[1].set_title('Invariance: relative change of output\n(channel actions vs. changing transmitter)')
ax[1].set_yscale('log')
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig('fig2_equivariance.png', dpi=130)
