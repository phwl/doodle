"""One-figure overview of the ChaRRNet idea and main simulation result."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import models, sim

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False})
BLUE, ORANGE, GREY, DARK = '#2f6db5', '#e07b28', '#9aa5b1', '#3e4c59'

# ------------------------------------------------------------------ data
rng = np.random.default_rng(11)
pop = sim.DevicePopulation(65, rng, pole_sigma=0.1)
x = pop.apply(sim.wifi_burst(rng), 3)
h = sim.ReflectorChannel(30, False, seed=30).draw(np.random.default_rng(0))
H = np.fft.fftshift(np.fft.fft(h, models.NFFT))
S0 = models.stft(x[None])[0]                 # clean   (T, F)
S1 = S0 * H                                  # through channel (ideal action)
F0 = 22                                      # highlighted frequency bin
fk = (np.arange(models.NFFT) - models.NFFT // 2) * 20 / models.NFFT   # MHz

res = {}
for f in ['res_bl_p.json', 'res_bl_n.json', 'res_charrnet_pristine.json', 'res_charrnet_nlos.json']:
    res.update(json.load(open(f))['results'])
mp = [t for t in ['nLOS 10', 'LOS 10', 'nLOS 30', 'LOS 100', 'nLOS 400', 'LOS 400']]
ours = {k: np.mean([100 * res[k]['test'][t] for t in mp]) for k in res}
ours_pr = {k: 100 * res[k]['test']['Pristine'] for k in res}
paper = {'baseline|Pristine': 9.5, 'baseline|nLOS 200': 76.7, 'charrnet|Pristine': 24.6, 'charrnet|nLOS 200': 84.3}

# ------------------------------------------------------------------ layout
fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1], width_ratios=[1.05, 1.15, 1.1],
                      hspace=0.36, wspace=0.32, left=0.05, right=0.985, top=0.92, bottom=0.08)
fig.suptitle('ChaRRNet: making RF fingerprints robust to multipath by building the channel '
             'symmetry into the network', fontsize=13, fontweight='bold', x=0.05, ha='left')

# ---- (a) pipeline -------------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.axis('off'); ax.set_xlim(-1.5, 100); ax.set_ylim(0, 20)
ax.text(0, 20.5, 'a   Pipeline', fontsize=11, fontweight='bold', va='top')
boxes = [('IQ burst\n+ multipath', 'r = h * x + n', '#e4e7eb'),
         ('Kaiser STFT', 'windows share\nthe same H(f)', '#e4e7eb'),
         ('Log-polar map', r'$re^{i\theta}\!\to(\ln r,\theta)$' + '\nchannel = shift', '#d9e8fb'),
         ('Equivariant layer', 'weighted Fréchet\nmean over windows', '#d9e8fb'),
         ('Invariant layer', 'distance of each\nwindow to the mean', '#fde5cf'),
         ('Real 1D CNN', 'device ID\n(65 classes)', '#e4e7eb')]
w, gap, y0 = 13.2, 3.6, 2.5
for i, (t, s, c) in enumerate(boxes):
    x0 = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, y0), w, 12.5, boxstyle='round,pad=0.3,rounding_size=1.2',
                                fc=c, ec=DARK, lw=0.8))
    ax.text(x0 + w / 2, y0 + 9.6, t, ha='center', va='center', fontweight='bold', fontsize=9.5)
    ax.text(x0 + w / 2, y0 + 4.2, s, ha='center', va='center', fontsize=8.3, color=DARK)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x0 + w + 0.5, y0 + 6.2), (x0 + w + gap - 0.5, y0 + 6.2),
                                     arrowstyle='-|>', mutation_scale=12, color=DARK))
ax.annotate('', xy=(2 * (w + gap) - 0.2, 0.9), xytext=(5 * (w + gap) - gap + 0.2, 0.9),
            arrowprops=dict(arrowstyle='-', color=BLUE, lw=1.4))
ax.text(3.5 * (w + gap) - gap / 2, -0.6, 'operations on the manifold ℝ⁺ × U(1), per frequency bin',
        ha='center', va='top', color=BLUE, fontsize=8.5)

# ---- (b) channel acts per frequency bin ----------------------------
gb = gs[1, 0].subgridspec(1, 2, width_ratios=[1, 0.32], wspace=0.06)
axs = fig.add_subplot(gb[0]); axh = fig.add_subplot(gb[1], sharey=axs)
spec = 20 * np.log10(np.abs(S1) + 1e-6).T
axs.imshow(spec, aspect='auto', origin='lower', cmap='Greys', vmin=spec.max() - 45,
           extent=[0, S1.shape[0], fk[0], fk[-1]])
axs.axhline(fk[F0], color=ORANGE, lw=1.6)
axs.set_xlabel('STFT window'); axs.set_ylabel('Frequency (MHz)')
axs.set_title('b   A multipath channel multiplies every\n     window by the same H(f)', loc='left',
              fontweight='bold', fontsize=10)
axs.text(1, fk[F0] + 0.9, f'bin f₀', color=ORANGE, fontsize=8.5, fontweight='bold', bbox=dict(fc='w', ec='none', pad=1))
axh.plot(20 * np.log10(np.abs(H)), fk, color=DARK, lw=1.1)
axh.plot(20 * np.log10(np.abs(H[F0])), fk[F0], 'o', color=ORANGE)
axh.set_xlabel('|H(f)| (dB)'); plt.setp(axh.get_yticklabels(), visible=False)
axh.set_ylim(-10, 10); axs.set_ylim(-10, 10)

# ---- (c) tangent-space picture -------------------------------------
axc = fig.add_subplot(gs[1, 1])
z0, z1 = S0[:, F0], S1[:, F0]
cand = z0[4:20]                               # windows whose phases cluster, for clarity
cm = np.angle(np.mean(np.exp(1j * np.angle(cand))))
dth = (np.angle(cand) - cm + np.pi) % (2 * np.pi) - np.pi
win = np.argsort(np.abs(dth))[:6]
p0 = np.c_[cm + dth[win], np.log(np.abs(cand[win]))]
p1 = p0 + [np.angle(H[F0]), np.log(np.abs(H[F0]))]      # channel = translation
m0, m1 = p0.mean(0), p1.mean(0)              # (uniform-weight) Fréchet means
for p, m, c, lab in [(p0, m0, BLUE, 'clean windows'), (p1, m1, ORANGE, 'after channel')]:
    for q in p:
        axc.plot([q[0], m[0]], [q[1], m[1]], color=c, lw=0.8, ls=':', alpha=0.9)
    axc.scatter(p[:, 0], p[:, 1], s=26, color=c, zorder=3, label=lab)
    axc.scatter(*m, marker='*', s=220, color=c, ec='k', lw=0.6, zorder=4)
axc.annotate('', xy=m1, xytext=m0, arrowprops=dict(arrowstyle='-|>', lw=2, color=DARK,
                                                   shrinkA=9, shrinkB=9))
mid = (m0 + m1) / 2
axc.text(mid[0] + 0.12, mid[1], 'channel: shift by\n(∠H(f₀), ln|H(f₀)|)', fontsize=8.5, color=DARK, va='center')
axc.set_xlabel('phase θ (rad)'); axc.set_ylabel('log-magnitude ln r')
axc.set_title('c   In log-polar coordinates the channel is a pure\n     translation, so means move with '
              'it and distances do not', loc='left', fontweight='bold', fontsize=10)
axc.scatter([], [], marker='*', s=120, color='w', ec='k', label='Fréchet mean (equivariant)')
axc.plot([], [], ls=':', color='k', label='distance to mean (invariant)')
axc.legend(fontsize=7.8, loc='lower right', frameon=False)
axc.margins(0.12)
axc.set_xlim(p1[:, 0].min() - 0.35, p0[:, 0].max() + 1.5)

# ---- (d) main result -----------------------------------------------
axd = fig.add_subplot(gs[1, 2])
keys = ['baseline|Pristine', 'charrnet|Pristine', 'baseline|nLOS 200', 'charrnet|nLOS 200']
xpos = np.array([0, 1, 2.6, 3.6])
cols = [GREY, BLUE, GREY, BLUE]
bars = axd.bar(xpos, [ours[k] for k in keys], 0.8, color=cols, ec='k', lw=0.5)
for b, k in zip(bars, keys):
    axd.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f'{ours[k]:.0f}%', ha='center',
             fontweight='bold', fontsize=9)
axd.scatter(xpos + 0.3, [paper[k] for k in keys], marker='D', s=34, color='w', ec='k', zorder=3,
            label='paper (Table I)')
axd.scatter(xpos, [ours_pr[k] for k in keys], marker='_', s=260, color=ORANGE, lw=2.5, zorder=3,
            label='same model, pristine test set')
axd.set_xticks(xpos, ['Baseline', 'ChaRRNet'] * 2)
for xc, t in [(0.5, 'trained without\nmultipath'), (3.1, 'trained with\nnLOS-200 multipath')]:
    axd.text(xc, -21, t, ha='center', va='top', fontsize=8.5, color=DARK)
axd.set_ylim(0, 122); axd.set_yticks(range(0, 101, 20)); axd.set_ylabel('Top-1 accuracy on unseen channels (%)')
axd.set_title('d   Accuracy on 6 unseen multipath channel\n     models (65 devices, mean)', loc='left',
              fontweight='bold', fontsize=10)
axd.legend(fontsize=7.8, loc='upper right', frameon=False)
axd.annotate('', xy=(1, ours['charrnet|Pristine'] + 9), xytext=(0, ours['baseline|Pristine'] + 9),
             arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.3, connectionstyle='arc3,rad=-0.35'))
axd.text(0.5, ours['charrnet|Pristine'] + 21, f"×{ours['charrnet|Pristine'] / ours['baseline|Pristine']:.1f}",
         ha='center', color=BLUE, fontweight='bold')
axd.text(3.1, 95,
         f"+{ours['charrnet|nLOS 200'] - ours['baseline|nLOS 200']:.0f} pts", ha='center', color=BLUE,
         fontweight='bold')

fig.text(0.05, 0.012, 'NumPy reproduction of Brown, Mattei & Draganov, arXiv:2105.03568 '
         '(simulated 802.11g, independent I/Q IIR fingerprints, reflector-geometry channels). '
         'Bars: this reproduction; diamonds: paper.', fontsize=7.5, color='#616e7c')
plt.savefig('charrnet_overview.png', dpi=200)
plt.savefig('charrnet_overview.pdf')
