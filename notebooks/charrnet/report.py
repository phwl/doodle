"""Collect results into a Table-I-style markdown table and a figure."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = {  # Table I of the paper: Baseline(Pristine, LOS200), ChaRRNet(Pristine, LOS200)
    'Pristine': (80.4, 23.5, 85.5, 24.1), 'nLOS 10': (11.7, 76.7, 20.9, 83.9),
    'LOS 10': (5.5, 76.5, 23.6, 84.0), 'nLOS 30': (11.8, 78.4, 26.8, 84.5),
    'LOS 100': (5.4, 70.5, 25.6, 83.0), 'nLOS 400': (10.8, 78.6, 25.4, 85.1),
    'LOS 400': (11.5, 79.3, 25.4, 85.3)}

res = {}
for f in ['res_bl_p.json', 'res_bl_n.json', 'res_charrnet_pristine.json', 'res_charrnet_nlos.json']:
    res.update(json.load(open(f))['results'])
cols = ['baseline|Pristine', 'baseline|nLOS 200', 'charrnet|Pristine', 'charrnet|nLOS 200']

lines = ['| Test set | Baseline / Pristine | Baseline / nLOS200 | ChaRRNet / Pristine | ChaRRNet / nLOS200 |',
         '|---|---|---|---|---|']
for t, paper in PAPER.items():
    cells = [f"**{100 * res[c]['test'][t]:.1f}** ({p})" for c, p in zip(cols, paper)]
    lines.append(f'| {t} | ' + ' | '.join(cells) + ' |')
ch_tests = [t for t in PAPER if t != 'Pristine']
avg = [np.mean([100 * res[c]['test'][t] for t in ch_tests]) for c in cols]
pavg = [np.mean([PAPER[t][i] for t in ch_tests]) for i in range(4)]
lines.append('| *Mean of 6 multipath sets* | ' + ' | '.join(f'**{a:.1f}** ({p:.1f})' for a, p in zip(avg, pavg)) + ' |')
lines.append('| *Trainable params* | ' + ' | '.join(f"{res[c]['params']:,}" for c in cols) + ' |')
open('table1.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
tests = list(PAPER)
x = np.arange(len(tests))
for k, (title, src) in enumerate([('This reproduction (simulation)', 'ours'), ('Paper, Table I', 'paper')]):
    for i, (c, lab) in enumerate(zip(cols, ['Baseline, trained pristine', 'Baseline, trained nLOS200',
                                            'ChaRRNet, trained pristine', 'ChaRRNet, trained nLOS200'])):
        v = [100 * res[c]['test'][t] for t in tests] if src == 'ours' else [PAPER[t][i] for t in tests]
        ax[k].bar(x + (i - 1.5) * 0.2, v, 0.2, label=lab,
                  color=['#9aa5b1', '#52606d', '#f0b429', '#cb6e17'][i])
    ax[k].set_xticks(x, tests, fontsize=8)
    ax[k].set_title(title)
    ax[k].grid(axis='y', alpha=.3)
ax[0].set_ylabel('Top-1 accuracy (%)')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig('table1_comparison.png', dpi=130)
