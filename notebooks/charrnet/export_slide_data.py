"""Export the numbers behind charrnet_overview.png for the PowerPoint build."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import models, sim
rng = np.random.default_rng(11)
pop = sim.DevicePopulation(65, rng, pole_sigma=0.1)
x = pop.apply(sim.wifi_burst(rng), 3)
h = sim.ReflectorChannel(30, False, seed=30).draw(np.random.default_rng(0))
H = np.fft.fftshift(np.fft.fft(h, models.NFFT)); S0 = models.stft(x[None])[0]; S1 = S0 * H; F0 = 22
fk = (np.arange(models.NFFT) - models.NFFT // 2) * 20 / models.NFFT
spec = 20 * np.log10(np.abs(S1) + 1e-6).T
plt.imsave('slide_spectrogram.png', spec[::-1], cmap='Greys', vmin=spec.max() - 45, vmax=spec.max())
cand = S0[4:20, F0]; cm = np.angle(np.mean(np.exp(1j * np.angle(cand))))
dth = (np.angle(cand) - cm + np.pi) % (2 * np.pi) - np.pi; win = np.argsort(np.abs(dth))[:6]
p0 = np.c_[cm + dth[win], np.log(np.abs(cand[win]))]; p1 = p0 + [np.angle(H[F0]), np.log(np.abs(H[F0]))]
res = {}
for f in ['res_bl_p.json', 'res_bl_n.json', 'res_charrnet_pristine.json', 'res_charrnet_nlos.json']:
    res.update(json.load(open(f))['results'])
mp = ['nLOS 10', 'LOS 10', 'nLOS 30', 'LOS 100', 'nLOS 400', 'LOS 400']
keys = ['baseline|Pristine', 'charrnet|Pristine', 'baseline|nLOS 200', 'charrnet|nLOS 200']
json.dump(dict(fk=fk.tolist(), HdB=(20 * np.log10(np.abs(H))).tolist(), f0=float(fk[F0]),
               HdB0=float(20 * np.log10(abs(H[F0]))), nwin=int(S1.shape[0]),
               p0=p0.tolist(), p1=p1.tolist(),
               ours=[float(np.mean([100 * res[k]['test'][t] for t in mp])) for k in keys],
               pristine=[100 * res[k]['test']['Pristine'] for k in keys],
               paper=[9.5, 24.6, 76.7, 84.3]), open('slide_data.json', 'w'), indent=1)
