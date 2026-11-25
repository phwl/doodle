# ChaRRNet reproduction (simulation)

NumPy reimplementation of *Brown, Mattei & Draganov, "ChaRRNets: Channel Robust
Representation Networks for RF Fingerprinting", arXiv:2105.03568* — the
simulated experiments only (Table I and Fig. 2). The real-world WiFi/ADS-B
datasets behind Tables II–III are not public.

## Files

Start with **`charrnet_tutorial.ipynb`**, which explains the implementation from first principles and ends by building the PowerPoint figure. Set `FULL = True` in its first cell for the complete reproduction (about 45 min) or leave it `False` for a 5-minute smoke test.

| File | Contents |
|---|---|
| `sim.py` | 802.11a/g OFDM bursts (STF + LTF + 4 QPSK symbols, 640 samples at 20 Msps); 65-device population with independent I/Q IIR low-pass fingerprints; reflector-geometry multipath channels (LOS/nLOS, N reflectors, attenuation U[-15,-5] dB, jittered nominal positions); AWGN + CFO augmentation |
| `ag.py` | Minimal reverse-mode autograd + Adam (PyTorch wasn't available) |
| `models.py` | Baseline complex-valued CNN, and ChaRRNet: Kaiser STFT → log-polar map onto R+×U(1) → wFM **equivariant** layer → manifold-distance **invariant** layer → real 1D CNN |
| `train.py` | Table I experiment |
| `fig2.py` | Fig. 2 experiment (equivariance/invariance under ideal vs physical channel, rect vs Kaiser window) |
| `report.py` | Builds `table1.md` and `table1_comparison.png` |
| `fig_overview.py`, `export_slide_data.py`, `build_slide.js`, `apply_theme.js` | Overview figure (PNG/PDF) and the native-shape PowerPoint slide (`SCALE=0.6 node build_slide.js`) |
| `gradcheck.py` | Finite-difference check of all gradients (max relative error < 5e-5) |

## Running

```bash
python3 gradcheck.py
python3 fig2.py
python3 train.py --models baseline --epochs 50 --pole_sigma 0.1 --train_sets Pristine  --out res_bl_p.json
python3 train.py --models baseline --epochs 50 --pole_sigma 0.1 --train_sets "nLOS 200" --out res_bl_n.json
python3 train.py --models charrnet --epochs 50 --pole_sigma 0.1 --train_sets Pristine  --out res_charrnet_pristine.json
python3 train.py --models charrnet --epochs 50 --pole_sigma 0.1 --train_sets "nLOS 200" --out res_charrnet_nlos.json
python3 report.py
```

On 2 CPU cores this takes about 15 min per ChaRRNet run and 6 min per baseline run.

## Results — Table I (ours, **bold**; paper in brackets)

65 devices, 100 training bursts/device (+10 for validation), 20 test bursts/device per test set, SNR U[15,30] dB, CFO ±40 kHz.

| Test set | Baseline / Pristine | Baseline / nLOS200 | ChaRRNet / Pristine | ChaRRNet / nLOS200 |
|---|---|---|---|---|
| Pristine | **98.8** (80.4) | **34.3** (23.5) | **79.1** (85.5) | **58.2** (24.1) |
| nLOS 10 | **10.2** (11.7) | **51.2** (76.7) | **35.2** (20.9) | **68.8** (83.9) |
| LOS 10 | **8.8** (5.5) | **47.8** (76.5) | **35.6** (23.6) | **65.4** (84.0) |
| nLOS 30 | **8.9** (11.8) | **49.2** (78.4) | **35.2** (26.8) | **67.4** (84.5) |
| LOS 100 | **10.5** (5.4) | **48.0** (70.5) | **36.9** (25.6) | **64.9** (83.0) |
| nLOS 400 | **11.5** (10.8) | **49.5** (78.6) | **34.3** (25.4) | **65.1** (85.1) |
| LOS 400 | **8.1** (11.5) | **46.8** (79.3) | **34.7** (25.4) | **66.2** (85.3) |
| *Mean of 6 multipath sets* | **9.7** (9.5) | **48.7** (76.7) | **35.3** (24.6) | **66.3** (84.3) |

What reproduces:
* When trained on pristine bursts, the baseline collapses to roughly 10% on every multipath set, almost exactly as in the paper.
* ChaRRNet degrades much less: 35% versus 10% here, and 25% versus 10% in the paper.
* When trained on nLOS200, ChaRRNet beats the baseline on every multipath set, and accuracy is flat across channel types. Here the gap is +18 points; in the paper it is +8.
* Training with channels is what drives generalization for both models, which is the paper's argument for channel augmentation.

What differs:
* Our absolute accuracies with channel training are lower (49%/66% against the paper's 77%/84%). Our models are 13–26× smaller than the paper's (188k and 314k parameters vs 5M and 4M), and the baseline overfits the channel-trained set (train loss ≈ 0.04, validation ≈ 50%).
* With pristine training, our baseline beats ChaRRNet on pristine test data (98.8 vs 79.1), whereas the paper has ChaRRNet ahead (85.5 vs 80.4). The invariant layer throws away absolute spectral shape, and our fingerprints are fairly strong.
* The paper reports that nLOS200-trained models drop to about 24% on pristine test data. Ours drop to 34% (baseline) and 58% (ChaRRNet).

## Results — Fig. 2

Median relative error over 200 bursts (see `fig2_equivariance.png`):

| | Ideal action | Physical, rect window | Physical, Kaiser |
|---|---|---|---|
| Equivariant layer, ‖f(Tx) − T f(x)‖/‖T f(x)‖ | 0.005 | 0.57 | 0.30 |
| Invariant layer, ‖f(Tx) − f(x)‖/‖f(x)‖ | 0.004 | 0.38 | 0.30 |
| Baseline 1st complex-conv layer (+CReLU) | — | 1.18 | — |

The layers are exactly equivariant and invariant under the ideal action; the residual comes from the log ε. Boundary effects break this under a physical channel, and the Kaiser window roughly halves the equivariance error, as the paper claims. Even so, the physical-channel change in ChaRRNet features (0.30) is about the same size as the change from swapping transmitters (0.28). For the baseline, the channel change is about 3× the device change (1.18 vs 0.43).

## Implementation choices (paper unspecified) and deviations

1. **Fingerprints must be widely linear.** If the IIR fingerprint were one complex LTI filter, it would multiply each frequency bin exactly as the channel does, and the invariant layer would erase it. Following the paper's Sec. II-A, each device therefore filters **I and Q with independently perturbed** 4th-order Butterworth filters (cutoff 0.85·Nyquist, pole perturbation σ = 0.1, image rejection about −25 dB). This IQ mismatch survives the per-bin channel ratio.
2. **GNU Radio replaced by NumPy.** The 802.11g preamble and symbols are standard; scrambling, coding and the SIGNAL field are omitted.
3. **Channel geometry.** Tx and Rx are 15 m apart, with reflectors in a ±25 m box. Each configuration has fixed nominal positions with 0.5 m jitter per burst, and the channel is normalized to unit energy. RMS delay spread is about 1 sample at 20 Msps.
4. **Training set name.** Table I's header says "LOS200", but the text says nLOS200. We used nLOS200.
5. **STFT.** 128-point Kaiser window (β = 8), hop 16, giving 33 windows × 128 bins. With 64-point windows, the Kaiser window did not reduce boundary error in our tests.
6. **Equivariant layer:** 8 output channels, a kernel of 4 windows and stride 2, with convex (softmax) weights so the wFM stays equivariant. **Invariant layer:** one learned wFM over 8 channels × 4 windows (stride 4), giving distances `sqrt(ln²(r'/r) + wrap(θ'−θ)²)`. The angle difference is wrapped to [−π, π). The resulting 96×128 real map feeds a 1D CNN along frequency (64-64 channels, then dense 128 → 65).
7. **Baseline:** three complex conv layers (16/32/32 channels, kernels 7/5/5) with CReLU and max-pooling, then a dense head. The paper only says it's a larger version of Mattei et al. 2019.
8. **Training:** 50 epochs of Adam (lr 1e-3, decayed ×0.2 at 70%, wd 1e-4), keeping the checkpoint with the best in-distribution validation accuracy. Each model was trained from one seed, so expect a few points of run-to-run variation.
