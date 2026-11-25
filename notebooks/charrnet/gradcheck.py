"""Finite-difference check of the NumPy autograd through both models (float64)."""
import numpy as np, ag, models, sim

rng = np.random.default_rng(1)
pop = sim.DevicePopulation(5, rng)
X, y = sim.make_dataset(pop, 1, None, rng)
for M in [models.BaselineCNN, models.ChaRRNet]:
    m = M(5, np.random.default_rng(0)); x = M.prep(X).astype(np.float64)
    for p in m.params():
        p.data = p.data.astype(np.float64)
    ag.cross_entropy(m(x), y).backward()
    worst = 0.0
    for name in m.state():
        P = getattr(m, name); g = P.grad.copy(); P.grad = None
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in P.shape); e = 1e-6
            P.data[i] += e; lp = ag.cross_entropy(m(x), y).data
            P.data[i] -= 2 * e; lm = ag.cross_entropy(m(x), y).data
            P.data[i] += e
            num = (lp - lm) / (2 * e)
            if abs(num) > 1e-8:
                worst = max(worst, abs(num - g[i]) / (abs(num) + abs(g[i])))
    print(f'{M.__name__}: max relative gradient error {worst:.2e}')
