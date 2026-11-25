"""Reproduce Table I: train Baseline / ChaRRNet on {Pristine, nLOS200},
test on 7 channel configurations."""
import argparse, json, time
import numpy as np
import ag, models, sim


def batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def predict(model, x, bs=256):
    return np.concatenate([model(x[i:i + bs]).data.argmax(1) for i in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--devices', type=int, default=65)
    ap.add_argument('--train_per_dev', type=int, default=100)
    ap.add_argument('--test_per_dev', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--pole_sigma', type=float, default=0.1)
    ap.add_argument('--models', default='baseline,charrnet')
    ap.add_argument('--train_sets', default='Pristine,nLOS 200')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='base', help="ChaRRNet variant: base | deep | deep_gt")
    ap.add_argument('--out', default='results.json')
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pop = sim.DevicePopulation(a.devices, rng, pole_sigma=a.pole_sigma)
    cfgs = sim.channel_configs()
    t0 = time.time()
    tests = {k: sim.make_dataset(pop, a.test_per_dev, ch, rng) for k, ch in cfgs.items()}
    trains = {}
    for name in a.train_sets.split(','):
        ch = None if name == 'Pristine' else sim.TRAIN_NLOS200
        trains[name] = (sim.make_dataset(pop, a.train_per_dev, ch, rng),
                        sim.make_dataset(pop, max(a.train_per_dev // 10, 5), ch, rng))  # val
    print(f'data generated in {time.time() - t0:.0f}s', flush=True)

    results = {}
    for mname in a.models.split(','):
        M = {'baseline': models.BaselineCNN, 'charrnet': models.ChaRRNet}[mname]
        test_x = {k: M.prep(X) for k, (X, _) in tests.items()}
        for tname, ((Xtr, ytr), (Xva, yva)) in trains.items():
            mrng = np.random.default_rng(a.seed + 1)
            model = M(a.devices, mrng, variant=a.variant) if mname == 'charrnet' else M(a.devices, mrng)
            n_par = sum(p.data.size for p in model.params())
            opt = ag.Adam(model.params(), lr=a.lr, wd=1e-4)
            xtr, xva = M.prep(Xtr), M.prep(Xva)
            best, best_state = -1, None
            for ep in range(a.epochs):
                t1 = time.time(); losses = []
                if ep == int(0.7 * a.epochs):
                    opt.lr *= 0.2
                for b in batches(len(xtr), 64, mrng):
                    L = ag.cross_entropy(model(xtr[b]), ytr[b]); L.backward(); opt.step()
                    losses.append(float(L.data))
                va = (predict(model, xva) == yva).mean()
                if va > best:
                    best, best_state = va, model.state()
                print(f'{mname:8s} train={tname:9s} ep{ep:2d} loss={np.mean(losses):.3f} '
                      f'val={va:.3f} ({time.time() - t1:.0f}s)', flush=True)
            for k, v in best_state.items():
                getattr(model, k).data = v
            accs = {k: float((predict(model, test_x[k]) == tests[k][1]).mean())
                    for k in tests}
            results[f'{mname}|{tname}'] = dict(params=int(n_par), val=float(best), test=accs)
            print(mname, tname, json.dumps({k: round(100 * v, 1) for k, v in accs.items()}), flush=True)
            json.dump(dict(args=vars(a), results=results), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
