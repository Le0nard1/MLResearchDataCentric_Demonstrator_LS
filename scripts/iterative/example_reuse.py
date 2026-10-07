"""
Minimal example: iterated weakspot curation with focus + coverage on your own data.

Replace ``make_data`` by your reserve (features, labels) and initial training set. The
loop is the one of the paper, reduced to its essentials: per round, measure the errors
on the remaining reserve, smooth them into the error landscape, set the focused share by
the size-adaptive schedule, select, and retrain on the new points plus a replay sample.

    python -m scripts.iterative.example_reuse
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from scripts.dataselect.budget import landscape
from scripts.iterative.selection import held_out_measures, select


def make_data(rs, n=4500):
    """Toy task: two Gaussian bumps on the unit square; the corner region x0, x1 > 0.7
    is missing from the initial set, so the initial model is weak there."""
    X = rs.rand(n, 2)
    y = (np.exp(-((X - 0.3) ** 2).sum(1) / 0.02) + 2 * np.exp(-((X - 0.8) ** 2).sum(1) / 0.01)
         + 0.05 * rs.randn(n))
    corner = (X > 0.7).all(1)
    init = np.flatnonzero(~corner)[:500]
    reserve = np.setdiff1d(np.arange(n), init)
    return X, y, init, reserve


def main(rounds=8, n=200, m=200, alpha=0.5, epochs=200, seed=0):
    rs = np.random.RandomState(seed)
    X, y, init, reserve = make_data(rs)
    scaler = StandardScaler().fit(X[init])           # frozen input standardisation
    U = scaler.transform(X)
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True, random_state=seed,
                         warm_start=True).fit(U[init], y[init])
    used, avail = list(init), np.ones(len(X), bool)
    avail[init] = False
    z_1 = None
    for t in range(1, rounds + 1):
        cand = np.flatnonzero(avail)
        e = np.abs(y[cand] - model.predict(U[cand]))  # errors on the remaining reserve
        land = landscape(U[cand], e, k=10)              # kNN error landscape
        z_t = held_out_measures(U[cand], e, rs)["size"]  # weakspot size, no ground truth
        z_1 = z_t if z_1 is None else z_1
        alpha_t = alpha * min(1.0, z_t / z_1) ** 2      # size-adaptive schedule, g(r) = r^2
        pick = select("region_adapt_kc", n, int(round(alpha_t * n)), U[cand], e, land,
                      U[used], rs)                      # focus + coverage
        new = cand[pick]
        replay = rs.choice(used, min(m, len(used)), replace=False)
        avail[new] = False
        used.extend(new.tolist())
        model.set_params(max_iter=epochs, early_stopping=False, n_iter_no_change=epochs + 1)
        model.best_loss_ = np.inf                       # warm start after early stopping
        model.fit(U[np.concatenate([new, replay])], y[np.concatenate([new, replay])])
        print(f"round {t}: alpha_t = {alpha_t:.2f}, weakspot size = {z_t:.3f}, "
              f"selected in the corner: {(X[new] > 0.7).all(1).sum()} of {n}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
