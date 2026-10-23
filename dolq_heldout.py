"""DoLQ benchmark: does a held-out initial condition expose structures that pass the NMSE test?

For each system we integrate every candidate ODE from (a) the training initial condition and
(b) the second initial condition listed in the benchmark spec (never used for fitting), over
the in-distribution time window, and compare with the ground truth.
Candidates: PySR (3 seeds, expressions saved at 4 significant digits), dense SINDy (refitted,
full precision), closed-book Llama-3.1-8B guess with the best training fit (full precision).
"""
import json, glob, os, re
import numpy as np, sympy as sp, pandas as pd
from scipy.integrate import solve_ivp
from dolq_baselines import load, fit_sindy

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-12


def nmse(a, b):
    return float(((a - b) ** 2).sum() / ((a ** 2).sum() + EPS))


def integrate(fs, x0, t):
    nc = [0]
    def rhs(tt, x):
        nc[0] += 1
        if nc[0] > 200000:
            raise RuntimeError("budget")
        return np.array([np.broadcast_to(f(*x), ()).item() for f in fs], dtype=float)
    try:
        sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
        if sol.success and sol.y.shape[1] == len(t):
            return sol.y.T
    except Exception:
        pass
    return None


def score(fs, truth_traj, x0, t):
    y = integrate(fs, x0, t)
    if y is None or not np.isfinite(y).all():
        return float("inf"), None
    return max(nmse(truth_traj[:, j], y[:, j]) for j in range(truth_traj.shape[1])), y


def main():
    cb = json.load(open(f"{HERE}/results_llmsr_rerun/closedbook_llama8b.json"))
    out = {}; traj = {}
    for pid in range(1, 9):
        spec, tr, te_id, _ = load(pid); d = spec["dimension"]
        xs = sp.symbols(f"x0:{d}")
        loc = {f"x{i}": xs[i] for i in range(d)}; loc.update({k: v for k, v in spec["parameters"].items()})
        truth = [sp.lambdify(xs, sp.sympify(e.replace("np.", ""), locals=loc), "numpy") for e in spec["equation"]["python"]]
        t = te_id["t"].values
        ics = [np.array(te_id[[f"x{j}" for j in range(d)]].values[0]), np.array(spec["initial_conditions"][1], float)]
        assert np.allclose(ics[0], spec["initial_conditions"][0], atol=1e-6), pid
        T = [integrate(truth, ic, t) for ic in ics]
        assert all(x is not None for x in T)
        cands = {}
        for s in (0, 1, 2):
            r = json.load(open(f"{HERE}/results/pysr_s{s}_n1000_m30_P{pid}.json"))
            cands[f"pysr_s{s}"] = [sp.lambdify(xs, sp.sympify(e, locals=loc), "numpy") for e in r["exprs"]]
        X = tr[[f"x{j}" for j in range(d)]].values; Y = tr[[f"x{j}_t" for j in range(d)]].values
        cands["sindy"] = [sp.lambdify(xs, e, "numpy") for e in fit_sindy(X, Y, tr["t"].values)]
        ok = [r for r in cb if r["pid"] == pid and "intID" in r]
        if ok:
            b = min(ok, key=lambda r: r["train_cost"]); c = b["consts"]
            def mk(e, c=c):
                e = re.sub(r"c(\d+)", lambda m: repr(c[int(m.group(1))]), e)
                return sp.lambdify(xs, sp.sympify(e.replace("np.", ""), locals={f"x{i}": xs[i] for i in range(d)}), "numpy")
            cands["closedbook"] = [mk(e) for e in b["eqs"]]
        res = {}
        for name, fs in cands.items():
            e0, y0 = score(fs, T[0], ics[0], t); e1, y1 = score(fs, T[1], ics[1], t)
            res[name] = dict(train_ic=e0, heldout_ic=e1)
            if pid == 4 and name in ("closedbook", "sindy"):
                traj[name] = [None if y0 is None else y0.tolist(), None if y1 is None else y1.tolist()]
        if pid == 4:
            traj["truth"] = [T[0].tolist(), T[1].tolist()]; traj["t"] = t.tolist()
        out[pid] = res
        print(f"P{pid}: " + "  ".join(f"{k}: {v['train_ic']:.1e}/{v['heldout_ic']:.1e}" for k, v in res.items()), flush=True)
    json.dump(out, open(f"{HERE}/results_llmsr_rerun/dolq_heldout.json", "w"), indent=1)
    json.dump(traj, open(f"{HERE}/results_llmsr_rerun/dolq_traj_P4.json", "w"))


if __name__ == "__main__":
    main()
