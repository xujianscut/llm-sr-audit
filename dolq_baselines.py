"""DoLQ (ICML 2026) 审计: 非 LLM 符号回归在 DoLQ 自带的 8 个 benchmark 上能拿几分.
协议照论文 App. A/C: 训练目标 = 数值导数 x_t (train.csv); 评测 = residual / integral NMSE,
NMSE_j = sum (x - x~)^2 / (sum x^2 + eps); 成功 = ID 积分 NMSE < 1e-3 (所有维度).
"""
import json, sys, time, glob, os
import numpy as np, pandas as pd, sympy as sp
from scipy.integrate import solve_ivp

DATA = os.path.join(os.path.dirname(__file__), "DoLQ", "data")
EPS = 1e-12


def load(pid):
    spec = json.load(open(f"{DATA}/json/ID_0{pid}.json"))
    d = spec["dimension"]
    base = f"{DATA}/{d}D/ID_0{pid}/ID_0{pid}"
    return spec, pd.read_csv(base + "_train.csv"), pd.read_csv(base + "_test_id.csv"), pd.read_csv(base + "_test_ood.csv")


def nmse(a, b):
    return float(((a - b) ** 2).sum() / ((a ** 2).sum() + EPS))


def evaluate(exprs, xs, tr, te_id, te_ood):
    """exprs: 每维一个 sympy 表达式(变量 x0..x{d-1})."""
    d = len(xs)
    fs = [sp.lambdify(xs, e, "numpy") for e in exprs]
    ncall = [0]
    def rhs(t, x):
        ncall[0] += 1
        if ncall[0] > 200000:                       # 刚性/发散方程: 判积分失败, 不让它空转
            raise RuntimeError("rhs budget exceeded")
        return np.array([np.broadcast_to(f(*x), ()).item() for f in fs], dtype=float)
    out = {}
    for name, df in [("ID", te_id), ("OOD", te_ood)]:
        X = df[[f"x{j}" for j in range(d)]].values
        res = []
        for j, f in enumerate(fs):
            pred = np.broadcast_to(np.asarray(f(*X.T), dtype=float), (len(X),))
            res.append(nmse(df[f"x{j}_t_gt"].values, pred))
        t = df["t"].values
        ncall[0] = 0
        try:
            sol = solve_ivp(rhs, (t[0], t[-1]), X[0], t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
            ok = sol.success and sol.y.shape[1] == len(t)
        except Exception:
            ok = False
        integ = [nmse(X[:, j], sol.y[j]) if ok else float("inf") for j in range(d)]
        out[name] = dict(residual=res, integral=integ)
    out["success_ID"] = bool(max(out["ID"]["integral"]) < 1e-3)
    out["success_OOD"] = bool(max(out["OOD"]["integral"]) < 1e-3)
    return out


def fit_pysr(X, Y, seed, niter, maxsize):
    from pysr import PySRRegressor
    exprs = []
    for j in range(Y.shape[1]):
        m = PySRRegressor(niterations=niter, binary_operators=["+", "-", "*", "/"],
                          unary_operators=["sin", "cos", "exp"], maxsize=maxsize,
                          random_state=seed, verbosity=0, progress=False,
                          temp_equation_file=True, model_selection="best")
        m.fit(X, Y[:, j], variable_names=[f"x{i}" for i in range(X.shape[1])])
        exprs.append(m.sympy())
    return exprs


def fit_sindy(X, Y, t):
    import pysindy as ps
    best = None
    for thr in [0.01, 0.03, 0.1, 0.3]:
        lib = ps.PolynomialLibrary(degree=3) + ps.FourierLibrary(n_frequencies=1)
        m = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=thr))
        m.fit(X, t=t, x_dot=Y)
        P = m.predict(X)
        k = np.count_nonzero(m.coefficients())
        n = len(X) * Y.shape[1]
        bic = n * np.log(((P - Y) ** 2).mean() + 1e-30) + k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, m)
    m = best[1]
    names = m.get_feature_names()
    xs = sp.symbols(f"x0:{X.shape[1]}")
    loc = {f"x{i}": xs[i] for i in range(X.shape[1])}
    loc.update(sin=sp.sin, cos=sp.cos)
    feats = [sp.sympify(n.replace(" ", "*").replace("^", "**"), locals=loc) for n in names]
    return [sum(c * f for c, f in zip(row, feats) if abs(c) > 0) for row in m.coefficients()]


def main():
    method = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    niter = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    maxsize = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    pids = [int(p) for p in sys.argv[5].split(",")] if len(sys.argv) > 5 else list(range(1, 9))
    os.makedirs("results", exist_ok=True)
    for pid in pids:
        spec, tr, te_id, te_ood = load(pid)
        d = spec["dimension"]
        X = tr[[f"x{j}" for j in range(d)]].values
        Y = tr[[f"x{j}_t" for j in range(d)]].values
        assert X.shape == (1000, d) and np.isfinite(Y).all()
        xs = sp.symbols(f"x0:{d}")
        t0 = time.time()
        if method == "pysr":
            exprs = fit_pysr(X, Y, seed, niter, maxsize)
            exprs = [e.subs({sp.Symbol(f"x{i}"): xs[i] for i in range(d)}) for e in exprs]
        else:
            exprs = fit_sindy(X, Y, tr["t"].values)
        wall = time.time() - t0
        json.dump(dict(pid=pid, exprs=[str(e) for e in exprs], wall_s=wall),
                  open(f"results/raw_{method}_s{seed}_n{niter}_P{pid}.json", "w"))
        ev = evaluate(exprs, xs, tr, te_id, te_ood)
        rec = dict(pid=pid, name=spec["name"], method=method, seed=seed, niter=niter, maxsize=maxsize,
                   wall_s=wall, exprs=[str(sp.N(e, 4)) for e in exprs], truth=spec["equation"]["python"],
                   params=spec["parameters"], **ev)
        tag = f"{method}_s{seed}_n{niter}_m{maxsize}" if method == "pysr" else method
        json.dump(rec, open(f"results/{tag}_P{pid}.json", "w"), indent=1)
        print(f"P{pid} {spec['name'][:40]:40s} {wall:6.0f}s succID={ev['success_ID']} succOOD={ev['success_OOD']} "
              f"intID={max(ev['ID']['integral']):.2e} intOOD={max(ev['OOD']['integral']):.2e}", flush=True)
        for j, e in enumerate(rec["exprs"]):
            print(f"     dx{j}/dt = {e}    | truth: {rec['truth'][j]}", flush=True)


if __name__ == "__main__":
    main()
