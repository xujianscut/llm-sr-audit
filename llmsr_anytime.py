"""Anytime curve of an LLM-SR run: after n samples, take the program with the best training score
among the first n, refit its parameters on train (BFGS, as LLM-SR's evaluator does) and report
ID/OOD NMSE together with the wall-clock time at which sample n was written (file mtime)."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [25, 50, 100, 200, 400, 800, 1000, 1600, 2400, 3000]


def data_dir(ds):
    return f"{HERE}/LLM-SR/data/{ds}" if not ds.startswith("CRK") else f"{HERE}/crk_llmsr/data/{ds}"


def nmse(y, p):
    return float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def evaluate(fn_src, ds, score=None, restarts=30):
    """Refit as LLM-SR's evaluator does (BFGS from ones). If the refit does not reach the training
    loss recorded during search (-score), retry from seeded random starts and keep the best fit:
    a single BFGS run from a fixed start is numerically fragile across numpy/scipy versions."""
    tr, ti, to = [pd.read_csv(f"{data_dir(ds)}/{s}.csv") for s in ["train", "test_id", "test_ood"]]
    ns = {"np": np}
    exec(fn_src, ns)
    fn = [v for k, v in ns.items() if callable(v) and k != "np"][0]
    X, y = tr.values[:, :-1], tr.values[:, -1]
    cols = [X[:, i] for i in range(X.shape[1])]
    def loss(p):
        with np.errstate(all="ignore"):
            v = np.mean((fn(*cols, p) - y) ** 2)
        return v if np.isfinite(v) else 1e300
    best = minimize(loss, [1.0] * 10, method="BFGS")
    target = None if score is None else -score
    rng = np.random.default_rng(0); k = 0
    while target is not None and best.fun > target * (1 + 1e-3) + 1e-300 and k < restarts:
        r = minimize(loss, rng.normal(1.0, 1.0, 10), method="BFGS"); k += 1
        if r.fun < best.fun: best = r
    with np.errstate(all="ignore"):
        pr = lambda df: fn(*[df.values[:, i] for i in range(df.shape[1] - 1)], best.x)
        return nmse(ti.values[:, -1], pr(ti)), nmse(to.values[:, -1], pr(to)), float(best.fun), k, (target is None or best.fun <= target * (1 + 1e-3))


def anytime(args):
    run_dir, ds = args
    S = []
    for f in glob.glob(f"{run_dir}/samples/samples_*.json"):
        d = json.load(open(f))
        sc = d.get("score")
        sc = None if sc in (None, "None") else float(sc)
        S.append((int(d["sample_order"]), sc, d.get("function"), os.path.getmtime(f)))
    S.sort()
    t0 = S[0][3]; nmax = len(S)
    out = []; best = None; cache = {}
    for n in [b for b in BUDGETS if b <= nmax] + ([nmax] if nmax not in BUDGETS else []):
        cand = [s for s in S[:n] if s[1] is not None and s[2]]
        cand.sort(key=lambda s: -s[1])
        res = None
        for s in cand[:5]:
            if s[0] in cache:
                res = cache[s[0]]; break
            try:
                res = evaluate(s[2], ds, s[1]); cache[s[0]] = res; break
            except Exception:
                continue
        if res is None:
            continue
        out.append(dict(n=n, wall_s=S[n - 1][3] - t0, ID=res[0], OOD=res[1], refit_loss=res[2], restarts=res[3], matched=bool(res[4])))
    return dict(run=run_dir, ds=ds, curve=out)


if __name__ == "__main__":
    root, outfile = sys.argv[1], sys.argv[2]
    jobs = []
    for d in sorted(glob.glob(f"{root}/llama8b_*")):
        if not os.path.isdir(f"{d}/samples"):
            continue
        name = os.path.basename(d)[len("llama8b_"):]
        ds = name.rsplit("_r", 1)[0]
        n = len(glob.glob(f"{d}/samples/*.json"))
        target = 1000 if ds.startswith("CRK") else 3000
        if n < target:
            print("skip incomplete", d, n); continue
        jobs.append((d, ds))
    with Pool(10) as p:
        res = p.map(anytime, jobs)
    json.dump(res, open(outfile, "w"), indent=1)
    print("runs:", len(res))
