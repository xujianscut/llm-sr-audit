"""Automatic term test for DoLQ candidates (replaces judgement by reading expressions).

Rule, applied to every right-hand side of every candidate:
 1. Pruning: repeatedly set to zero the numeric constant (smallest magnitude first) whose removal
    changes the right-hand side by less than delta in relative RMS over the training points;
    removals that make the right-hand side non-finite are rejected.
 2. Snapping: constants within 1e-4 of a nonzero multiple of pi/2 become exact (so that
    cos(x + 1.5708) simplifies to -sin(x)), constants within 1e-3 of an integer become integers.
 3. Skeleton: expand into additive terms, drop each term's leading coefficient, replace every
    remaining non-integer number by its sign.
Structure is recovered when the skeleton sets of all right-hand sides equal those of the truth.
Stability is checked separately on the unpruned model: integration from both the training and
the held-out initial condition must succeed (dolq_heldout.json)."""
import json, os, re
import numpy as np, sympy as sp
from dolq_baselines import load, fit_sindy

HERE = os.path.dirname(os.path.abspath(__file__))
DELTAS = (1e-3, 1e-2)


def rel_change(f0, f1, xs, X):
    a = np.broadcast_to(sp.lambdify(xs, f0, "numpy")(*X.T), (len(X),))
    with np.errstate(all="ignore"):
        b = np.broadcast_to(sp.lambdify(xs, f1, "numpy")(*X.T), (len(X),))
    if not np.all(np.isfinite(b)):
        return np.inf
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-300))


def prune(e, xs, X, delta):
    while True:
        consts = sorted({c for c in e.atoms(sp.Float) if c != 0}, key=lambda c: abs(float(c)))
        for c in consts:
            e1 = e.xreplace({c: sp.Integer(0)})
            if e1 != e and rel_change(e, e1, xs, X) < delta:
                e = e1; break
        else:
            return e


def snap(e):
    rep = {}
    for c in e.atoms(sp.Float):
        v = float(c); k = round(v / (np.pi / 2))
        if k != 0 and abs(v - k * np.pi / 2) < 1e-4:
            rep[c] = k * sp.pi / 2
        elif abs(v - round(v)) < 1e-3:
            rep[c] = sp.Integer(round(v))
    return sp.simplify(e.xreplace(rep))


def skeleton(e):
    out = set()
    for t in sp.Add.make_args(sp.expand(e)):
        _, m = t.as_coeff_Mul()
        m = m.xreplace({c: sp.sign(c) for c in m.atoms(sp.Float)})
        m = m.xreplace({c: sp.sign(c) for c in m.atoms(sp.Rational) if not c.is_Integer})
        if m != 0:
            out.add(sp.srepr(sp.simplify(m)))
    return frozenset(out)


def judge(exprs, truth, xs, X, delta):
    pruned = [snap(prune(e, xs, X, delta)) for e in exprs]
    return all(skeleton(p) == skeleton(t) for p, t in zip(pruned, truth)), [str(p) for p in pruned]


def main():
    cb = json.load(open(f"{HERE}/results_llmsr_rerun/closedbook_llama8b.json"))
    out = {}
    for pid in range(1, 9):
        spec, tr, _, _ = load(pid); d = spec["dimension"]
        xs = sp.symbols(f"x0:{d}"); loc = {f"x{i}": xs[i] for i in range(d)}
        X = tr[[f"x{j}" for j in range(d)]].values
        tl = dict(loc); tl.update({k: sp.Float(v) for k, v in spec["parameters"].items()})
        truth = [snap(sp.sympify(e.replace("np.", ""), locals=tl)) for e in spec["equation"]["python"]]
        cands = {f"pysr_s{s}": [sp.sympify(e, locals=loc) for e in json.load(open(f"{HERE}/results/pysr_s{s}_n1000_m30_P{pid}.json"))["exprs"]]
                 for s in (0, 1, 2)}
        cands["sindy"] = list(fit_sindy(X, tr[[f"x{j}_t" for j in range(d)]].values, tr["t"].values))
        ok = [r for r in cb if r["pid"] == pid and "intID" in r]
        if ok:
            b = min(ok, key=lambda r: r["train_cost"]); c = b["consts"]
            cands["closedbook"] = [sp.sympify(re.sub(r"c(\d+)", lambda m: repr(c[int(m.group(1))]), e).replace("np.", ""), locals=loc)
                                   for e in b["eqs"]]
        out[pid] = {}
        for k, ex in cands.items():
            out[pid][k] = {}
            for dl in DELTAS:
                rec, pr = judge(ex, truth, xs, X, dl)
                out[pid][k][f"{dl:g}"] = dict(recovered=rec, pruned=pr)
        print(f"P{pid} truth={[str(t) for t in truth]}")
        for k, v in out[pid].items():
            print(f"   {k:10s} " + " ".join(f"{dl}:{'OK' if w['recovered'] else '--'}" for dl, w in v.items()), v["0.01"]["pruned"] if k in ("sindy", "pysr_s0") and pid in (1, 8) else "")
    json.dump(out, open(f"{HERE}/results_llmsr_rerun/dolq_terms.json", "w"), indent=1)


if __name__ == "__main__":
    main()
